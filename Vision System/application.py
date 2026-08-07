"""Central runtime coordinator for reference setup, tracking, data, and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from dataset import DatasetSession
from force_model import ForceModel, ForcePrediction
from geometry import GeometryResult, extract_geometry
from processing import (
	AppConfig,
	calibrate_from_blob,
	cfg,
	create_red_mask,
	detect_circles,
	draw_blob_calibration_info,
	draw_circles,
	draw_measurement,
	find_blob_index_for_click,
	find_red_blob_data,
	sample_blob_hsv_near_point,
	sync_area_tuning_window_from_config,
)
from tracking import (
	PointGroup,
	PointStatus,
	PointTracker,
	RansacResult,
	ReferenceProfile,
	TrackedPoint,
	draw_tracking,
	initialise_reference,
	load_reference_profile,
	make_reference_profile,
	save_reference_profile,
)


class ApplicationMode(str, Enum):
	DIAGNOSTIC = "DIAGNOSTIC"
	REFERENCE_SETUP = "REFERENCE_SETUP"
	TRACKING = "TRACKING"
	DATA_COLLECTION = "DATA_COLLECTION"
	FORCE_MEASUREMENT = "FORCE_MEASUREMENT"


class ReferenceSetupMode(str, Enum):
	IDLE = "IDLE"
	ORIGIN_REQUIRED = "ORIGIN_REQUIRED"
	AWAITING_ORIGIN_CLICK = "AWAITING_ORIGIN_CLICK"
	READY_TO_PREVIEW = "READY_TO_PREVIEW"
	PROPOSAL_READY = "PROPOSAL_READY"


@dataclass
class DisplayOptions:
	show_points: bool = True
	show_lines: bool = True
	show_outliers: bool = True
	show_geometry: bool = False
	show_features: bool = False
	show_circles: bool = True
	show_status: bool = True
	show_warnings: bool = True
	show_mask_only: bool = False

	@classmethod
	def from_mapping(cls, values: dict[str, Any]) -> "DisplayOptions":
		return cls(**{
			name: bool(values.get(name, default))
			for name, default in (
				("show_points", True),
				("show_lines", True),
				("show_outliers", True),
				("show_geometry", False),
				("show_features", False),
				("show_circles", True),
				("show_status", True),
				("show_warnings", True),
				("show_mask_only", False),
			)
		})


@dataclass
class ApplicationState:
	"""Own all mutable state used by the camera loop and command surface."""

	config: AppConfig = field(default_factory=lambda: cfg)
	current_raw_frame: np.ndarray | None = None
	current_gray_frame: np.ndarray | None = None
	current_detections: list[dict[str, Any]] = field(default_factory=list)
	current_circles: list[tuple[int, int, int]] = field(default_factory=list)
	current_tracked_points: list[TrackedPoint] = field(default_factory=list)
	current_frame_number: int = 0
	current_timestamp: float = 0.0
	mode: ApplicationMode = ApplicationMode.DIAGNOSTIC
	display_options: DisplayOptions | None = None
	reference_setup_mode: ReferenceSetupMode = ReferenceSetupMode.IDLE
	reference_origin_selection_mode: bool = False
	origin_reacquisition_selection_mode: bool = False
	selected_origin_detection_index: int | None = None
	selected_origin_click: tuple[int, int] | None = None
	selected_origin_candidate_center: tuple[float, float] | None = None
	rigid_reference_detection_indices: tuple[int, ...] = ()
	rigid_reference_candidate_centers: tuple[tuple[float, float], ...] = ()
	proposed_ransac_result: RansacResult | None = None
	reference_proposal_detections: list[dict[str, Any]] = field(default_factory=list)
	reference_profile: ReferenceProfile | None = None
	tracker: PointTracker | None = None
	current_geometry: GeometryResult | None = None
	current_feature_names: list[str] = field(default_factory=list)
	dataset_session: DatasetSession | None = None
	known_reference_force: float | None = None
	pending_dataset_frames: int = 0
	pending_dataset_invalid_retries: int = 0
	loaded_force_model: ForceModel | None = None
	current_force_prediction: ForcePrediction | None = None
	force_measurement_enabled: bool = False
	calibration_mode: bool = False
	measure_mode: bool = False
	measure_points: list[tuple[int, int]] = field(default_factory=list)
	selected_calibration_blob: dict[str, Any] | None = None
	tracking_enabled: bool = False
	last_dataset_result: str = ""
	runtime_warnings: list[str] = field(default_factory=list)

	def __post_init__(self) -> None:
		if self.display_options is None:
			self.display_options = DisplayOptions.from_mapping(self.config.display)

	# Compatibility names used by the command surface.
	@property
	def tracker_state(self) -> PointTracker | None:
		return self.tracker

	@tracker_state.setter
	def tracker_state(self, value: PointTracker | None) -> None:
		self.tracker = value
		self.tracking_enabled = value is not None

	@property
	def ransac_result(self) -> RansacResult | None:
		return self.proposed_ransac_result

	@ransac_result.setter
	def ransac_result(self, value: RansacResult | None) -> None:
		self.proposed_ransac_result = value

	@property
	def feature_names(self) -> list[str]:
		return self.current_feature_names

	@property
	def loaded_model(self) -> ForceModel | None:
		return self.loaded_force_model

	@loaded_model.setter
	def loaded_model(self, value: ForceModel | None) -> None:
		self.loaded_force_model = value

	@property
	def predicted_force(self) -> ForcePrediction | None:
		return self.current_force_prediction

	def _profile_path(self, path: str | Path | None) -> Path:
		value = Path(path or self.config.reference.get("profile_path", "reference-profile.json"))
		return value if value.is_absolute() else Path(__file__).resolve().parent / value

	def _dataset_directory(self) -> Path:
		value = Path(self.config.dataset.get("directory", "datasets"))
		return value if value.is_absolute() else Path(__file__).resolve().parent / value

	def _tracking_config(self) -> dict[str, Any]:
		return {**self.config.tracking, **self.config.reference}

	def _reference_config(self) -> dict[str, Any]:
		return {
			**self.config.tracking,
			**self.config.reference,
			**self.config.reference_ransac,
			"geometry_reference_mode": self.config.geometry.get(
				"geometry_reference_mode", "translation_only"
			),
		}

	def reference_start(self, indices: tuple[int, ...] = ()) -> None:
		if not self.current_detections:
			raise RuntimeError("No red blobs are currently detected")
		normalised = tuple(sorted(set(int(index) for index in indices)))
		if any(index < 0 or index >= len(self.current_detections) for index in normalised):
			raise ValueError("A rigid-reference detection index is outside the current detections")
		self.rigid_reference_detection_indices = normalised
		self.rigid_reference_candidate_centers = tuple(
			tuple(float(value) for value in self.current_detections[index]["center"])
			for index in normalised
		)
		self.proposed_ransac_result = None
		self.reference_proposal_detections = []
		self.selected_origin_detection_index = None
		self.selected_origin_click = None
		self.selected_origin_candidate_center = None
		self.reference_origin_selection_mode = False
		self.origin_reacquisition_selection_mode = False
		self.reference_setup_mode = ReferenceSetupMode.ORIGIN_REQUIRED
		self.mode = ApplicationMode.REFERENCE_SETUP
		self.stop_dataset()
		self.stop_force_inference()
		self.stop_tracking()
		self.mode = ApplicationMode.REFERENCE_SETUP

	def begin_reference_origin_selection(self) -> None:
		if self.mode != ApplicationMode.REFERENCE_SETUP:
			raise RuntimeError("Run reference_start before selecting the origin")
		if (
			self.measure_mode
			or self.calibration_mode
			or self.origin_reacquisition_selection_mode
		):
			raise RuntimeError("Turn off the other mouse modes first")
		self.reference_origin_selection_mode = True
		self.reference_setup_mode = ReferenceSetupMode.AWAITING_ORIGIN_CLICK

	def clear_reference_origin(self) -> None:
		self.reference_origin_selection_mode = False
		self.selected_origin_detection_index = None
		self.selected_origin_click = None
		self.selected_origin_candidate_center = None
		self.proposed_ransac_result = None
		self.reference_proposal_detections = []
		if self.mode == ApplicationMode.REFERENCE_SETUP:
			self.reference_setup_mode = ReferenceSetupMode.ORIGIN_REQUIRED

	def begin_origin_reacquisition(self) -> None:
		if self.tracker is None or self.reference_profile is None:
			raise RuntimeError("Active permanent-ID tracking is required")
		if not self.tracker.origin_reacquisition_required:
			raise RuntimeError(
				"The reference origin is not awaiting manual reacquisition"
			)
		if (
			self.measure_mode
			or self.calibration_mode
			or self.reference_origin_selection_mode
		):
			raise RuntimeError("Turn off the other mouse modes first")
		self.origin_reacquisition_selection_mode = True

	def cancel_origin_reacquisition(self) -> None:
		self.origin_reacquisition_selection_mode = False

	def handle_mouse_click(
		self,
		event: int,
		x: int,
		y: int,
		flags: int = 0,
	) -> bool:
		"""Handle measurement, calibration, setup-origin, then recovery clicks."""
		del flags
		if event != cv2.EVENT_LBUTTONDOWN:
			return False
		click = (int(x), int(y))
		if self.measure_mode:
			self.measure_points = (
				[*self.measure_points, click]
				if len(self.measure_points) < 2 else [click]
			)
			return True
		if self.calibration_mode:
			index = find_blob_index_for_click(self.current_detections, click, 0.0)
			if index is None:
				self.selected_calibration_blob = None
				print(f"Calibration ignored: no detected red blob at ({x}, {y}).")
				return False
			blob = self.current_detections[index]
			sample = sample_blob_hsv_near_point(self.current_raw_frame, blob, click)
			if sample is None:
				self.selected_calibration_blob = None
				print(f"Calibration ignored: no red pixel could be sampled near ({x}, {y}).")
				return False
			selected = dict(blob)
			selected["click_point"] = click
			selected["sample_hsv"] = sample
			self.selected_calibration_blob = selected
			calibration = calibrate_from_blob(selected)
			sync_area_tuning_window_from_config()
			print(
				f"Calibrated HSV={calibration['sample_hsv']}, "
				f"area={calibration['area_min']}..{calibration['area_max']}"
			)
			return True
		if self.reference_origin_selection_mode:
			maximum = float(
				self.config.reference.get("origin_selection_max_distance_px", 25.0)
			)
			index = find_blob_index_for_click(self.current_detections, click, maximum)
			if index is None:
				print(
					f"Origin selection rejected: no blob within {maximum:.1f}px "
					f"of ({x}, {y})."
				)
				return False
			self.selected_origin_detection_index = index
			self.selected_origin_click = click
			self.selected_origin_candidate_center = tuple(
				float(v) for v in self.current_detections[index]["center"]
			)
			self.reference_origin_selection_mode = False
			self.reference_setup_mode = ReferenceSetupMode.READY_TO_PREVIEW
			self.proposed_ransac_result = None
			self.reference_proposal_detections = []
			print(f"Selected ORIGIN candidate at {self.selected_origin_candidate_center}.")
			return True
		if self.origin_reacquisition_selection_mode:
			maximum = float(
				self.config.reference.get("origin_selection_max_distance_px", 25.0)
			)
			index = find_blob_index_for_click(
				self.current_detections, click, maximum
			)
			if index is None:
				print(
					f"Origin reacquisition rejected: no blob within {maximum:.1f}px "
					f"of ({x}, {y})."
				)
				return False
			detection = self.current_detections[index]
			if (
				self.config.circles.require_validation_for_tracking
				and not bool(detection.get("circle_validation", False))
			):
				print(
					"Origin reacquisition rejected: the selected blob failed "
					"required Hough-circle validation."
				)
				return False
			if self.tracker is None:
				raise RuntimeError("Tracker stopped during origin reacquisition")
			origin = self.tracker.confirm_origin_reacquisition(detection)
			self.current_tracked_points = self.tracker.points
			self.origin_reacquisition_selection_mode = False
			print(
				f"Confirmed {origin.id} reacquisition at "
				f"{origin.current_position}."
			)
			return True
		return False

	def _nearest_candidate_index(
		self,
		center: tuple[float, float],
		maximum_distance: float,
		excluded_indices: set[int] | None = None,
		ambiguity_margin: float = 0.0,
	) -> int | None:
		excluded = excluded_indices or set()
		distances = sorted(
			(
				float(np.linalg.norm(np.subtract(detection["center"], center))),
				index,
			)
			for index, detection in enumerate(self.current_detections)
			if index not in excluded
		)
		if not distances or distances[0][0] > maximum_distance:
			return None
		if (
			len(distances) > 1
			and distances[1][0] <= maximum_distance
			and distances[1][0] - distances[0][0] <= ambiguity_margin
		):
			raise RuntimeError(
				"The selected origin candidate is ambiguous; select it again"
			)
		return distances[0][1]

	def _current_origin_detection_index(self) -> int:
		if self.selected_origin_candidate_center is None:
			raise RuntimeError("Select the reference origin first")
		maximum = float(self.config.reference.get(
			"origin_setup_identity_lock_distance_px",
			self.config.reference.get("origin_identity_lock_distance_px", 2.0),
		))
		ambiguity_margin = float(
			self.config.reference.get("origin_setup_ambiguity_margin_px", 1.0)
		)
		index = self._nearest_candidate_index(
			self.selected_origin_candidate_center,
			maximum,
			ambiguity_margin=ambiguity_margin,
		)
		if index is None:
			raise RuntimeError(
				"The selected origin candidate is no longer visible inside the "
				f"{maximum:.1f}px setup identity lock; select it again"
			)
		return index

	def reference_preview(self) -> RansacResult:
		if self.mode != ApplicationMode.REFERENCE_SETUP:
			raise RuntimeError("Run reference_start first")
		origin_index = self._current_origin_detection_index()
		maximum = float(self.config.reference.get("origin_selection_max_distance_px", 25.0))
		resolved_rigid_indices: list[int] = []
		for candidate in self.rigid_reference_candidate_centers:
			distances = [
				float(np.linalg.norm(np.subtract(detection["center"], candidate)))
				for detection in self.current_detections
			]
			nearest_index = int(np.argmin(distances))
			if nearest_index == origin_index and distances[nearest_index] <= maximum:
				raise ValueError(
					"The reference origin cannot also be a rigid-reference marker"
				)
			available = [
				index for index in np.argsort(distances)
				if index != origin_index and index not in resolved_rigid_indices
			]
			if not available or distances[int(available[0])] > maximum:
				raise RuntimeError("A selected rigid-reference candidate is no longer visible")
			resolved_rigid_indices.append(int(available[0]))
		if origin_index in resolved_rigid_indices:
			raise ValueError("The reference origin cannot also be a rigid-reference marker")

		proposal_detections = self.current_detections
		proposal_origin_index = origin_index
		proposal_rigid_indices = resolved_rigid_indices
		if self.config.circles.require_validation_for_reference:
			kept_indices = [
				index for index, detection in enumerate(self.current_detections)
				if bool(detection.get("circle_validation", False))
			]
			if origin_index not in kept_indices:
				raise ValueError(
					"The selected reference origin failed required Hough-circle validation"
				)
			if any(index not in kept_indices for index in resolved_rigid_indices):
				raise ValueError(
					"One or more selected rigid-reference markers failed required "
					"Hough-circle validation"
				)
			index_map = {
				source_index: filtered_index
				for filtered_index, source_index in enumerate(kept_indices)
			}
			proposal_detections = [
				self.current_detections[index] for index in kept_indices
			]
			proposal_origin_index = index_map[origin_index]
			proposal_rigid_indices = [
				index_map[index] for index in resolved_rigid_indices
			]
		resolution = (
			(self.current_raw_frame.shape[1], self.current_raw_frame.shape[0])
			if self.current_raw_frame is not None
			else (
				int(self.config.camera.get("width", 640)),
				int(self.config.camera.get("height", 480)),
			)
		)
		result = initialise_reference(
			proposal_detections,
			self._reference_config(),
			resolution,
			proposal_rigid_indices,
			reference_origin_detection_index=proposal_origin_index,
		)
		self.selected_origin_detection_index = origin_index
		self.rigid_reference_detection_indices = tuple(resolved_rigid_indices)
		self.rigid_reference_candidate_centers = tuple(
			tuple(
				float(value)
				for value in self.current_detections[index]["center"]
			)
			for index in resolved_rigid_indices
		)
		self.selected_origin_candidate_center = tuple(
			float(value) for value in self.current_detections[origin_index]["center"]
		)
		self.proposed_ransac_result = result
		self.reference_proposal_detections = [
			dict(detection) for detection in proposal_detections
		]
		self.reference_setup_mode = ReferenceSetupMode.PROPOSAL_READY
		return result

	def reference_accept(self) -> ReferenceProfile:
		result = self.proposed_ransac_result
		if result is None:
			raise RuntimeError("Run reference_preview first")
		if not result.valid:
			detail = "; ".join(getattr(result, "fatal_errors", ()) or result.warnings)
			raise ValueError(f"Reference proposal is invalid: {detail}")
		resolution = (
			(self.current_raw_frame.shape[1], self.current_raw_frame.shape[0])
			if self.current_raw_frame is not None
			else (
				int(self.config.camera.get("width", 640)),
				int(self.config.camera.get("height", 480)),
			)
		)
		profile = make_reference_profile(
			result, resolution, self._reference_config(), self.config.geometry
		)
		self.reference_profile = profile
		self.tracker = PointTracker(profile, self._tracking_config())
		self.tracking_enabled = True
		self.current_tracked_points = self.tracker.points
		self.reference_setup_mode = ReferenceSetupMode.IDLE
		self.reference_origin_selection_mode = False
		self.origin_reacquisition_selection_mode = False
		self.selected_origin_detection_index = None
		self.selected_origin_click = None
		self.selected_origin_candidate_center = None
		self.rigid_reference_detection_indices = ()
		self.rigid_reference_candidate_centers = ()
		self.reference_proposal_detections = []
		self.mode = ApplicationMode.TRACKING
		return profile

	def reject_reference(self) -> None:
		self.proposed_ransac_result = None
		self.reference_proposal_detections = []
		self.reference_setup_mode = (
			ReferenceSetupMode.READY_TO_PREVIEW
			if self.selected_origin_detection_index is not None
			else ReferenceSetupMode.ORIGIN_REQUIRED
		)

	def clear_reference(self) -> None:
		self.stop_dataset()
		self.stop_force_inference()
		self.stop_tracking()
		self.reference_profile = None
		self.proposed_ransac_result = None
		self.reference_proposal_detections = []
		self.rigid_reference_detection_indices = ()
		self.rigid_reference_candidate_centers = ()
		self.clear_reference_origin()
		self.reference_setup_mode = ReferenceSetupMode.IDLE
		self.mode = ApplicationMode.DIAGNOSTIC

	def reference_save(self, path: str | Path | None = None) -> Path:
		if self.reference_profile is None:
			raise RuntimeError("There is no accepted reference profile")
		target = self._profile_path(path)
		save_reference_profile(self.reference_profile, target)
		return target

	def reference_load(self, path: str | Path | None = None) -> ReferenceProfile:
		profile = load_reference_profile(self._profile_path(path))
		origin_points = [
			point for point in profile.points
			if point.point_group == PointGroup.REFERENCE_ORIGIN
		]
		if (
			not profile.reference_origin_point_id
			or len(origin_points) != 1
			or origin_points[0].id != profile.reference_origin_point_id
		):
			raise ValueError("Reference profile has no real reference-origin identity")
		expected = {
			"LINE_A": int(self.config.reference_ransac["expected_points_line_a"]),
			"LINE_B": int(self.config.reference_ransac["expected_points_line_b"]),
		}
		if profile.expected_points != expected:
			raise ValueError(
				f"Reference profile point counts {profile.expected_points} "
				f"do not match configured counts {expected}"
			)
		expected_reference_version = str(
			self.config.reference_ransac.get("configuration_version", "1")
		)
		if profile.configuration_version != expected_reference_version:
			raise ValueError(
				"Reference profile configuration version does not match "
				"camera-config.yaml"
			)
		expected_connections = str(
			self.config.geometry.get("structural_connection_version", "1")
		)
		if profile.structural_connection_version != expected_connections:
			raise ValueError(
				"Reference profile structural-connection version does not match "
				"camera-config.yaml"
			)
		self.stop_dataset()
		self.stop_force_inference()
		self.stop_tracking()
		self.reference_profile = profile
		self.proposed_ransac_result = None
		self.reference_proposal_detections = []
		self.reference_setup_mode = ReferenceSetupMode.IDLE
		self.reference_origin_selection_mode = False
		self.origin_reacquisition_selection_mode = False
		self.selected_origin_detection_index = None
		self.selected_origin_click = None
		self.selected_origin_candidate_center = None
		self.rigid_reference_detection_indices = ()
		self.rigid_reference_candidate_centers = ()
		self.mode = ApplicationMode.DIAGNOSTIC
		return profile

	def start_tracking(self) -> None:
		if self.reference_profile is None:
			raise RuntimeError("Load or accept a reference profile first")
		self.tracker = PointTracker(self.reference_profile, self._tracking_config())
		self.current_tracked_points = self.tracker.points
		self.tracking_enabled = True
		self.origin_reacquisition_selection_mode = False
		self.mode = ApplicationMode.TRACKING

	def stop_tracking(self) -> None:
		self.tracker = None
		self.tracking_enabled = False
		self.origin_reacquisition_selection_mode = False
		self.current_tracked_points = []
		self.current_geometry = None
		self.current_feature_names = []
		self.current_force_prediction = None

	def start_dataset(self, experiment_id: str, known_force_N: float) -> DatasetSession:
		if self.reference_profile is None or self.tracker is None:
			raise RuntimeError("An accepted reference and active tracking are required")
		force = float(known_force_N)
		if not np.isfinite(force):
			raise ValueError("Known force must be finite")
		self.known_reference_force = force
		self.dataset_session = DatasetSession.start(
			str(self._dataset_directory()),
			experiment_id,
			self.reference_profile,
			str(self.config.geometry.get("geometry_configuration_version", "2")),
			bool(self.config.dataset.get("save_images", True)),
			feature_schema_version=str(
				self.config.geometry.get("feature_schema_version", "2")
			),
		)
		self.pending_dataset_frames = 0
		self.pending_dataset_invalid_retries = 0
		self.mode = (
			ApplicationMode.FORCE_MEASUREMENT
			if self.force_measurement_enabled else ApplicationMode.DATA_COLLECTION
		)
		return self.dataset_session

	def set_dataset_force(self, known_force_N: float) -> None:
		if self.dataset_session is None:
			raise RuntimeError("No dataset session is active")
		force = float(known_force_N)
		if not np.isfinite(force):
			raise ValueError("Known force must be finite")
		self.known_reference_force = force

	def queue_dataset_samples(self, count: int) -> None:
		if self.dataset_session is None:
			raise RuntimeError("No dataset session is active")
		if int(count) < 1:
			raise ValueError("Sample frame count must be positive")
		self.pending_dataset_frames += int(count)
		self.pending_dataset_invalid_retries = 0

	def stop_dataset(self) -> DatasetSession | None:
		session = self.dataset_session
		self.dataset_session = None
		self.pending_dataset_frames = 0
		self.pending_dataset_invalid_retries = 0
		if self.mode == ApplicationMode.DATA_COLLECTION:
			self.mode = ApplicationMode.TRACKING if self.tracker else ApplicationMode.DIAGNOSTIC
		return session

	def load_force_model(
		self,
		model_path: str | Path,
		metadata_path: str | Path | None = None,
	) -> ForceModel:
		self.loaded_force_model = ForceModel.load(model_path, metadata_path)
		return self.loaded_force_model

	def start_force_inference(self) -> None:
		if self.loaded_force_model is None:
			raise RuntimeError("Load a force model first")
		if self.reference_profile is None or self.tracker is None:
			raise RuntimeError("An accepted reference and active tracking are required")
		self.force_measurement_enabled = True
		self.current_force_prediction = None
		self.mode = ApplicationMode.FORCE_MEASUREMENT

	def stop_force_inference(self) -> None:
		self.force_measurement_enabled = False
		self.current_force_prediction = None
		if self.mode == ApplicationMode.FORCE_MEASUREMENT:
			if self.dataset_session is not None:
				self.mode = ApplicationMode.DATA_COLLECTION
			else:
				self.mode = (
					ApplicationMode.TRACKING if self.tracker
					else ApplicationMode.DIAGNOSTIC
				)

	def _annotate_circle_validation(self) -> None:
		for detection in self.current_detections:
			x, y = detection["center"]
			detection["circle_validation"] = any(
				(float(x) - float(cx)) ** 2 + (float(y) - float(cy)) ** 2
				<= float(radius) ** 2
				for cx, cy, radius in self.current_circles
			)

	def _refresh_reference_origin_candidate(self) -> None:
		if (
			self.mode != ApplicationMode.REFERENCE_SETUP
			or self.reference_setup_mode == ReferenceSetupMode.PROPOSAL_READY
			or self.selected_origin_candidate_center is None
		):
			return
		try:
			index = self._current_origin_detection_index()
		except RuntimeError as exc:
			self.selected_origin_detection_index = None
			self.selected_origin_candidate_center = None
			self.reference_setup_mode = ReferenceSetupMode.ORIGIN_REQUIRED
			self.runtime_warnings.append(str(exc))
			return
		self.selected_origin_detection_index = index
		self.selected_origin_candidate_center = tuple(
			float(value) for value in self.current_detections[index]["center"]
		)

	def update_frame(
		self,
		raw_frame: np.ndarray,
		detections: list[dict[str, Any]] | None = None,
		timestamp: float | None = None,
	) -> np.ndarray:
		"""Run the single detector→tracker→geometry→consumer frame pipeline."""
		if raw_frame is None:
			raise ValueError("raw_frame cannot be None")
		self.current_frame_number += 1
		self.current_timestamp = float(time.time() if timestamp is None else timestamp)
		self.current_raw_frame = raw_frame
		self.current_gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		self.current_detections = (
			find_red_blob_data(raw_frame) if detections is None else detections
		)
		red_mask = create_red_mask(raw_frame)
		self.current_circles = detect_circles(red_mask)
		self._annotate_circle_validation()
		self.runtime_warnings = []
		self._refresh_reference_origin_candidate()
		if self.proposed_ransac_result is not None:
			self.runtime_warnings.extend(
				getattr(self.proposed_ransac_result, "fatal_errors", ())
			)
			self.runtime_warnings.extend(self.proposed_ransac_result.warnings)

		if self.tracking_enabled and self.tracker is not None:
			tracking_detections = self.current_detections
			if self.config.circles.require_validation_for_tracking:
				tracking_detections = [
					detection for detection in self.current_detections
					if bool(detection.get("circle_validation", False))
				]
				rejected_count = len(self.current_detections) - len(tracking_detections)
				if rejected_count:
					self.runtime_warnings.append(
						f"Hough validation rejected {rejected_count} tracking detection(s)"
					)
			self.current_tracked_points = self.tracker.update(
				self.current_gray_frame, tracking_detections
			)
			if (
				self.origin_reacquisition_selection_mode
				and not self.tracker.origin_reacquisition_required
			):
				self.origin_reacquisition_selection_mode = False
				self.runtime_warnings.append(
					"Origin recovery click cancelled because LK continuity resumed"
				)
			if self.reference_profile is None:
				raise RuntimeError("Tracker is active without a reference profile")
			self.current_geometry = extract_geometry(
				self.reference_profile,
				self.current_tracked_points,
				{**self.config.geometry, **self.config.reference},
			)
			self.current_feature_names = list(self.current_geometry.feature_names)
			self.runtime_warnings.extend(self.current_geometry.warnings)
			self.runtime_warnings.extend(
				getattr(self.current_geometry, "fatal_errors", ())
			)
		else:
			self.current_tracked_points = []
			self.current_geometry = None
			self.current_feature_names = []

		self.process_pending_dataset_sample()
		self.process_force_inference()
		return self.draw_overlays(raw_frame)

	def _invalid_frame_reason(self) -> str | None:
		if self.tracker is None:
			return "Tracker state is invalid"
		if self.current_geometry is None:
			return "Geometry is unavailable"
		if not self.current_geometry.origin_valid:
			return "Reference origin is invalid"
		if not self.tracker.valid:
			return "Tracker state is invalid"
		if not self.current_geometry.valid:
			errors = getattr(self.current_geometry, "fatal_errors", ())
			return "; ".join(errors) if errors else "Geometry is invalid"
		return None

	def status_snapshot(self) -> dict[str, Any]:
		"""Return detached scalar state for GUI/status consumers."""
		tracker = self.tracker
		geometry = self.current_geometry
		proposal = self.proposed_ransac_result
		session = self.dataset_session
		profile = self.reference_profile
		reacquisition_required = bool(
			tracker is not None and tracker.origin_reacquisition_required
		)
		recording_reason = self._invalid_frame_reason()
		return {
			"frame_number": self.current_frame_number,
			"frame_size": (
				None if self.current_raw_frame is None
				else (self.current_raw_frame.shape[1], self.current_raw_frame.shape[0])
			),
			"mode": self.mode.value,
			"reference_setup_mode": self.reference_setup_mode.value,
			"reference_configured": profile is not None,
			"reference_profile_id": "" if profile is None else profile.profile_id,
			"reference_origin_selection": self.reference_origin_selection_mode,
			"origin_reacquisition_selection": self.origin_reacquisition_selection_mode,
			"origin_reacquisition_required": reacquisition_required,
			"origin_candidate_index": self.selected_origin_detection_index,
			"proposal_available": proposal is not None,
			"proposal_valid": None if proposal is None else proposal.valid,
			"proposal_quality": None if proposal is None else proposal.quality,
			"proposal_errors": [] if proposal is None else list(proposal.fatal_errors),
			"proposal_warnings": [] if proposal is None else list(proposal.warnings),
			"tracking_active": tracker is not None and self.tracking_enabled,
			"tracking_valid": False if tracker is None else tracker.valid,
			"tracking_quality": 0.0 if tracker is None else tracker.quality,
			"detected_marker_count": len(self.current_detections),
			"geometry_valid": False if geometry is None else geometry.valid,
			"geometry_quality": 0.0 if geometry is None else geometry.quality,
			"origin_valid": False if geometry is None else geometry.origin_valid,
			"runtime_warnings": list(self.runtime_warnings),
			"display_options": {
				name: bool(getattr(self.display_options, name))
				for name in self.display_options.__dataclass_fields__
			},
			"dataset_active": session is not None,
			"dataset_experiment_id": "" if session is None else session.experiment_id,
			"dataset_force_N": self.known_reference_force,
			"dataset_accepted": 0 if session is None else session.accepted,
			"dataset_rejected": 0 if session is None else session.rejected,
			"dataset_pending": self.pending_dataset_frames,
			"dataset_last_result": self.last_dataset_result,
			"recording_valid": recording_reason is None,
			"recording_invalid_reason": recording_reason or "",
		}

	def process_pending_dataset_sample(self) -> bool | None:
		if self.dataset_session is None or self.pending_dataset_frames <= 0:
			return None
		reason = self._invalid_frame_reason()
		if reason is not None:
			self.dataset_session.record_rejection(
				reason,
				frame_number=self.current_frame_number,
				timestamp=self.current_timestamp,
			)
			self.last_dataset_result = f"REJECTED: {reason}"
			self.pending_dataset_invalid_retries += 1
			retry = bool(self.config.dataset.get("retry_invalid_frames", True))
			limit = max(
				1, int(self.config.dataset.get("maximum_invalid_frame_retries", 120))
			)
			if not retry or self.pending_dataset_invalid_retries >= limit:
				self.pending_dataset_frames -= 1
				self.pending_dataset_invalid_retries = 0
			print(
				f"Dataset frame {self.current_frame_number} rejected: {reason}; "
				f"pending={self.pending_dataset_frames}"
			)
			return False
		assert self.dataset_session is not None
		assert self.current_geometry is not None
		assert self.tracker is not None
		assert self.known_reference_force is not None
		assert self.reference_profile is not None
		origin = self.tracker.get_point(
			self.reference_profile.reference_origin_point_id
		)
		accepted = self.dataset_session.add_sample(
			timestamp=self.current_timestamp,
			frame_number=self.current_frame_number,
			known_force_N=self.known_reference_force,
			points=self.current_tracked_points,
			geometry=self.current_geometry,
			raw_frame=self.current_raw_frame,
			tracker_valid=self.tracker.valid,
			tracker_quality=self.tracker.quality,
			feature_schema_version=str(
				self.config.geometry.get("feature_schema_version", "2")
			),
			origin_point=origin,
		)
		if accepted:
			self.pending_dataset_frames -= 1
			self.pending_dataset_invalid_retries = 0
			self.last_dataset_result = "ACCEPTED"
			print(
				f"Dataset frame {self.current_frame_number} accepted; "
				f"pending={self.pending_dataset_frames}"
			)
		else:
			reason = (
				self.dataset_session.last_rejection_reason
				or "dataset validation failed"
			)
			self.last_dataset_result = f"REJECTED: {reason}"
			self.pending_dataset_invalid_retries += 1
			retry = bool(self.config.dataset.get("retry_invalid_frames", True))
			limit = max(
				1, int(self.config.dataset.get("maximum_invalid_frame_retries", 120))
			)
			if not retry or self.pending_dataset_invalid_retries >= limit:
				self.pending_dataset_frames -= 1
				self.pending_dataset_invalid_retries = 0
			print(
				f"Dataset frame {self.current_frame_number} rejected: {reason}; "
				f"pending={self.pending_dataset_frames}"
			)
		return accepted

	def _invalid_prediction(self, warning: str) -> ForcePrediction:
		model_name = (
			self.loaded_force_model.metadata.model_name
			if self.loaded_force_model is not None else "no-model"
		)
		tracking_quality = self.tracker.quality if self.tracker is not None else 0.0
		geometry_quality = (
			self.current_geometry.quality if self.current_geometry is not None else 0.0
		)
		reference_quality = (
			self.current_geometry.transform.quality
			if self.current_geometry is not None else 0.0
		)
		return ForcePrediction(
			raw_force_N=None,
			displayed_force_N=None,
			valid=False,
			model_name=model_name,
			tracking_quality=tracking_quality,
			reference_quality=reference_quality,
			geometry_quality=geometry_quality,
			warning=warning,
		)

	def process_force_inference(self) -> ForcePrediction | None:
		if not self.force_measurement_enabled:
			return None
		if self.loaded_force_model is None:
			self.current_force_prediction = self._invalid_prediction("No force model loaded")
			return self.current_force_prediction
		reason = self._invalid_frame_reason()
		if reason is not None:
			self.current_force_prediction = self._invalid_prediction(reason)
			return self.current_force_prediction
		assert self.current_geometry is not None
		assert self.reference_profile is not None
		self.current_force_prediction = self.loaded_force_model.predict(
			self.current_geometry.feature_vector,
			self.current_geometry.feature_names,
			tracking_quality=self.tracker.quality if self.tracker else 0.0,
			reference_quality=self.current_geometry.transform.quality,
			geometry_quality=self.current_geometry.quality,
			reference_profile_id=self.reference_profile.profile_id,
			geometry_version=str(
				self.config.geometry.get("geometry_configuration_version", "2")
			),
			feature_schema_version=str(
				self.config.geometry.get("feature_schema_version", "2")
			),
		)
		return self.current_force_prediction

	def _draw_reference_proposal(self, frame: np.ndarray) -> None:
		result = self.proposed_ransac_result
		if result is None:
			return
		by_id = {point.id: point for point in result.points}
		if self.display_options.show_lines:
			for index, line in enumerate(result.lines):
				line_points = [
					by_id[point_id].reference_position
					for point_id in line.point_ids if point_id in by_id
				]
				if len(line_points) >= 2:
					color = (255, 190, 0) if index == 0 else (0, 190, 255)
					first = tuple(int(round(v)) for v in line_points[0])
					last = tuple(int(round(v)) for v in line_points[-1])
					cv2.line(frame, first, last, color, 2, cv2.LINE_AA)
		if self.display_options.show_points:
			draw_tracking(frame, result.points)
		if self.display_options.show_outliers:
			proposal_detections = (
				self.reference_proposal_detections or self.current_detections
			)
			for index in result.rejected_detection_indices:
				if 0 <= index < len(proposal_detections):
					position = tuple(
						int(round(v)) for v in proposal_detections[index]["center"]
					)
					cv2.circle(frame, position, 9, (255, 0, 255), 2)
					cv2.putText(
						frame, "OUTLIER", (position[0] + 7, position[1] + 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1,
						cv2.LINE_AA,
						)

	def _draw_detection_indices(self, frame: np.ndarray) -> None:
		if not self.display_options.show_points:
			return
		if self.tracker is not None and self.mode != ApplicationMode.REFERENCE_SETUP:
			return
		for index, detection in enumerate(self.current_detections):
			position = tuple(int(round(value)) for value in detection["center"])
			cv2.circle(frame, position, 3, (0, 0, 255), -1)
			cv2.putText(
				frame,
				f"D{index}",
				(position[0] + 5, position[1] + 12),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.35,
				(0, 0, 255),
				1,
				cv2.LINE_AA,
			)

	def _draw_origin_candidate(self, frame: np.ndarray) -> None:
		for candidate in self.rigid_reference_candidate_centers:
			rigid_position = tuple(int(round(value)) for value in candidate)
			cv2.circle(frame, rigid_position, 10, (255, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(
				frame, "RIGID", (rigid_position[0] + 12, rigid_position[1] - 6),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
				cv2.LINE_AA,
			)
		if self.selected_origin_candidate_center is None:
			return
		position = tuple(
			int(round(v)) for v in self.selected_origin_candidate_center
		)
		cv2.circle(frame, position, 12, (255, 0, 255), 3, cv2.LINE_AA)
		cv2.putText(
			frame, "ORIGIN", (position[0] + 14, position[1] - 8),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA,
		)

	def _draw_geometry(self, frame: np.ndarray) -> None:
		if not self.display_options.show_geometry or self.current_geometry is None:
			return
		points = self.current_geometry.raw_tracked_coordinates
		for key in self.current_geometry.connections:
			a_id, b_id = key.split("--", 1)
			if a_id not in points or b_id not in points:
				continue
			first = tuple(int(round(v)) for v in points[a_id])
			second = tuple(int(round(v)) for v in points[b_id])
			cv2.line(frame, first, second, (190, 190, 50), 1, cv2.LINE_AA)

	def _status_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
		lines: list[tuple[str, tuple[int, int, int]]] = [
			(f"MODE: {self.mode.value}", (255, 255, 255)),
		]
		if self.mode == ApplicationMode.REFERENCE_SETUP:
			lines.append((
				f"REFERENCE: {self.reference_setup_mode.value}",
				(0, 255, 255),
			))
			if self.reference_origin_selection_mode:
				lines.append(("Click the real moving-finger red blob", (255, 0, 255)))
		if self.origin_reacquisition_selection_mode:
			lines.append((
				"Click the restored physical ORIGIN marker to confirm its ID",
				(255, 0, 255),
			))
		elif (
			self.tracker is not None
			and self.tracker.origin_reacquisition_required
		):
			lines.append((
				"ORIGIN ID LOCKED: run origin_reacquire, then click the marker",
				(0, 0, 255),
			))
		if self.tracker is not None:
			lines.append((
				f"TRACKING: {'VALID' if self.tracker.valid else 'INVALID'} "
				f"q={self.tracker.quality:.2f}",
				(0, 255, 0) if self.tracker.valid else (0, 0, 255),
			))
		if self.current_geometry is not None:
			lines.append((
				f"ORIGIN: {'VALID' if self.current_geometry.origin_valid else 'LOST'} "
				f"q={self.current_geometry.origin_tracking_quality:.2f}",
				(255, 0, 255) if self.current_geometry.origin_valid else (0, 0, 255),
			))
			lines.append((
				f"GEOMETRY: {'VALID' if self.current_geometry.valid else 'INVALID'} "
				f"q={self.current_geometry.quality:.2f}",
				(0, 255, 0) if self.current_geometry.valid else (0, 0, 255),
			))
			transform = self.current_geometry.transform
			frame_label = (
				"TRANSLATION ONLY (camera axes; no rotation)"
				if transform.mode == "translation_only"
				else "TRANSLATION + ROTATION (rigid references)"
			)
			lines.append((f"FRAME: {frame_label}", (200, 200, 200)))
		if self.dataset_session is not None:
			lines.append((
				f"DATA: accepted={self.dataset_session.accepted} "
				f"rejected={self.dataset_session.rejected} "
				f"pending={self.pending_dataset_frames}",
				(255, 255, 0),
			))
			if self.last_dataset_result:
				lines.append((
					f"DATA LAST: {self.last_dataset_result}",
					(
						(0, 255, 0)
						if self.last_dataset_result == "ACCEPTED"
						else (0, 165, 255)
					),
				))
		if self.force_measurement_enabled:
			prediction = self.current_force_prediction
			if prediction is not None and prediction.displayed_force_N is not None:
				status = "VALID" if prediction.valid else "INVALID"
				lines.append((
					f"FORCE: {prediction.displayed_force_N:.3f} N ({status})",
					(0, 255, 0) if prediction.valid else (0, 165, 255),
				))
				lines.append((
					f"FORCE RAW: {prediction.raw_force_N:.3f} N",
					(200, 200, 200),
				))
			else:
				warning = (
					prediction.warning
					if prediction is not None and prediction.warning
					else "waiting"
				)
				lines.append((f"FORCE: PAUSED ({warning})", (0, 0, 255)))
			if prediction is not None:
				lines.append((
					f"MODEL: {prediction.model_name}",
					(200, 200, 200),
				))
				lines.append((
					f"QUALITY: tracking={prediction.tracking_quality:.2f} "
					f"reference={prediction.reference_quality:.2f} "
					f"geometry={prediction.geometry_quality:.2f}",
					(200, 200, 200),
				))
				if prediction.warning:
					lines.append((
						f"FORCE WARNING: {prediction.warning}",
						(0, 165, 255),
					))
		return lines

	def draw_overlays(self, raw_frame: np.ndarray) -> np.ndarray:
		if self.display_options.show_mask_only:
			mask = create_red_mask(raw_frame)
			frame = np.zeros_like(raw_frame)
			frame[:, :, 2] = mask
		else:
			frame = raw_frame.copy()
		if self.display_options.show_circles:
			draw_circles(frame, self.current_circles)
		self._draw_detection_indices(frame)
		if self.tracker is not None and self.display_options.show_points:
			draw_tracking(frame, self.current_tracked_points)
		self._draw_reference_proposal(frame)
		self._draw_origin_candidate(frame)
		self._draw_geometry(frame)
		draw_blob_calibration_info(frame, self.selected_calibration_blob)
		if len(self.measure_points) == 2:
			draw_measurement(frame, self.measure_points[0], self.measure_points[1])

		if self.current_geometry is not None and not self.current_geometry.origin_valid:
			height, width = frame.shape[:2]
			text = "REFERENCE ORIGIN LOST"
			size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
			cv2.putText(
				frame, text, ((width - size[0]) // 2, max(35, height // 8)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
			)
		if self.display_options.show_status:
			y = 20
			for text, color in self._status_lines():
				cv2.putText(
					frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
					0.45, color, 1, cv2.LINE_AA,
				)
				y += 18
		if self.display_options.show_warnings and self.runtime_warnings:
			height = frame.shape[0]
			for offset, warning in enumerate(self.runtime_warnings[-4:]):
				cv2.putText(
					frame, warning[:100], (10, height - 10 - 17 * offset),
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1,
					cv2.LINE_AA,
				)
		if (
			self.display_options.show_features
			and self.current_geometry is not None
			and self.current_geometry.feature_names
		):
			x = max(10, frame.shape[1] - 275)
			cv2.putText(
				frame,
				f"FEATURES ({len(self.current_geometry.feature_names)})",
				(x, 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.4,
				(220, 220, 220),
				1,
				cv2.LINE_AA,
			)
			for index, (name, value) in enumerate(zip(
				self.current_geometry.feature_names[:6],
				self.current_geometry.feature_vector[:6],
			), start=1):
				cv2.putText(
					frame,
					f"{name}: {float(value):.3g}",
					(x, 20 + 16 * index),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.35,
					(220, 220, 220),
					1,
					cv2.LINE_AA,
				)
		return frame
