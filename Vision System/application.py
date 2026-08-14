"""Central runtime coordinator for reference setup, tracking, data, and inference."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

from blue_tracking import BlueBlobTracker, BlueTrackStatus
from dataset import DatasetSession
from force_model import ForceModel, ForcePrediction
from geometry import GeometryResult, extract_geometry
from processing import (
	AppConfig,
	apply_calibration_preset,
	calibrate_blue_from_blob,
	calibrate_from_blob,
	cfg,
	configuration_compatibility_notes,
	create_blue_mask,
	create_red_mask,
	detect_circles,
	draw_circles,
	draw_measurement,
	find_blob_index_for_click,
	find_blue_blob_data,
	find_red_blob_data,
	sample_blue_blob_hsv_near_point,
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


@dataclass(frozen=True)
class DatasetBatch:
	"""A label and acquisition request frozen at the instant it is queued."""

	known_force_N: float
	force_step_id: str
	loading_direction: str
	repetition: int
	requested_samples: int
	remaining_samples: int
	selected_point_id: str = ""
	require_blue_target: bool = False
	require_motor_telemetry: bool = False
	maximum_motor_telemetry_age_s: float | None = None
	sample_interval_ms: int = 0
	next_sample_monotonic_s: float = 0.0


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
	show_blue_mask_only: bool = False

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
				("show_blue_mask_only", False),
			)
		})


@dataclass
class ApplicationState:
	"""Own all mutable state used by the camera loop and command surface."""

	config: AppConfig = field(default_factory=lambda: cfg)
	current_raw_frame: np.ndarray | None = None
	current_gray_frame: np.ndarray | None = None
	current_detections: list[dict[str, Any]] = field(default_factory=list)
	current_blue_detections: list[dict[str, Any]] = field(default_factory=list)
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
	pending_dataset_batches: list[DatasetBatch] = field(default_factory=list)
	dataset_batch_sequence: int = 0
	loaded_force_model: ForceModel | None = None
	current_force_prediction: ForcePrediction | None = None
	force_measurement_enabled: bool = False
	calibration_mode: bool = False
	blue_calibration_mode: bool = False
	measure_mode: bool = False
	measure_points: list[tuple[int, int]] = field(default_factory=list)
	selected_calibration_blob: dict[str, Any] | None = None
	selected_blue_calibration_blob: dict[str, Any] | None = None
	blue_tracker: BlueBlobTracker | None = None
	blue_origin_relative_position: tuple[float, float] | None = None
	blue_relative_unavailable_reason: str = "Blue target is not selected"
	last_click_rejection_reason: str = ""
	tracking_enabled: bool = False
	last_dataset_result: str = ""
	runtime_warnings: list[str] = field(default_factory=list)

	def __post_init__(self) -> None:
		if self.display_options is None:
			self.display_options = DisplayOptions.from_mapping(self.config.display)
		if self.blue_tracker is None:
			self.blue_tracker = BlueBlobTracker(self.config.blue_tracking)

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

	def start_calibration(self) -> None:
		"""Arm blob-click calibration using the configured HSV preset."""
		if (
			self.measure_mode
			or self.reference_origin_selection_mode
			or self.origin_reacquisition_selection_mode
			or self.blue_calibration_mode
		):
			raise RuntimeError("Turn off the other mouse modes first")
		apply_calibration_preset(self.config)
		self.calibration_mode = True
		self.selected_calibration_blob = None

	def stop_calibration(self) -> None:
		self.calibration_mode = False

	def clear_calibration_selection(self) -> None:
		self.selected_calibration_blob = None

	def start_blue_calibration(self) -> None:
		"""Arm blue click calibration; a successful click also binds tracking."""
		if (
			self.measure_mode
			or self.calibration_mode
			or self.reference_origin_selection_mode
			or self.origin_reacquisition_selection_mode
		):
			raise RuntimeError("Turn off the other mouse modes first")
		self.blue_calibration_mode = True
		self.selected_blue_calibration_blob = None

	def stop_blue_calibration(self) -> None:
		"""Leave click mode without stopping an already selected target track."""
		self.blue_calibration_mode = False

	def clear_blue_target(self) -> None:
		self.selected_blue_calibration_blob = None
		self.blue_origin_relative_position = None
		self.blue_relative_unavailable_reason = "Blue target is not selected"
		if self.blue_tracker is not None:
			self.blue_tracker.clear()

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
			or self.blue_calibration_mode
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
			or self.blue_calibration_mode
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
		"""Handle measurement, color calibration, setup-origin, then recovery clicks."""
		del flags
		if event != cv2.EVENT_LBUTTONDOWN:
			return False
		self.last_click_rejection_reason = ""
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
				self.last_click_rejection_reason = self._explain_calibration_rejection(click)
				print(self.last_click_rejection_reason, file=sys.stderr)
				return False
			blob = self.current_detections[index]
			sample = sample_blob_hsv_near_point(self.current_raw_frame, blob, click)
			if sample is None:
				self.selected_calibration_blob = None
				self.last_click_rejection_reason = (
					f"Calibration rejected at ({x}, {y}). Red-mask sampling: FAIL — "
					"the selected contour has no red-mask pixel in the latest frame. "
					"The marker may have moved; try clicking it again."
				)
				print(self.last_click_rejection_reason, file=sys.stderr)
				return False
			selected = dict(blob)
			selected["click_point"] = click
			selected["sample_hsv"] = sample
			self.selected_calibration_blob = selected
			calibration = calibrate_from_blob(selected, config=self.config)
			sync_area_tuning_window_from_config()
			print(
				f"Calibrated HSV={calibration['sample_hsv']}, "
				f"area={calibration['area_min']}..{calibration['area_max']}"
			)
			return True
		if self.blue_calibration_mode:
			index = find_blob_index_for_click(
				self.current_blue_detections, click, 0.0
			)
			if index is None:
				self.selected_blue_calibration_blob = None
				self.last_click_rejection_reason = (
					self._explain_blue_calibration_rejection(click)
				)
				print(self.last_click_rejection_reason, file=sys.stderr)
				return False
			blob = self.current_blue_detections[index]
			sample = sample_blue_blob_hsv_near_point(
				self.current_raw_frame, blob, click, config=self.config
			)
			if sample is None:
				self.selected_blue_calibration_blob = None
				self.last_click_rejection_reason = (
					f"Blue calibration rejected at ({x}, {y}). Blue-mask "
					"sampling failed; the marker may have moved. Try clicking it again."
				)
				print(self.last_click_rejection_reason, file=sys.stderr)
				return False
			selected = dict(blob)
			selected["click_point"] = click
			selected["sample_hsv"] = sample
			calibration = calibrate_blue_from_blob(selected, config=self.config)
			if self.current_gray_frame is None or self.blue_tracker is None:
				raise RuntimeError("No processed camera frame is available for blue tracking")
			self.blue_tracker.bind(self.current_gray_frame, selected)
			self.selected_blue_calibration_blob = selected
			self._update_blue_origin_relative_position()
			print(
				f"Blue target selected at {self.blue_tracker.track.current_position}; "
				f"HSV={calibration['sample_hsv']}, "
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

	def _explain_calibration_rejection(self, click: tuple[int, int]) -> str:
		"""Explain red-mask and area-filter failures in evaluation order."""
		frame = self.current_raw_frame
		x, y = click
		if frame is None:
			return "Calibration rejected: no camera frame is available yet."
		height, width = frame.shape[:2]
		if x < 0 or y < 0 or x >= width or y >= height:
			return (
				f"Calibration rejected at ({x}, {y}): click is outside the "
				f"{width}x{height} camera frame."
			)

		calibration = self.config.calibration
		mask = create_red_mask(
			frame,
			calibration.red_lower_1,
			calibration.red_upper_1,
			calibration.red_lower_2,
			calibration.red_upper_2,
		)
		pixel_is_red = bool(mask[y, x])
		contours, _ = cv2.findContours(
			mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)
		contour = next((
			candidate for candidate in contours
			if cv2.pointPolygonTest(
				candidate, (float(x), float(y)), False
			) >= 0
		), None)
		if contour is None:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
			return (
				f"Calibration rejected at ({x}, {y}). Red-mask check: FAIL — "
				f"pixel HSV={tuple(int(value) for value in hsv)} is not inside a "
				"red-mask contour. Active HSV ranges: "
				f"{tuple(int(v) for v in calibration.red_lower_1)}.."
				f"{tuple(int(v) for v in calibration.red_upper_1)} and "
				f"{tuple(int(v) for v in calibration.red_lower_2)}.."
				f"{tuple(int(v) for v in calibration.red_upper_2)}."
			)

		area = float(cv2.contourArea(contour))
		mask_result = (
			"pixel is in the red mask"
			if pixel_is_red else
			"click is inside a red-mask contour (the clicked pixel itself is not masked)"
		)
		minimum = int(calibration.min_area)
		maximum = calibration.max_area
		if area < minimum:
			return (
				f"Calibration rejected at ({x}, {y}). Red-mask check: PASS — "
				f"{mask_result}. Area check: FAIL — contour area {area:.1f} px² "
				f"is below the minimum {minimum} px²."
			)
		if maximum is not None and area > int(maximum):
			return (
				f"Calibration rejected at ({x}, {y}). Red-mask check: PASS — "
				f"{mask_result}. Area check: FAIL — contour area {area:.1f} px² "
				f"is above the maximum {int(maximum)} px²."
			)
		if cv2.moments(contour)["m00"] == 0:
			return (
				f"Calibration rejected at ({x}, {y}). Red-mask check: PASS — "
				f"{mask_result}. Contour check: FAIL — the contour has zero moment "
				"and no usable centre."
			)
		return (
			f"Calibration rejected at ({x}, {y}). Red-mask check: PASS — "
			f"{mask_result}; contour area {area:.1f} px² is within the active "
			f"range {minimum}..{'unlimited' if maximum is None else int(maximum)} px². "
			"The displayed frame may be older than the latest camera frame; try clicking again."
		)

	def _explain_blue_calibration_rejection(
		self,
		click: tuple[int, int],
	) -> str:
		"""Explain why a click did not identify a valid blue contour."""
		frame = self.current_raw_frame
		x, y = click
		if frame is None:
			return "Blue calibration rejected: no camera frame is available yet."
		height, width = frame.shape[:2]
		if x < 0 or y < 0 or x >= width or y >= height:
			return (
				f"Blue calibration rejected at ({x}, {y}): click is outside "
				f"the {width}x{height} camera frame."
			)
		blue = self.config.blue_calibration
		mask = create_blue_mask(frame, config=self.config)
		pixel_is_blue = bool(mask[y, x])
		contours, _ = cv2.findContours(
			mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)
		contour = next((
			candidate for candidate in contours
			if cv2.pointPolygonTest(
				candidate, (float(x), float(y)), False
			) >= 0
		), None)
		if contour is None:
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
			return (
				f"Blue calibration rejected at ({x}, {y}). Blue-mask check: "
				f"FAIL — pixel HSV={tuple(int(value) for value in hsv)} is not "
				"inside a blue-mask contour. Active HSV ranges: "
				f"{tuple(int(v) for v in blue.blue_lower_1)}.."
				f"{tuple(int(v) for v in blue.blue_upper_1)} and "
				f"{tuple(int(v) for v in blue.blue_lower_2)}.."
				f"{tuple(int(v) for v in blue.blue_upper_2)}."
			)
		area = float(cv2.contourArea(contour))
		mask_result = (
			"pixel is in the blue mask"
			if pixel_is_blue else
			"click is inside a blue-mask contour"
		)
		if area < int(blue.min_area):
			return (
				f"Blue calibration rejected at ({x}, {y}). Blue-mask check: "
				f"PASS — {mask_result}. Area check: FAIL — contour area "
				f"{area:.1f} px² is below the minimum {int(blue.min_area)} px²."
			)
		if blue.max_area is not None and area > int(blue.max_area):
			return (
				f"Blue calibration rejected at ({x}, {y}). Blue-mask check: "
				f"PASS — {mask_result}. Area check: FAIL — contour area "
				f"{area:.1f} px² is above the maximum {int(blue.max_area)} px²."
			)
		return (
			f"Blue calibration rejected at ({x}, {y}). Blue-mask check: PASS — "
			f"{mask_result}; contour area {area:.1f} px² is within the active "
			"range. The displayed frame may be stale; try clicking again."
		)

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
			save_raw_coordinates=bool(
				self.config.dataset.get("save_raw_coordinates", True)
			),
			save_compensated_coordinates=bool(
				self.config.dataset.get("save_compensated_coordinates", True)
			),
			save_origin_relative_coordinates=bool(
				self.config.dataset.get("save_origin_relative_coordinates", True)
			),
		)
		self.pending_dataset_frames = 0
		self.pending_dataset_invalid_retries = 0
		self.pending_dataset_batches = []
		self.dataset_batch_sequence = 0
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
		if self.pending_dataset_frames:
			raise RuntimeError(
				"Cannot change the force label while a sample batch is pending"
			)
		self.known_reference_force = force

	def queue_dataset_samples(self, count: int) -> None:
		"""Queue a compatibility/manual batch without requiring blue or motor data."""
		if self.dataset_session is None:
			raise RuntimeError("No dataset session is active")
		if self.known_reference_force is None:
			raise RuntimeError("Set a known force before queuing samples")
		self.dataset_batch_sequence += 1
		self.queue_dataset_batch(
			count=count,
			known_force_N=self.known_reference_force,
			force_step_id=f"manual-{self.dataset_batch_sequence:04d}",
			loading_direction="unspecified",
			repetition=1,
			require_blue_target=False,
			require_motor_telemetry=False,
			sample_interval_ms=int(
				self.config.dataset.get("sample_interval_ms", 0)
			),
		)

	def queue_dataset_batch(
		self,
		*,
		count: int,
		known_force_N: float,
		force_step_id: str,
		loading_direction: str,
		repetition: int,
		require_blue_target: bool = True,
		require_motor_telemetry: bool = False,
		maximum_motor_telemetry_age_s: float | None = None,
		sample_interval_ms: int = 0,
	) -> DatasetBatch:
		"""Freeze one experiment step so later UI edits cannot relabel its rows."""
		if self.dataset_session is None:
			raise RuntimeError("No dataset session is active")
		try:
			count_value = int(count)
			repetition_value = int(repetition)
			force = float(known_force_N)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError("Dataset batch values are invalid") from exc
		if count_value < 1:
			raise ValueError("Sample frame count must be positive")
		if repetition_value < 1:
			raise ValueError("Repetition must be positive")
		try:
			interval_ms = int(sample_interval_ms)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError("Sample interval must be an integer") from exc
		if interval_ms < 0:
			raise ValueError("Sample interval cannot be negative")
		if not np.isfinite(force):
			raise ValueError("Known force must be finite")
		step_id = str(force_step_id).strip()
		if not step_id:
			raise ValueError("Force step ID must not be empty")
		direction = str(loading_direction).strip().lower()
		if direction not in {"loading", "unloading", "baseline", "unspecified"}:
			raise ValueError("Loading direction is invalid")
		age_limit = maximum_motor_telemetry_age_s
		if require_motor_telemetry:
			if age_limit is None:
				raise ValueError(
					"A maximum motor telemetry age is required for automated samples"
				)
			age_limit = float(age_limit)
			if not np.isfinite(age_limit) or age_limit <= 0.0:
				raise ValueError("Maximum motor telemetry age must be positive")

		selected_point_id = ""
		if require_blue_target:
			selected_point_id = self._nearest_deforming_point_to_blue().id

		batch = DatasetBatch(
			known_force_N=force,
			force_step_id=step_id,
			loading_direction=direction,
			repetition=repetition_value,
			requested_samples=count_value,
			remaining_samples=count_value,
			selected_point_id=selected_point_id,
			require_blue_target=bool(require_blue_target),
			require_motor_telemetry=bool(require_motor_telemetry),
			maximum_motor_telemetry_age_s=age_limit,
			sample_interval_ms=interval_ms,
			next_sample_monotonic_s=time.monotonic(),
		)
		self.pending_dataset_batches.append(batch)
		self._sync_pending_dataset_frames()
		self.pending_dataset_invalid_retries = 0
		return batch

	def abort_pending_dataset_batch(self) -> DatasetBatch | None:
		"""Discard only the current queued request; accepted CSV rows stay intact."""
		if not self.pending_dataset_batches:
			return None
		batch = self.pending_dataset_batches.pop(0)
		self.pending_dataset_invalid_retries = 0
		self._sync_pending_dataset_frames()
		self.last_dataset_result = f"ABORTED: {batch.force_step_id}"
		return batch

	def _sync_pending_dataset_frames(self) -> None:
		self.pending_dataset_frames = sum(
			batch.remaining_samples for batch in self.pending_dataset_batches
		)

	def _consume_pending_dataset_frame(self) -> None:
		batch = self.pending_dataset_batches[0]
		remaining = batch.remaining_samples - 1
		if remaining <= 0:
			self.pending_dataset_batches.pop(0)
		else:
			self.pending_dataset_batches[0] = replace(
				batch,
				remaining_samples=remaining,
				next_sample_monotonic_s=(
					time.monotonic() + batch.sample_interval_ms / 1000.0
				),
			)
		self._sync_pending_dataset_frames()

	def _schedule_pending_dataset_retry(self, batch: DatasetBatch) -> None:
		if self.pending_dataset_batches and self.pending_dataset_batches[0] is batch:
			self.pending_dataset_batches[0] = replace(
				batch,
				next_sample_monotonic_s=(
					time.monotonic() + batch.sample_interval_ms / 1000.0
				),
			)

	def _nearest_deforming_point_to_blue(self) -> TrackedPoint:
		blue_tracker = self.blue_tracker
		if (
			blue_tracker is None
			or not blue_tracker.valid
			or blue_tracker.track.current_position is None
		):
			reason = (
				"Blue target is not valid"
				if blue_tracker is None else blue_tracker.invalid_reason
			)
			raise RuntimeError(reason or "Blue target is not valid")
		blue = np.asarray(blue_tracker.track.current_position, dtype=float)
		candidates: list[tuple[float, str, TrackedPoint]] = []
		for point in self.current_tracked_points:
			if (
				point.point_group != PointGroup.DEFORMING
				or point.status in {PointStatus.MISSING, PointStatus.INVALID}
			):
				continue
			position = np.asarray(point.current_position, dtype=float)
			if position.shape != (2,) or not np.all(np.isfinite(position)):
				continue
			candidates.append((float(np.linalg.norm(position - blue)), point.id, point))
		if not candidates:
			raise RuntimeError("No valid permanent deforming red point is available")
		return min(candidates, key=lambda item: (item[0], item[1]))[2]

	def stop_dataset(self) -> DatasetSession | None:
		session = self.dataset_session
		self.dataset_session = None
		self.pending_dataset_frames = 0
		self.pending_dataset_invalid_retries = 0
		self.pending_dataset_batches = []
		if self.mode == ApplicationMode.DATA_COLLECTION:
			self.mode = ApplicationMode.TRACKING if self.tracker else ApplicationMode.DIAGNOSTIC
		return session

	def load_force_model(
		self,
		model_path: str | Path,
		metadata_path: str | Path | None = None,
	) -> ForceModel:
		model = Path(model_path)
		if not model.is_absolute():
			model = Path(__file__).resolve().parent / model
		metadata = None if metadata_path is None else Path(metadata_path)
		if metadata is not None and not metadata.is_absolute():
			metadata = Path(__file__).resolve().parent / metadata
		self.loaded_force_model = ForceModel.load(model, metadata)
		return self.loaded_force_model

	def activate_force_model(
		self,
		model_path: str | Path,
		metadata_path: str | Path | None = None,
	) -> ForceModel:
		"""Validate a candidate completely before replacing the active model."""
		candidate = ForceModel.load(model_path, metadata_path)
		if self.reference_profile is None:
			raise RuntimeError(
				"An accepted reference profile is required to activate a model"
			)
		candidate.validate_context(
			reference_profile_id=self.reference_profile.profile_id,
			geometry_version=str(
				self.config.geometry.get("geometry_configuration_version", "2")
			),
			feature_schema_version=str(
				self.config.geometry.get("feature_schema_version", "2")
			),
		)
		if self.current_feature_names:
			candidate.validate_features(self.current_feature_names)
		# Assignment occurs only after loading and every compatibility check passes.
		self.loaded_force_model = candidate
		self.current_force_prediction = None
		return candidate

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

	def _update_blue_origin_relative_position(self) -> None:
		"""Refresh the blue target's camera-axis displacement from REFERENCE_ORIGIN."""
		self.blue_origin_relative_position = None
		blue_tracker = self.blue_tracker
		if blue_tracker is None or not blue_tracker.selected:
			self.blue_relative_unavailable_reason = "Blue target is not selected"
			return
		if not blue_tracker.valid or blue_tracker.track.current_position is None:
			self.blue_relative_unavailable_reason = (
				blue_tracker.invalid_reason
				or f"Blue target tracking is {blue_tracker.track.status.value}"
			)
			return
		if self.tracker is None or self.current_geometry is None:
			self.blue_relative_unavailable_reason = "Reference origin is not being tracked"
			return
		if not self.current_geometry.origin_valid:
			self.blue_relative_unavailable_reason = "Reference origin is invalid"
			return
		origin = self.tracker.get_point(self.tracker.origin_point_id)
		if origin is None:
			self.blue_relative_unavailable_reason = "Reference origin is unavailable"
			return
		blue_position = np.asarray(
			blue_tracker.track.current_position, dtype=float
		)
		origin_position = np.asarray(origin.current_position, dtype=float)
		if (
			blue_position.shape != (2,)
			or origin_position.shape != (2,)
			or not np.all(np.isfinite(blue_position))
			or not np.all(np.isfinite(origin_position))
		):
			self.blue_relative_unavailable_reason = "A tracked position is invalid"
			return
		relative = blue_position - origin_position
		self.blue_origin_relative_position = (
			float(relative[0]), float(relative[1])
		)
		self.blue_relative_unavailable_reason = ""

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
		blue_detections: list[dict[str, Any]] | None = None,
		motor_telemetry: dict[str, Any] | None = None,
	) -> np.ndarray:
		"""Run the single detector→tracker→geometry→consumer frame pipeline."""
		if raw_frame is None:
			raise ValueError("raw_frame cannot be None")
		self.current_frame_number += 1
		self.current_timestamp = float(time.time() if timestamp is None else timestamp)
		self.current_raw_frame = raw_frame
		self.current_gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
		self.current_detections = (
			find_red_blob_data(
				raw_frame,
				min_area=self.config.calibration.min_area,
				max_area=self.config.calibration.max_area,
			)
			if detections is None else detections
		)
		self.current_blue_detections = (
			find_blue_blob_data(
				raw_frame,
				min_area=self.config.blue_calibration.min_area,
				max_area=self.config.blue_calibration.max_area,
				config=self.config,
			)
			if blue_detections is None else blue_detections
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

		if self.blue_tracker is not None:
			self.blue_tracker.update(
				self.current_gray_frame, self.current_blue_detections
			)
		self._update_blue_origin_relative_position()

		self.process_pending_dataset_sample(motor_telemetry)
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
		self._update_blue_origin_relative_position()
		tracker = self.tracker
		geometry = self.current_geometry
		proposal = self.proposed_ransac_result
		session = self.dataset_session
		profile = self.reference_profile
		calibration_blob = self.selected_calibration_blob
		blue_calibration_blob = self.selected_blue_calibration_blob
		blue_tracker = self.blue_tracker
		blue_track = None if blue_tracker is None else blue_tracker.track
		reacquisition_required = bool(
			tracker is not None and tracker.origin_reacquisition_required
		)
		recording_reason = self._invalid_frame_reason()
		nearest_red_point_id = ""
		try:
			nearest_red_point_id = self._nearest_deforming_point_to_blue().id
		except RuntimeError:
			pass
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
			"expected_marker_count": 0 if profile is None else len(profile.points),
			"detected_blue_candidate_count": len(self.current_blue_detections),
			"geometry_valid": False if geometry is None else geometry.valid,
			"geometry_quality": 0.0 if geometry is None else geometry.quality,
			"origin_valid": False if geometry is None else geometry.origin_valid,
			"origin_tracking_quality": (
				0.0 if geometry is None
				else float(getattr(geometry, "origin_tracking_quality", 0.0))
			),
			"feature_schema_version": str(
				self.config.geometry.get("feature_schema_version", "2")
			),
			"feature_names": tuple(self.current_feature_names),
			"feature_values": (
				() if geometry is None
				else tuple(
					float(value) for value in getattr(geometry, "feature_vector", ())
				)
			),
			"runtime_warnings": list(self.runtime_warnings),
			"configuration_compatibility_notes": configuration_compatibility_notes(
				self.config
			),
			"display_options": {
				name: bool(getattr(self.display_options, name))
				for name in self.display_options.__dataclass_fields__
			},
			"calibration_mode": self.calibration_mode,
			"calibration_selection_available": calibration_blob is not None,
			"calibration_sample_hsv": (
				None if calibration_blob is None
				else calibration_blob.get("sample_hsv")
			),
			"calibration_blob_area": (
				None if calibration_blob is None
				else float(calibration_blob.get("area", 0.0))
			),
			"calibration_min_area": int(self.config.calibration.min_area),
			"calibration_max_area": self.config.calibration.max_area,
			"blue_calibration_mode": self.blue_calibration_mode,
			"blue_calibration_selection_available": blue_calibration_blob is not None,
			"blue_calibration_sample_hsv": (
				None if blue_calibration_blob is None
				else blue_calibration_blob.get("sample_hsv")
			),
			"blue_calibration_blob_area": (
				None if blue_calibration_blob is None
				else float(blue_calibration_blob.get("area", 0.0))
			),
			"blue_calibration_min_area": int(self.config.blue_calibration.min_area),
			"blue_calibration_max_area": self.config.blue_calibration.max_area,
			"blue_target_selected": bool(
				blue_tracker is not None and blue_tracker.selected
			),
			"blue_tracking_valid": bool(
				blue_tracker is not None and blue_tracker.valid
			),
			"blue_tracking_status": (
				BlueTrackStatus.UNSELECTED.value
				if blue_track is None else blue_track.status.value
			),
			"blue_tracking_quality": (
				0.0 if blue_track is None else float(blue_track.quality)
			),
			"blue_tracking_reason": (
				"" if (
					blue_tracker is None
					or not blue_tracker.selected
					or blue_tracker.valid
				)
				else blue_tracker.invalid_reason
			),
			"blue_tracking_information": (
				"" if blue_track is None else blue_track.status_reason
			),
			"blue_position": (
				None if blue_track is None or blue_track.current_position is None
				else tuple(float(value) for value in blue_track.current_position)
			),
			"blue_origin_relative_position": self.blue_origin_relative_position,
			"blue_relative_position_valid": (
				self.blue_origin_relative_position is not None
			),
			"blue_relative_unavailable_reason": self.blue_relative_unavailable_reason,
			"nearest_red_point_to_blue": nearest_red_point_id,
			"dataset_active": session is not None,
			"dataset_directory": str(self._dataset_directory()),
			"dataset_experiment_id": "" if session is None else session.experiment_id,
			"dataset_root": "" if session is None else str(session.root),
			"dataset_force_N": self.known_reference_force,
			"dataset_accepted": 0 if session is None else session.accepted,
			"dataset_rejected": 0 if session is None else session.rejected,
			"dataset_pending": self.pending_dataset_frames,
			"dataset_pending_batch": (
				None if not self.pending_dataset_batches else {
					"force_step_id": self.pending_dataset_batches[0].force_step_id,
					"known_force_N": self.pending_dataset_batches[0].known_force_N,
					"loading_direction": self.pending_dataset_batches[0].loading_direction,
					"repetition": self.pending_dataset_batches[0].repetition,
					"requested_samples": self.pending_dataset_batches[0].requested_samples,
					"remaining_samples": self.pending_dataset_batches[0].remaining_samples,
					"selected_point_id": self.pending_dataset_batches[0].selected_point_id,
				}
			),
			"dataset_last_result": self.last_dataset_result,
			"recording_valid": recording_reason is None,
			"recording_invalid_reason": recording_reason or "",
		}

	def _reject_pending_dataset_frame(
		self,
		reason: str,
		*,
		already_recorded: bool = False,
	) -> bool:
		assert self.dataset_session is not None
		batch = self.pending_dataset_batches[0]
		if not already_recorded:
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
			self._consume_pending_dataset_frame()
			self.pending_dataset_invalid_retries = 0
		else:
			self._schedule_pending_dataset_retry(batch)
		print(
			f"Dataset frame {self.current_frame_number} rejected: {reason}; "
			f"pending={self.pending_dataset_frames}"
		)
		return False

	@staticmethod
	def _validated_motor_telemetry(
		batch: DatasetBatch,
		motor_telemetry: dict[str, Any] | None,
	) -> tuple[dict[str, Any] | None, str | None]:
		if not batch.require_motor_telemetry:
			return None, None
		if not isinstance(motor_telemetry, dict):
			return None, "Fresh motor telemetry is unavailable"
		required_numeric = (
			"position_rad",
			"velocity_rad_s",
			"target_torque_Nm",
			"measured_torque_Nm",
			"temperature_C",
			"feedback_monotonic",
		)
		detached: dict[str, Any] = {}
		for key in required_numeric:
			try:
				value = float(motor_telemetry[key])
			except (KeyError, TypeError, ValueError, OverflowError):
				return None, f"Motor telemetry field {key} is unavailable"
			if not np.isfinite(value):
				return None, f"Motor telemetry field {key} is not finite"
			detached[key] = value
		state = motor_telemetry.get("state")
		if not isinstance(state, str) or not state:
			return None, "Motor telemetry state is unavailable"
		detached["state"] = state
		age = time.monotonic() - detached["feedback_monotonic"]
		if age < 0.0:
			return None, "Motor telemetry timestamp is in the future"
		limit = batch.maximum_motor_telemetry_age_s
		assert limit is not None
		if age > limit:
			return None, (
				f"Motor telemetry is stale ({age:.3f}s; limit {limit:.3f}s)"
			)
		detached["telemetry_age_s"] = age
		return detached, None

	def process_pending_dataset_sample(
		self,
		motor_telemetry: dict[str, Any] | None = None,
	) -> bool | None:
		if self.dataset_session is None or not self.pending_dataset_batches:
			return None
		batch = self.pending_dataset_batches[0]
		if time.monotonic() < batch.next_sample_monotonic_s:
			return None
		reason = self._invalid_frame_reason()
		if reason is not None:
			return self._reject_pending_dataset_frame(reason)

		blue_position: tuple[float, float] | None = None
		blue_relative: tuple[float, float] | None = None
		if batch.require_blue_target:
			self._update_blue_origin_relative_position()
			blue_tracker = self.blue_tracker
			if (
				blue_tracker is None
				or not blue_tracker.valid
				or blue_tracker.track.current_position is None
			):
				reason = (
					"Blue target is invalid"
					if blue_tracker is None else blue_tracker.invalid_reason
				)
				return self._reject_pending_dataset_frame(
					reason or "Blue target is invalid"
				)
			if self.blue_origin_relative_position is None:
				return self._reject_pending_dataset_frame(
					self.blue_relative_unavailable_reason
					or "Blue position relative to origin is unavailable"
				)
			selected = next(
				(
					point for point in self.current_tracked_points
					if point.id == batch.selected_point_id
					and point.point_group == PointGroup.DEFORMING
					and point.status not in {PointStatus.MISSING, PointStatus.INVALID}
				),
				None,
			)
			if selected is None:
				return self._reject_pending_dataset_frame(
					"The selected permanent red point is invalid"
				)
			blue_position = tuple(
				float(value) for value in blue_tracker.track.current_position
			)
			blue_relative = self.blue_origin_relative_position

		validated_motor, reason = self._validated_motor_telemetry(
			batch, motor_telemetry
		)
		if reason is not None:
			return self._reject_pending_dataset_frame(reason)

		assert self.current_geometry is not None
		assert self.tracker is not None
		assert self.reference_profile is not None
		origin = self.tracker.get_point(
			self.reference_profile.reference_origin_point_id
		)
		accepted = self.dataset_session.add_sample(
			timestamp=self.current_timestamp,
			frame_number=self.current_frame_number,
			known_force_N=batch.known_force_N,
			points=self.current_tracked_points,
			geometry=self.current_geometry,
			raw_frame=self.current_raw_frame,
			tracker_valid=self.tracker.valid,
			tracker_quality=self.tracker.quality,
			feature_schema_version=str(
				self.config.geometry.get("feature_schema_version", "2")
			),
			origin_point=origin,
			force_step_id=batch.force_step_id,
			loading_direction=batch.loading_direction,
			repetition=batch.repetition,
			selected_point_id=batch.selected_point_id,
			blue_target_position=blue_position,
			blue_origin_relative_position=blue_relative,
			motor_telemetry=validated_motor,
		)
		if accepted:
			self._consume_pending_dataset_frame()
			self.pending_dataset_invalid_retries = 0
			self.last_dataset_result = "ACCEPTED"
			print(
				f"Dataset frame {self.current_frame_number} accepted; "
				f"pending={self.pending_dataset_frames}"
			)
			return True
		reason = self.dataset_session.last_rejection_reason or "dataset validation failed"
		return self._reject_pending_dataset_frame(reason, already_recorded=True)

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

	def _draw_blue_target(self, frame: np.ndarray) -> None:
		if self.blue_calibration_mode:
			for detection in self.current_blue_detections:
				cv2.drawContours(
					frame, [detection["contour"]], -1, (255, 180, 0), 1,
					cv2.LINE_AA,
				)
		blue_tracker = self.blue_tracker
		if (
			blue_tracker is None
			or not blue_tracker.selected
			or blue_tracker.track.current_position is None
		):
			return
		position = tuple(
			int(round(value)) for value in blue_tracker.track.current_position
		)
		color = (255, 0, 0) if blue_tracker.valid else (100, 100, 255)
		cv2.circle(frame, position, 10, color, 2, cv2.LINE_AA)
		cv2.circle(frame, position, 3, color, -1, cv2.LINE_AA)
		label = f"BLUE {blue_tracker.track.status.value}"
		if self.blue_origin_relative_position is not None:
			dx, dy = self.blue_origin_relative_position
			label += f" d=({dx:+.1f}, {dy:+.1f}) px"
			if self.tracker is not None:
				origin = self.tracker.get_point(self.tracker.origin_point_id)
				if origin is not None:
					origin_position = tuple(
						int(round(value)) for value in origin.current_position
					)
					cv2.line(
						frame, origin_position, position, (255, 120, 0), 1,
						cv2.LINE_AA,
					)
		cv2.putText(
			frame, label, (position[0] + 12, position[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
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
		if self.blue_tracker is not None and self.blue_tracker.selected:
			blue = self.blue_tracker.track
			if self.blue_origin_relative_position is None:
				lines.append((
					f"BLUE: {blue.status.value} q={blue.quality:.2f}; "
					f"relative unavailable ({self.blue_relative_unavailable_reason})",
					(0, 165, 255),
				))
			else:
				dx, dy = self.blue_origin_relative_position
				lines.append((
					f"BLUE FROM ORIGIN: dx={dx:+.2f}px dy={dy:+.2f}px "
					f"q={blue.quality:.2f}",
					(255, 120, 0),
				))
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
		if self.display_options.show_blue_mask_only:
			mask = create_blue_mask(raw_frame, config=self.config)
			frame = np.zeros_like(raw_frame)
			frame[:, :, 0] = mask
		elif self.display_options.show_mask_only:
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
		self._draw_blue_target(frame)
		self._draw_geometry(frame)
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
