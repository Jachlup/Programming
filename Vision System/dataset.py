"""Experiment-oriented storage for force-calibration samples."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Iterable

import cv2
import numpy as np
import yaml

from geometry import GeometryResult
from tracking import PointStatus, ReferenceProfile, TrackedPoint


DATASET_SCHEMA_VERSION = "3"


def _utc_now() -> datetime:
	return datetime.now(timezone.utc)


def _status_value(point: TrackedPoint) -> str:
	status = getattr(point, "status", "")
	return str(status.value if isinstance(status, Enum) else status)


def _json_ready(value: Any) -> Any:
	"""Convert NumPy/enumeration values to strict JSON-compatible values."""
	if isinstance(value, np.ndarray):
		return [_json_ready(item) for item in value.tolist()]
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Enum):
		return value.value
	if isinstance(value, dict):
		return {
			str(key): _json_ready(item)
			for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
		}
	if isinstance(value, (list, tuple)):
		return [_json_ready(item) for item in value]
	return value


def _json_dump(value: Any) -> str:
	return json.dumps(_json_ready(value), allow_nan=False, separators=(",", ":"))


def _optional_point(
	value: tuple[float, float] | None,
	label: str,
) -> tuple[float, float] | None:
	if value is None:
		return None
	point = np.asarray(value, dtype=np.float64)
	if point.shape != (2,) or not np.all(np.isfinite(point)):
		raise ValueError(f"{label} must contain two finite coordinates")
	return float(point[0]), float(point[1])


def _motor_telemetry_values(snapshot: dict[str, Any] | None) -> dict[str, Any]:
	fields = {
		"motor_position_rad": "position_rad",
		"motor_velocity_rad_s": "velocity_rad_s",
		"motor_target_torque_Nm": "target_torque_Nm",
		"motor_measured_torque_Nm": "measured_torque_Nm",
		"motor_temperature_C": "temperature_C",
		"motor_telemetry_monotonic_s": "feedback_monotonic",
		"motor_telemetry_age_s": "telemetry_age_s",
	}
	if snapshot is None:
		return {
			**{output: "" for output in fields},
			"motor_state": "",
		}
	if not isinstance(snapshot, dict):
		raise ValueError("motor telemetry must be a detached mapping")

	result: dict[str, Any] = {}
	for output, source in fields.items():
		value = snapshot.get(source)
		if value is None or value == "":
			result[output] = ""
			continue
		try:
			numeric = float(value)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError(f"{source} is not numeric") from exc
		if not math.isfinite(numeric):
			raise ValueError(f"{source} is not finite")
		if source == "telemetry_age_s" and numeric < 0.0:
			raise ValueError("telemetry_age_s cannot be negative")
		result[output] = numeric
	state = snapshot.get("state", "")
	if state is None:
		state = ""
	if isinstance(state, Enum):
		state = state.value
	if not isinstance(state, str):
		raise ValueError("motor state must be text")
	result["motor_state"] = state
	return result


def _coordinates_from_points(points: Iterable[TrackedPoint]) -> dict[str, tuple[float, float]]:
	return {
		point.id: (float(point.current_position[0]), float(point.current_position[1]))
		for point in sorted(points, key=lambda item: item.id)
	}


def _profile_origin_id(profile: ReferenceProfile) -> str:
	origin_id = getattr(profile, "reference_origin_point_id", "")
	if origin_id:
		return str(origin_id)
	for point in getattr(profile, "points", []):
		group = getattr(point, "point_group", "")
		group_value = group.value if isinstance(group, Enum) else str(group)
		if group_value == "REFERENCE_ORIGIN":
			return str(point.id)
	return ""


def _safe_identifier(value: str, label: str) -> str:
	identifier = value.strip()
	if not identifier or identifier in {".", ".."}:
		raise ValueError(f"{label} must not be empty")
	if Path(identifier).name != identifier:
		raise ValueError(f"{label} must be a single path-safe name")
	return identifier


@dataclass
class DatasetSession:
	"""One non-overwriting acquisition batch and its accepted sample CSV."""

	root: Path
	experiment_id: str
	profile_id: str
	geometry_id: str
	save_images: bool = True
	save_raw_coordinates: bool = True
	save_compensated_coordinates: bool = True
	save_origin_relative_coordinates: bool = True
	feature_schema_version: str = "1"
	reference_origin_point_id: str = ""
	acquisition_batch_id: str = ""
	profile_configuration_version: str = "1"
	structural_connection_version: str = "1"
	accepted: int = 0
	rejected: int = 0
	rejection_reasons: dict[str, int] = field(default_factory=dict)
	last_rejection_reason: str = ""
	_csv_initialised: bool = False

	CSV_FIELDS: ClassVar[tuple[str, ...]] = (
		"dataset_schema_version",
		"feature_schema_version",
		"timestamp",
		"frame_number",
		"experiment_id",
		"acquisition_batch_id",
		"force_step_id",
		"loading_direction",
		"repetition",
		"sample_id",
		"known_force_N",
		"tracking_valid",
		"tracking_quality",
		"origin_valid",
		"reference_origin_point_id",
		"raw_origin_position",
		"compensated_origin_position",
		"origin_status",
		"origin_tracking_quality",
		"geometry_valid",
		"geometry_quality",
		"reference_transformation_valid",
		"reference_transformation_applied",
		"reference_transformation_mode",
		"reference_quality",
		"reference_transformation_warning",
		"raw_point_coordinates",
		"compensated_point_coordinates",
		"origin_relative_point_coordinates",
		"unloaded_origin_relative_point_coordinates",
		"relative_point_displacements",
		"point_statuses",
		"point_tracking_qualities",
		"geometry_warnings",
		"geometry_fatal_errors",
		"feature_names",
		"feature_values",
		"selected_point_id",
		"selected_point_origin_relative_position",
		"blue_target_position",
		"blue_origin_relative_position",
		"motor_position_rad",
		"motor_velocity_rad_s",
		"motor_target_torque_Nm",
		"motor_measured_torque_Nm",
		"motor_temperature_C",
		"motor_state",
		"motor_telemetry_monotonic_s",
		"motor_telemetry_age_s",
		"image_filename",
		"reference_profile_identifier",
		"reference_profile_configuration_version",
		"structural_connection_version",
		"geometry_configuration_identifier",
	)

	@classmethod
	def start(
		cls,
		dataset_directory: str,
		experiment_id: str,
		profile: ReferenceProfile,
		geometry_id: str,
		save_images: bool = True,
		*,
		feature_schema_version: str = "1",
		acquisition_batch_id: str | None = None,
		reference_origin_point_id: str | None = None,
		save_raw_coordinates: bool = True,
		save_compensated_coordinates: bool = True,
		save_origin_relative_coordinates: bool = True,
	) -> "DatasetSession":
		"""Create a fresh acquisition directory without overwriting prior samples."""
		experiment = _safe_identifier(str(experiment_id), "experiment_id")
		base = Path(dataset_directory) / experiment
		base.mkdir(parents=True, exist_ok=True)

		explicit_batch = acquisition_batch_id is not None
		batch = _safe_identifier(
			str(acquisition_batch_id)
			if explicit_batch
			else _utc_now().strftime("%Y%m%dT%H%M%S%fZ"),
			"acquisition_batch_id",
		)
		root = base / batch
		if explicit_batch:
			root.mkdir(parents=False, exist_ok=False)
		else:
			suffix = 1
			while True:
				try:
					root.mkdir(parents=False, exist_ok=False)
					break
				except FileExistsError:
					root = base / f"{batch}-{suffix:02d}"
					suffix += 1
			batch = root.name

		if save_images:
			(root / "images").mkdir(exist_ok=False)

		origin_id = (
			str(reference_origin_point_id)
			if reference_origin_point_id is not None
			else _profile_origin_id(profile)
		)
		metadata = {
			"dataset_schema_version": DATASET_SCHEMA_VERSION,
			"feature_schema_version": str(feature_schema_version),
			"experiment_id": experiment,
			"acquisition_batch_id": batch,
			"created_at": _utc_now().isoformat(),
			"reference_profile_identifier": str(profile.profile_id),
			"reference_profile_configuration_version": str(profile.configuration_version),
			"reference_origin_point_id": origin_id,
			"reference_profile_image_resolution": list(profile.image_resolution),
			"reference_profile_expected_points": dict(profile.expected_points),
			"reference_profile_quality": float(profile.quality),
			"structural_connection_version": str(
				profile.structural_connection_version
			),
			"geometry_configuration_identifier": str(geometry_id),
			"save_images": bool(save_images),
			"save_raw_coordinates": bool(save_raw_coordinates),
			"save_compensated_coordinates": bool(save_compensated_coordinates),
			"save_origin_relative_coordinates": bool(
				save_origin_relative_coordinates
			),
		}
		with (root / "metadata.yaml").open("x", encoding="utf-8") as stream:
			yaml.safe_dump(metadata, stream, sort_keys=False)

		return cls(
			root=root,
			experiment_id=experiment,
			profile_id=str(profile.profile_id),
			geometry_id=str(geometry_id),
			save_images=bool(save_images),
			save_raw_coordinates=bool(save_raw_coordinates),
			save_compensated_coordinates=bool(save_compensated_coordinates),
			save_origin_relative_coordinates=bool(
				save_origin_relative_coordinates
			),
			feature_schema_version=str(feature_schema_version),
			reference_origin_point_id=origin_id,
			acquisition_batch_id=batch,
			profile_configuration_version=str(profile.configuration_version),
			structural_connection_version=str(profile.structural_connection_version),
		)

	def record_rejection(
		self,
		reason: str,
		*,
		frame_number: int | None = None,
		timestamp: float | None = None,
	) -> None:
		"""Record an invalid sampling attempt without consuming a sample ID."""
		clean_reason = str(reason).strip() or "unspecified"
		self.rejected += 1
		self.last_rejection_reason = clean_reason
		self.rejection_reasons[clean_reason] = self.rejection_reasons.get(clean_reason, 0) + 1
		path = self.root / "rejections.csv"
		write_header = not path.exists()
		with path.open("a", newline="", encoding="utf-8") as stream:
			writer = csv.DictWriter(stream, fieldnames=("timestamp", "frame_number", "reason"))
			if write_header:
				writer.writeheader()
			writer.writerow({
				"timestamp": _utc_now().timestamp() if timestamp is None else timestamp,
				"frame_number": "" if frame_number is None else frame_number,
				"reason": clean_reason,
			})

	def add_sample(
		self,
		*,
		timestamp: float,
		frame_number: int,
		known_force_N: float,
		points: list[TrackedPoint],
		geometry: GeometryResult,
		raw_frame: Any,
		tracker_valid: bool | None = None,
		tracker_quality: float | None = None,
		feature_schema_version: str | None = None,
		origin_point: TrackedPoint | None = None,
		origin_valid: bool | None = None,
		force_step_id: str | int = "",
		loading_direction: str = "unspecified",
		repetition: int = 1,
		selected_point_id: str = "",
		blue_target_position: tuple[float, float] | None = None,
		blue_origin_relative_position: tuple[float, float] | None = None,
		motor_telemetry: dict[str, Any] | None = None,
	) -> bool:
		"""Validate and append one complete force-sensing sample."""
		try:
			actual_timestamp = float(timestamp)
			actual_frame_number = int(frame_number)
			actual_force = float(known_force_N)
		except (TypeError, ValueError, OverflowError):
			return self._reject("timestamp, frame number, or force is invalid", 0, 0.0)
		if not math.isfinite(actual_timestamp) or not math.isfinite(actual_force):
			return self._reject(
				"timestamp or known force is not finite",
				actual_frame_number,
				actual_timestamp if math.isfinite(actual_timestamp) else 0.0,
			)
		step_id = str(force_step_id).strip()
		if not step_id:
			return self._reject(
				"force_step_id is empty", actual_frame_number, actual_timestamp
			)
		direction = str(loading_direction).strip().lower()
		if direction not in {"loading", "unloading", "baseline", "unspecified"}:
			return self._reject(
				"loading direction is invalid", actual_frame_number, actual_timestamp
			)
		try:
			repetition_value = int(repetition)
		except (TypeError, ValueError, OverflowError):
			return self._reject(
				"repetition is invalid", actual_frame_number, actual_timestamp
			)
		if repetition_value < 1:
			return self._reject(
				"repetition must be positive", actual_frame_number, actual_timestamp
			)

		schema_version = (
			self.feature_schema_version
			if feature_schema_version is None
			else str(feature_schema_version)
		)
		if schema_version != self.feature_schema_version:
			return self._reject(
				f"feature schema {schema_version!r} does not match session "
				f"{self.feature_schema_version!r}",
				actual_frame_number,
				actual_timestamp,
			)

		if not points:
			return self._reject("no tracked points", actual_frame_number, actual_timestamp)
		statuses = {point.id: _status_value(point) for point in points}
		invalid_statuses = {PointStatus.MISSING.value, PointStatus.INVALID.value}
		derived_tracker_valid = all(status not in invalid_statuses for status in statuses.values())
		actual_tracker_valid = derived_tracker_valid if tracker_valid is None else bool(tracker_valid)
		if not actual_tracker_valid:
			return self._reject("tracker is invalid", actual_frame_number, actual_timestamp)

		if tracker_quality is None:
			actual_tracker_quality = float(np.mean([
				float(point.tracking_quality) for point in points
			]))
		else:
			actual_tracker_quality = float(tracker_quality)
		if not math.isfinite(actual_tracker_quality):
			return self._reject(
				"tracker quality is not finite", actual_frame_number, actual_timestamp
			)

		if not bool(getattr(geometry, "valid", False)):
			return self._reject("geometry is invalid", actual_frame_number, actual_timestamp)

		origin_id = self.reference_origin_point_id
		if origin_point is None and origin_id:
			origin_point = next((point for point in points if point.id == origin_id), None)
		if origin_point is None:
			return self._reject(
				"reference origin is unavailable", actual_frame_number, actual_timestamp
			)
		if origin_id and origin_point.id != origin_id:
			return self._reject(
				"reference origin identity does not match the profile",
				actual_frame_number,
				actual_timestamp,
			)
		if not origin_id:
			origin_id = str(origin_point.id)

		origin_status = _status_value(origin_point)
		derived_origin_valid = origin_status not in invalid_statuses
		geometry_origin_valid = getattr(geometry, "origin_valid", None)
		if geometry_origin_valid is not None:
			derived_origin_valid = derived_origin_valid and bool(geometry_origin_valid)
		actual_origin_valid = (
			derived_origin_valid
			if origin_valid is None
			else derived_origin_valid and bool(origin_valid)
		)
		if not actual_origin_valid:
			return self._reject(
				"reference origin is invalid", actual_frame_number, actual_timestamp
			)

		origin_quality = float(origin_point.tracking_quality)
		if not math.isfinite(origin_quality):
			return self._reject(
				"reference origin quality is not finite", actual_frame_number, actual_timestamp
			)

		feature_names = list(getattr(geometry, "feature_names", []))
		feature_values = np.asarray(
			getattr(geometry, "feature_vector", np.empty(0)), dtype=np.float64
		)
		if feature_values.ndim != 1 or len(feature_names) != feature_values.size:
			return self._reject(
				"feature names and values have incompatible shapes",
				actual_frame_number,
				actual_timestamp,
			)
		if not feature_names or not all(isinstance(name, str) and name for name in feature_names):
			return self._reject(
				"feature names are empty or invalid", actual_frame_number, actual_timestamp
			)
		if len(set(feature_names)) != len(feature_names):
			return self._reject(
				"feature names are not unique", actual_frame_number, actual_timestamp
			)
		if not np.all(np.isfinite(feature_values)):
			return self._reject(
				"feature values are not finite", actual_frame_number, actual_timestamp
			)

		try:
			raw_points = getattr(geometry, "raw_points", None) or _coordinates_from_points(points)
			compensated_points = dict(getattr(geometry, "compensated_points", {}))
			origin_relative_points = dict(getattr(geometry, "origin_relative_points", {}))
			unloaded_relative_points = dict(
				getattr(geometry, "reference_origin_relative_points", {})
				or getattr(geometry, "unloaded_origin_relative_points", {})
				or getattr(geometry, "reference_relative_points", {})
			)
			relative_displacements = dict(
				getattr(geometry, "relative_displacements", {})
				or getattr(geometry, "displacements", {})
			)
			raw_origin_position = (
				float(origin_point.current_position[0]),
				float(origin_point.current_position[1]),
			)
			compensated_origin_position = compensated_points.get(origin_id)
			if compensated_origin_position is not None:
				compensated_origin_position = (
					float(compensated_origin_position[0]),
					float(compensated_origin_position[1]),
				)
			selected_id = str(selected_point_id).strip()
			selected_relative_position = None
			if selected_id:
				if selected_id not in origin_relative_points:
					return self._reject(
						"selected red point is unavailable in origin-relative geometry",
						actual_frame_number,
						actual_timestamp,
					)
				selected = origin_relative_points[selected_id]
				selected_relative_position = (float(selected[0]), float(selected[1]))
			blue_target = _optional_point(blue_target_position, "blue target position")
			blue_relative = _optional_point(
				blue_origin_relative_position, "blue origin-relative position"
			)
		except (TypeError, ValueError, IndexError) as exc:
			return self._reject(
				f"coordinate serialization failed: {exc}",
				actual_frame_number,
				actual_timestamp,
			)

		transform = getattr(geometry, "transform", None)
		transform_valid = bool(getattr(transform, "valid", False))
		transform_quality = float(getattr(transform, "quality", 0.0))
		transform_warning = str(getattr(transform, "warning", ""))
		geometry_quality = float(getattr(geometry, "quality", 0.0))
		if not math.isfinite(transform_quality) or not math.isfinite(geometry_quality):
			return self._reject(
				"geometry quality is not finite", actual_frame_number, actual_timestamp
			)

		sample_id = self.accepted + 1
		point_qualities = {
			point.id: float(point.tracking_quality)
			for point in sorted(points, key=lambda item: item.id)
		}
		if not all(math.isfinite(value) for value in point_qualities.values()):
			return self._reject(
				"point tracking quality is not finite",
				actual_frame_number,
				actual_timestamp,
			)
		try:
			motor_values = _motor_telemetry_values(motor_telemetry)
			serialised = {
				"raw_origin_position": (
					_json_dump(raw_origin_position) if self.save_raw_coordinates else ""
				),
				"compensated_origin_position": (
					_json_dump(compensated_origin_position)
					if self.save_compensated_coordinates else ""
				),
				"raw_point_coordinates": (
					_json_dump(raw_points) if self.save_raw_coordinates else ""
				),
				"compensated_point_coordinates": (
					_json_dump(compensated_points)
					if self.save_compensated_coordinates else ""
				),
				"origin_relative_point_coordinates": (
					_json_dump(origin_relative_points)
					if self.save_origin_relative_coordinates else ""
				),
				"unloaded_origin_relative_point_coordinates": (
					_json_dump(unloaded_relative_points)
					if self.save_origin_relative_coordinates else ""
				),
				"relative_point_displacements": (
					_json_dump(relative_displacements)
					if self.save_origin_relative_coordinates else ""
				),
				"point_statuses": _json_dump(statuses),
				"point_tracking_qualities": _json_dump(point_qualities),
				"geometry_warnings": _json_dump(getattr(geometry, "warnings", [])),
				"geometry_fatal_errors": _json_dump(
					getattr(geometry, "fatal_errors", [])
				),
				"feature_names": _json_dump(feature_names),
				"feature_values": _json_dump(feature_values),
				"selected_point_origin_relative_position": _json_dump(
					selected_relative_position
				),
				"blue_target_position": _json_dump(blue_target),
				"blue_origin_relative_position": _json_dump(blue_relative),
			}
		except (TypeError, ValueError) as exc:
			return self._reject(
				f"sample contains non-serializable data: {exc}",
				actual_frame_number,
				actual_timestamp,
			)

		image_name = ""
		if self.save_images:
			if raw_frame is None:
				return self._reject(
					"raw frame is unavailable", actual_frame_number, actual_timestamp
				)
			file_name = (
				f"sample_{sample_id:06d}_frame_{actual_frame_number:06d}.png"
			)
			image_path = self.root / "images" / file_name
			if image_path.exists():
				return self._reject(
					"raw image path already exists", actual_frame_number, actual_timestamp
				)
			if not cv2.imwrite(str(image_path), raw_frame):
				return self._reject(
					"raw image could not be written", actual_frame_number, actual_timestamp
				)
			image_name = str(Path("images") / file_name)

		row = {
			"dataset_schema_version": DATASET_SCHEMA_VERSION,
			"feature_schema_version": schema_version,
			"timestamp": actual_timestamp,
			"frame_number": actual_frame_number,
			"experiment_id": self.experiment_id,
			"acquisition_batch_id": self.acquisition_batch_id,
			"force_step_id": step_id,
			"loading_direction": direction,
			"repetition": repetition_value,
			"sample_id": sample_id,
			"known_force_N": actual_force,
			"tracking_valid": actual_tracker_valid,
			"tracking_quality": actual_tracker_quality,
			"origin_valid": actual_origin_valid,
			"reference_origin_point_id": origin_id,
			"raw_origin_position": serialised["raw_origin_position"],
			"compensated_origin_position": serialised["compensated_origin_position"],
			"origin_status": origin_status,
			"origin_tracking_quality": origin_quality,
			"geometry_valid": bool(geometry.valid),
			"geometry_quality": geometry_quality,
			"reference_transformation_valid": transform_valid,
			"reference_transformation_applied": bool(
				getattr(transform, "applied", False)
			),
			"reference_transformation_mode": str(
				getattr(transform, "mode", "unknown")
			),
			"reference_quality": transform_quality,
			"reference_transformation_warning": transform_warning,
			"raw_point_coordinates": serialised["raw_point_coordinates"],
			"compensated_point_coordinates": serialised[
				"compensated_point_coordinates"
			],
			"origin_relative_point_coordinates": serialised[
				"origin_relative_point_coordinates"
			],
			"unloaded_origin_relative_point_coordinates": serialised[
				"unloaded_origin_relative_point_coordinates"
			],
			"relative_point_displacements": serialised[
				"relative_point_displacements"
			],
			"point_statuses": serialised["point_statuses"],
			"point_tracking_qualities": serialised["point_tracking_qualities"],
			"geometry_warnings": serialised["geometry_warnings"],
			"geometry_fatal_errors": serialised["geometry_fatal_errors"],
			"feature_names": serialised["feature_names"],
			"feature_values": serialised["feature_values"],
			"selected_point_id": selected_id,
			"selected_point_origin_relative_position": serialised[
				"selected_point_origin_relative_position"
			],
			"blue_target_position": serialised["blue_target_position"],
			"blue_origin_relative_position": serialised[
				"blue_origin_relative_position"
			],
			**motor_values,
			"image_filename": image_name,
			"reference_profile_identifier": self.profile_id,
			"reference_profile_configuration_version": self.profile_configuration_version,
			"structural_connection_version": self.structural_connection_version,
			"geometry_configuration_identifier": self.geometry_id,
		}
		self._append_row(row)
		self.accepted += 1
		return True

	def _reject(self, reason: str, frame_number: int, timestamp: float) -> bool:
		self.record_rejection(reason, frame_number=frame_number, timestamp=timestamp)
		return False

	def _append_row(self, row: dict[str, Any]) -> None:
		path = self.root / "samples.csv"
		if not self._csv_initialised and path.exists():
			with path.open(newline="", encoding="utf-8") as stream:
				reader = csv.reader(stream)
				existing_header = tuple(next(reader, ()))
			if existing_header != self.CSV_FIELDS:
				raise ValueError("Existing samples.csv uses a different schema")
			self._csv_initialised = True

		mode = "a" if self._csv_initialised else "x"
		with path.open(mode, newline="", encoding="utf-8") as stream:
			writer = csv.DictWriter(stream, fieldnames=self.CSV_FIELDS)
			if not self._csv_initialised:
				writer.writeheader()
			writer.writerow(row)
		self._csv_initialised = True
