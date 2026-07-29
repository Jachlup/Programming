"""Origin-relative Fin Ray geometry with optional rigid-frame compensation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import cv2
import numpy as np

from tracking import PointGroup, PointStatus, ReferenceProfile, TrackedPoint


@dataclass
class TransformResult:
	valid: bool
	matrix: np.ndarray | None
	quality: float
	warning: str = ""
	applied: bool = True
	mode: str = "translation_and_rotation"


@dataclass
class GeometryResult:
	raw_tracked_coordinates: dict[str, tuple[float, float]] = field(default_factory=dict)
	compensated_coordinates: dict[str, tuple[float, float]] = field(default_factory=dict)
	origin_relative_coordinates: dict[str, tuple[float, float]] = field(default_factory=dict)
	reference_origin_relative_coordinates: dict[str, tuple[float, float]] = field(default_factory=dict)
	relative_displacements: dict[str, tuple[float, float]] = field(default_factory=dict)
	connections: dict[str, dict[str, float]] = field(default_factory=dict)
	feature_vector: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
	feature_names: list[str] = field(default_factory=list)
	origin_valid: bool = False
	origin_status: PointStatus | None = None
	origin_quality: float = 0.0
	transform_valid: bool = False
	valid: bool = False
	quality: float = 0.0
	transform: TransformResult = field(
		default_factory=lambda: TransformResult(
			False, None, 0.0, "Geometry has not been evaluated", False, "unknown"
		)
	)
	warnings: list[str] = field(default_factory=list)
	fatal_errors: list[str] = field(default_factory=list)

	# Compatibility names used by the first permanent-ID pipeline and dataset
	# readers.  All aliases reference the canonical dictionaries above.
	@property
	def raw_points(self) -> dict[str, tuple[float, float]]:
		return self.raw_tracked_coordinates

	@property
	def compensated_points(self) -> dict[str, tuple[float, float]]:
		return self.compensated_coordinates

	@property
	def origin_relative_points(self) -> dict[str, tuple[float, float]]:
		return self.origin_relative_coordinates

	@property
	def reference_origin_relative_points(self) -> dict[str, tuple[float, float]]:
		return self.reference_origin_relative_coordinates

	@property
	def unloaded_origin_relative_points(self) -> dict[str, tuple[float, float]]:
		return self.reference_origin_relative_coordinates

	@property
	def displacements(self) -> dict[str, tuple[float, float]]:
		return self.relative_displacements

	@property
	def origin_tracking_quality(self) -> float:
		return self.origin_quality


def estimate_reference_transform(
		profile: ReferenceProfile,
		points: list[TrackedPoint],
		allow_scale: bool = False,
		threshold: float = 3.0,
		minimum_tracking_quality: float = 0.0,
		allow_tracked_only: bool = True,
	) -> TransformResult:
	"""Estimate a current-image to unloaded-image rigid transform.

	Only markers explicitly declared as ``RIGID_REFERENCE`` in the profile
	participate.  The moving origin is deliberately never used to infer rotation.
	"""
	by_id = {point.id: point for point in points}
	reference: list[tuple[float, float]] = []
	current: list[tuple[float, float]] = []
	tracking_qualities: list[float] = []
	for point_id in profile.rigid_reference_point_ids:
		point = by_id.get(point_id)
		if (
			point is not None
			and point.point_group == PointGroup.RIGID_REFERENCE
			and point.status not in (PointStatus.MISSING, PointStatus.INVALID)
			and (
				point.status != PointStatus.TRACKED_ONLY
				or allow_tracked_only
			)
			and point.tracking_quality >= float(minimum_tracking_quality)
		):
			reference.append(point.reference_position)
			current.append(point.current_position)
			tracking_qualities.append(float(point.tracking_quality))
	if len(reference) < 2:
		return TransformResult(
			False, None, 0.0, "Insufficient rigid reference points",
			True, "translation_and_rotation",
		)

	reference_array = np.asarray(reference, dtype=np.float64)
	current_array = np.asarray(current, dtype=np.float64)
	if (
		np.ptp(reference_array, axis=0).dot(np.ptp(reference_array, axis=0)) <= 1e-12
		or np.ptp(current_array, axis=0).dot(np.ptp(current_array, axis=0)) <= 1e-12
	):
		return TransformResult(
			False, None, 0.0, "Rigid reference markers are geometrically degenerate",
			True, "translation_and_rotation",
		)

	matrix, inliers = cv2.estimateAffinePartial2D(
		current_array.astype(np.float32),
		reference_array.astype(np.float32),
		method=cv2.RANSAC,
		ransacReprojThreshold=float(threshold),
	)
	if matrix is None:
		return TransformResult(
			False, None, 0.0, "Rigid transform estimation failed",
			True, "translation_and_rotation",
		)
	matrix = np.asarray(matrix, dtype=np.float64)
	if not allow_scale:
		rotation = matrix[:, :2]
		scale = math.sqrt(abs(float(np.linalg.det(rotation))))
		if scale <= 1e-9:
			return TransformResult(
				False, None, 0.0, "Rigid transform scale is degenerate",
				True, "translation_and_rotation",
			)
		matrix[:, :2] = rotation / scale
		# Normalising the estimated rotation also requires a consistent
		# translation; otherwise the transform pivots around the image origin.
		matrix[:, 2] = (
			reference_array.mean(axis=0)
			- matrix[:, :2] @ current_array.mean(axis=0)
		)

	predicted_reference = (
		current_array @ matrix[:, :2].T + matrix[:, 2]
	)
	errors = np.linalg.norm(predicted_reference - reference_array, axis=1)
	inlier_ratio = float(np.mean(inliers)) if inliers is not None else 1.0
	error_scale = max(float(threshold), 1e-6)
	residual_quality = float(np.clip(1.0 - np.sqrt(np.mean(errors ** 2)) / error_scale, 0.0, 1.0))
	tracking_quality = float(np.mean(tracking_qualities))
	return TransformResult(
		True, matrix, inlier_ratio * residual_quality * tracking_quality, "",
		True, "translation_and_rotation",
	)


def _transform(
		position: tuple[float, float],
		matrix: np.ndarray,
	) -> tuple[float, float]:
	value = matrix @ np.array([position[0], position[1], 1.0], dtype=float)
	return float(value[0]), float(value[1])


def _angle_delta(current: float, reference: float) -> float:
	return float((current - reference + math.pi) % (2 * math.pi) - math.pi)


def _normalise_mode(config: dict[str, Any]) -> str:
	mode = str(
		config.get(
			"geometry_reference_mode",
			config.get("reference_mode", "translation_only"),
		)
	).strip().lower().replace("-", "_")
	aliases = {
		"translation": "translation_only",
		"origin": "translation_only",
		"rigid": "translation_and_rotation",
		"rigid_compensated": "translation_and_rotation",
		"translation_rotation": "translation_and_rotation",
	}
	return aliases.get(mode, mode)


def _ordered_deforming_points(profile: ReferenceProfile) -> list[TrackedPoint]:
	return sorted(
		(
			point for point in profile.points
			if point.point_group == PointGroup.DEFORMING
		),
		key=lambda point: (
			point.line_id or "",
			point.line_order if point.line_order is not None else math.inf,
			point.id,
		),
	)


def _connection_specs(raw_connections: Any) -> list[tuple[str, str]]:
	if isinstance(raw_connections, dict):
		values: Iterable[Any] = (
			raw_connections[key] for key in sorted(raw_connections)
		)
	else:
		values = raw_connections or []
	result: list[tuple[str, str]] = []
	for connection in values:
		if isinstance(connection, dict):
			a_id, b_id = connection.get("a"), connection.get("b")
		else:
			try:
				a_id, b_id = connection
			except (TypeError, ValueError):
				continue
		if a_id is not None and b_id is not None:
			result.append((str(a_id), str(b_id)))
	return result


def _append_unique(messages: list[str], message: str) -> None:
	if message not in messages:
		messages.append(message)


def extract_geometry(
		profile: ReferenceProfile,
		points: list[TrackedPoint],
		config: dict[str, Any],
	) -> GeometryResult:
	"""Extract a fixed-schema feature vector in the selected origin's frame."""
	result = GeometryResult()
	result.raw_tracked_coordinates = {
		point.id: tuple(float(value) for value in point.current_position)
		for point in sorted(points, key=lambda item: item.id)
	}
	runtime_by_id = {point.id: point for point in points}
	profile_by_id = {point.id: point for point in profile.points}
	if len(runtime_by_id) != len(points):
		_append_unique(result.fatal_errors, "Tracked point IDs are not unique")
	if len(profile_by_id) != len(profile.points):
		_append_unique(result.fatal_errors, "Reference profile point IDs are not unique")

	origin_id = profile.reference_origin_point_id
	origin_reference = profile.reference_origin_position
	explicit_profile_origins = [
		point for point in profile.points
		if point.point_group == PointGroup.REFERENCE_ORIGIN
	]
	if (
		origin_id is None
		or origin_reference is None
		or len(explicit_profile_origins) != 1
		or explicit_profile_origins[0].id != origin_id
	):
		_append_unique(
			result.fatal_errors,
			"Reference profile has no unambiguous explicit reference origin",
		)
	else:
		origin_reference = tuple(float(value) for value in origin_reference)
		if not np.allclose(
			explicit_profile_origins[0].reference_position,
			origin_reference,
			atol=1e-6,
		):
			_append_unique(
				result.fatal_errors,
				"Stored reference-origin position disagrees with its point definition",
			)

	if origin_reference is not None:
		origin_reference_array = np.asarray(origin_reference, dtype=float)
		for point in sorted(profile.points, key=lambda item: item.id):
			relative = np.asarray(point.reference_position, dtype=float) - origin_reference_array
			result.reference_origin_relative_coordinates[point.id] = (
				float(relative[0]), float(relative[1])
			)
		if origin_id in result.reference_origin_relative_coordinates:
			result.reference_origin_relative_coordinates[origin_id] = (0.0, 0.0)

	origin = runtime_by_id.get(origin_id) if origin_id is not None else None
	if origin is None:
		_append_unique(result.fatal_errors, "REFERENCE ORIGIN LOST: marker is unavailable")
	else:
		result.origin_status = origin.status
		result.origin_quality = float(origin.tracking_quality)
		minimum_origin_quality = float(
			config.get("minimum_origin_tracking_quality", 0.0)
		)
		status_valid = origin.status not in (PointStatus.MISSING, PointStatus.INVALID)
		tracked_only_valid = (
			origin.status != PointStatus.TRACKED_ONLY
			or bool(config.get("allow_tracked_only_origin", False))
		)
		result.origin_valid = (
			origin.point_group == PointGroup.REFERENCE_ORIGIN
			and status_valid
			and tracked_only_valid
			and result.origin_quality >= minimum_origin_quality
		)
		if not result.origin_valid:
			_append_unique(
				result.fatal_errors,
				f"REFERENCE ORIGIN LOST: status={origin.status.value}, "
				f"quality={result.origin_quality:.3f}",
			)
		elif origin.status == PointStatus.TRACKED_ONLY:
			result.warnings.append("Reference origin is optical-flow tracked only")

	mode = _normalise_mode(config)
	if mode == "translation_only":
		transform = TransformResult(
			True, np.eye(2, 3, dtype=float), 1.0, "",
			False, "translation_only",
		)
	elif mode == "translation_and_rotation":
		minimum_rigid = max(
			2, int(config.get("minimum_valid_rigid_reference_points", 2))
		)
		if len(profile.rigid_reference_point_ids) < minimum_rigid:
			transform = TransformResult(
				False, None, 0.0,
				f"At least {minimum_rigid} configured rigid reference markers are required",
				True, "translation_and_rotation",
			)
		else:
			transform = estimate_reference_transform(
				profile,
				points,
				bool(config.get("allow_scale_compensation", False)),
				float(config.get("transform_ransac_threshold_px", 3.0)),
				float(config.get(
					"minimum_rigid_reference_tracking_quality", 0.0
				)),
				bool(config.get("allow_tracked_only_rigid_reference", True)),
			)
		if not transform.valid:
			_append_unique(result.fatal_errors, transform.warning)
	else:
		transform = TransformResult(
			False, None, 0.0, f"Unknown geometry reference mode: {mode}",
			False, mode,
		)
		_append_unique(result.fatal_errors, transform.warning)
	result.transform = transform
	result.transform_valid = transform.valid

	allow_tracked_deforming = bool(
		config.get("allow_tracked_only_deforming", True)
	)
	minimum_deforming_quality = float(
		config.get("minimum_deforming_tracking_quality", 0.0)
	)
	deforming_points = _ordered_deforming_points(profile)
	for reference_point in deforming_points:
		point = runtime_by_id.get(reference_point.id)
		if point is None:
			_append_unique(
				result.fatal_errors,
				f"Missing required deforming point {reference_point.id}",
			)
			continue
		usable = point.status not in (PointStatus.MISSING, PointStatus.INVALID)
		if point.status == PointStatus.TRACKED_ONLY:
			usable = allow_tracked_deforming
			if usable:
				result.warnings.append(
					f"Deforming point {point.id} is optical-flow tracked only"
				)
		if point.tracking_quality < minimum_deforming_quality:
			usable = False
		if not usable:
			_append_unique(
				result.fatal_errors,
				f"Required deforming point {point.id} is invalid "
				f"(status={point.status.value}, quality={point.tracking_quality:.3f})",
			)

	if transform.valid and transform.matrix is not None:
		for point in sorted(points, key=lambda item: item.id):
			if point.status in (PointStatus.MISSING, PointStatus.INVALID):
				continue
			if transform.applied:
				position = _transform(point.current_position, transform.matrix)
			else:
				position = tuple(float(value) for value in point.current_position)
			result.compensated_coordinates[point.id] = position

	if (
		result.origin_valid
		and origin_id is not None
		and origin_id in result.compensated_coordinates
	):
		current_origin = np.asarray(
			result.compensated_coordinates[origin_id], dtype=float
		)
		for point_id in sorted(result.compensated_coordinates):
			relative = (
				np.asarray(result.compensated_coordinates[point_id], dtype=float)
				- current_origin
			)
			result.origin_relative_coordinates[point_id] = (
				float(relative[0]), float(relative[1])
			)
		result.origin_relative_coordinates[origin_id] = (0.0, 0.0)
		for point_id, current_relative in result.origin_relative_coordinates.items():
			reference_relative = result.reference_origin_relative_coordinates.get(point_id)
			if reference_relative is None:
				continue
			displacement = (
				np.asarray(current_relative, dtype=float)
				- np.asarray(reference_relative, dtype=float)
			)
			result.relative_displacements[point_id] = (
				float(displacement[0]), float(displacement[1])
			)
		result.relative_displacements[origin_id] = (0.0, 0.0)

	groups = set(config.get("enabled_feature_groups", [
		"point_displacements",
		"length_changes",
		"relative_length_changes",
		"angle_changes",
	]))
	known_groups = {
		"point_displacements",
		"length_changes",
		"relative_length_changes",
		"angle_changes",
	}
	for group in sorted(groups - known_groups):
		result.warnings.append(f"Unknown geometry feature group ignored: {group}")

	feature_values: list[float] = []
	if "point_displacements" in groups:
		for point in deforming_points:
			result.feature_names.extend([
				f"{point.id}.dx_px",
				f"{point.id}.dy_px",
			])
			displacement = result.relative_displacements.get(point.id)
			if displacement is None:
				feature_values.extend([math.nan, math.nan])
			else:
				feature_values.extend(displacement)

	seen_connections: set[str] = set()
	for a_id, b_id in _connection_specs(config.get("structural_connections", [])):
		key = f"{a_id}--{b_id}"
		if key in seen_connections:
			_append_unique(result.fatal_errors, f"Duplicate structural connection {key}")
			continue
		seen_connections.add(key)
		reference_a = result.reference_origin_relative_coordinates.get(a_id)
		reference_b = result.reference_origin_relative_coordinates.get(b_id)
		current_a = result.origin_relative_coordinates.get(a_id)
		current_b = result.origin_relative_coordinates.get(b_id)
		connection_values: dict[str, float] | None = None
		if (
			reference_a is None or reference_b is None
			or current_a is None or current_b is None
		):
			_append_unique(result.fatal_errors, f"Connection {key} is unavailable")
		else:
			reference_vector = np.subtract(reference_b, reference_a)
			current_vector = np.subtract(current_b, current_a)
			reference_length = float(np.linalg.norm(reference_vector))
			current_length = float(np.linalg.norm(current_vector))
			if reference_length <= 1e-9:
				_append_unique(
					result.fatal_errors,
					f"Zero-length reference connection {key}",
				)
			else:
				length_change = current_length - reference_length
				relative_length_change = length_change / reference_length
				reference_angle = math.atan2(
					reference_vector[1], reference_vector[0]
				)
				current_angle = math.atan2(current_vector[1], current_vector[0])
				connection_values = {
					"reference_length": reference_length,
					"current_length": current_length,
					"length_change": length_change,
					"relative_length_change": relative_length_change,
					"reference_angle": reference_angle,
					"current_angle": current_angle,
					"angle_change": _angle_delta(current_angle, reference_angle),
				}
				result.connections[key] = connection_values

		for group, suffix, value_name in (
			("length_changes", "length_change_px", "length_change"),
			(
				"relative_length_changes",
				"relative_length_change",
				"relative_length_change",
			),
			("angle_changes", "angle_change_rad", "angle_change"),
		):
			if group in groups:
				result.feature_names.append(f"{key}.{suffix}")
				feature_values.append(
					math.nan
					if connection_values is None
					else connection_values[value_name]
				)

	result.feature_vector = np.asarray(feature_values, dtype=float)
	if result.feature_vector.size and not np.all(np.isfinite(result.feature_vector)):
		_append_unique(result.fatal_errors, "Feature vector contains unavailable values")

	required_quality = [result.origin_quality if result.origin_valid else 0.0]
	for point in deforming_points:
		runtime_point = runtime_by_id.get(point.id)
		required_quality.append(
			0.0 if runtime_point is None else float(runtime_point.tracking_quality)
		)
	point_quality = float(np.mean(required_quality)) if required_quality else 0.0
	transform_quality = transform.quality if transform.valid else 0.0
	result.quality = point_quality * transform_quality
	result.valid = not result.fatal_errors
	return result
