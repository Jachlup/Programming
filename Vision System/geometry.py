"""Rigid-reference compensation and deterministic Fin Ray geometry features."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import cv2
import numpy as np

from tracking import PointGroup, PointStatus, ReferenceProfile, TrackedPoint


@dataclass
class TransformResult:
	valid: bool
	matrix: np.ndarray | None
	quality: float
	warning: str = ""


@dataclass
class GeometryResult:
	compensated_points: dict[str, tuple[float, float]] = field(default_factory=dict)
	displacements: dict[str, tuple[float, float]] = field(default_factory=dict)
	connections: dict[str, dict[str, float]] = field(default_factory=dict)
	feature_vector: np.ndarray = field(default_factory=lambda: np.empty(0))
	feature_names: list[str] = field(default_factory=list)
	valid: bool = False
	quality: float = 0.0
	transform: TransformResult = field(default_factory=lambda: TransformResult(False, None, 0.0))
	warnings: list[str] = field(default_factory=list)


def estimate_reference_transform(profile: ReferenceProfile, points: list[TrackedPoint],
								 allow_scale: bool = False, threshold: float = 3.0) -> TransformResult:
	by_id = {p.id: p for p in points}
	reference, current = [], []
	for point_id in profile.rigid_reference_point_ids:
		point = by_id.get(point_id)
		if point and point.status not in (PointStatus.MISSING, PointStatus.INVALID):
			reference.append(point.reference_position)
			current.append(point.current_position)
	if len(reference) < 2:
		return TransformResult(False, None, 0.0, "Insufficient rigid reference points")
	matrix, inliers = cv2.estimateAffinePartial2D(
		np.asarray(current, np.float32), np.asarray(reference, np.float32),
		method=cv2.RANSAC, ransacReprojThreshold=threshold,
	)
	if matrix is None:
		return TransformResult(False, None, 0.0, "Rigid transform estimation failed")
	if not allow_scale:
		rotation = matrix[:, :2]
		scale = math.sqrt(abs(float(np.linalg.det(rotation))))
		if scale > 1e-9:
			matrix[:, :2] /= scale
	quality = float(np.mean(inliers)) if inliers is not None else 1.0
	return TransformResult(True, matrix, quality)


def _transform(position: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
	value = matrix @ np.array([position[0], position[1], 1.0])
	return float(value[0]), float(value[1])


def _angle_delta(current: float, reference: float) -> float:
	return float((current - reference + math.pi) % (2 * math.pi) - math.pi)


def extract_geometry(profile: ReferenceProfile, points: list[TrackedPoint],
					 config: dict[str, Any]) -> GeometryResult:
	transform = estimate_reference_transform(
		profile, points, bool(config.get("allow_scale_compensation", False)),
		float(config.get("transform_ransac_threshold_px", 3.0)),
	)
	result = GeometryResult(transform=transform)
	if not transform.valid or transform.matrix is None:
		result.warnings.append(transform.warning)
		return result
	by_id = {p.id: p for p in points}
	for point in sorted(points, key=lambda p: p.id):
		if point.status in (PointStatus.MISSING, PointStatus.INVALID):
			continue
		compensated = _transform(point.current_position, transform.matrix)
		result.compensated_points[point.id] = compensated
		result.displacements[point.id] = (
			compensated[0] - point.reference_position[0],
			compensated[1] - point.reference_position[1],
		)
	groups = set(config.get("enabled_feature_groups", [
		"point_displacements", "length_changes", "relative_length_changes", "angle_changes"
	]))
	deforming = sorted((p for p in profile.points if p.point_group == PointGroup.DEFORMING),
					   key=lambda p: (p.line_id or "", p.line_order or 0))
	if "point_displacements" in groups:
		for point in deforming:
			if point.id not in result.displacements:
				result.warnings.append(f"Missing geometry point {point.id}")
				continue
			dx, dy = result.displacements[point.id]
			result.feature_names.extend([f"{point.id}.dx_px", f"{point.id}.dy_px"])
			result.feature_vector = np.append(result.feature_vector, [dx, dy])
	for connection in config.get("structural_connections", []):
		a_id, b_id = connection if isinstance(connection, list) else (connection["a"], connection["b"])
		if a_id not in result.compensated_points or b_id not in result.compensated_points:
			result.warnings.append(f"Connection {a_id}--{b_id} is unavailable")
			continue
		ref_a, ref_b = by_id[a_id].reference_position, by_id[b_id].reference_position
		cur_a, cur_b = result.compensated_points[a_id], result.compensated_points[b_id]
		ref_vector = np.subtract(ref_b, ref_a)
		cur_vector = np.subtract(cur_b, cur_a)
		ref_length, cur_length = float(np.linalg.norm(ref_vector)), float(np.linalg.norm(cur_vector))
		if ref_length <= 1e-9:
			result.warnings.append(f"Zero-length reference connection {a_id}--{b_id}")
			continue
		change = cur_length - ref_length
		relative = change / ref_length
		angle_change = _angle_delta(math.atan2(cur_vector[1], cur_vector[0]),
									math.atan2(ref_vector[1], ref_vector[0]))
		key = f"{a_id}--{b_id}"
		result.connections[key] = {
			"reference_length": ref_length, "current_length": cur_length,
			"length_change": change, "relative_length_change": relative,
			"reference_angle": math.atan2(ref_vector[1], ref_vector[0]),
			"current_angle": math.atan2(cur_vector[1], cur_vector[0]),
			"angle_change": angle_change,
		}
		for group, suffix, value in (
			("length_changes", "length_change_px", change),
			("relative_length_changes", "relative_length_change", relative),
			("angle_changes", "angle_change_rad", angle_change),
		):
			if group in groups:
				result.feature_names.append(f"{key}.{suffix}")
				result.feature_vector = np.append(result.feature_vector, value)
	result.valid = not result.warnings
	point_quality = [p.tracking_quality for p in points if p.id in result.compensated_points]
	result.quality = transform.quality * (float(np.mean(point_quality)) if point_quality else 0.0)
	return result
