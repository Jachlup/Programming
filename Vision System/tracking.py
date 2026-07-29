"""Reference assignment and permanent-ID runtime tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Dependency-free Hungarian assignment for a rectangular cost matrix."""
	if cost.size == 0:
		return np.empty(0, dtype=int), np.empty(0, dtype=int)
	transposed = cost.shape[0] > cost.shape[1]
	matrix = cost.T if transposed else cost
	n, m = matrix.shape
	u = np.zeros(n + 1)
	v = np.zeros(m + 1)
	p = np.zeros(m + 1, dtype=int)
	way = np.zeros(m + 1, dtype=int)
	for i in range(1, n + 1):
		p[0] = i
		j0 = 0
		minimum = np.full(m + 1, np.inf)
		used = np.zeros(m + 1, dtype=bool)
		while True:
			used[j0] = True
			i0 = p[j0]
			delta, j1 = np.inf, 0
			for j in range(1, m + 1):
				if not used[j]:
					current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
					if current < minimum[j]:
						minimum[j], way[j] = current, j0
					if minimum[j] < delta:
						delta, j1 = minimum[j], j
			for j in range(m + 1):
				if used[j]:
					u[p[j]] += delta
					v[j] -= delta
				else:
					minimum[j] -= delta
			j0 = j1
			if p[j0] == 0:
				break
		while True:
			j1 = way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0:
				break
	rows, cols = [], []
	for j in range(1, m + 1):
		if p[j]:
			rows.append(p[j] - 1)
			cols.append(j - 1)
	if transposed:
		return np.asarray(cols, dtype=int), np.asarray(rows, dtype=int)
	return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


class PointStatus(str, Enum):
	DETECTED = "DETECTED"
	TRACKED_ONLY = "TRACKED_ONLY"
	RECOVERED = "RECOVERED"
	MISSING = "MISSING"
	INVALID = "INVALID"


class PointGroup(str, Enum):
	DEFORMING = "DEFORMING"
	RIGID_REFERENCE = "RIGID_REFERENCE"


@dataclass
class TrackedPoint:
	id: str
	name: str
	point_group: PointGroup
	line_id: str | None
	line_order: int | None
	reference_position: tuple[float, float]
	previous_position: tuple[float, float]
	current_position: tuple[float, float]
	predicted_position: tuple[float, float]
	detected_position: tuple[float, float] | None = None
	status: PointStatus = PointStatus.DETECTED
	tracking_quality: float = 1.0
	missing_frame_count: int = 0
	reference_line_residual: float = 0.0


@dataclass
class ReferenceLine:
	line_id: str
	anchor: tuple[float, float]
	direction: tuple[float, float]
	point_ids: list[str]
	residuals: list[float]
	mean_residual: float
	length: float


@dataclass
class RansacResult:
	valid: bool
	points: list[TrackedPoint] = field(default_factory=list)
	lines: list[ReferenceLine] = field(default_factory=list)
	rejected_detection_indices: list[int] = field(default_factory=list)
	quality: float = 0.0
	warnings: list[str] = field(default_factory=list)


@dataclass
class ReferenceProfile:
	profile_id: str
	created_at: str
	image_resolution: tuple[int, int]
	expected_points: dict[str, int]
	points: list[TrackedPoint]
	lines: list[ReferenceLine]
	rigid_reference_point_ids: list[str]
	structural_connection_version: str
	configuration_version: str
	quality: float


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	anchor = points.mean(axis=0)
	_, _, vt = np.linalg.svd(points - anchor)
	direction = vt[0] / np.linalg.norm(vt[0])
	return anchor, direction


def _distances(points: np.ndarray, anchor: np.ndarray, direction: np.ndarray) -> np.ndarray:
	delta = points - anchor
	return np.abs(delta[:, 0] * direction[1] - delta[:, 1] * direction[0])


def _normalise_direction(direction: np.ndarray, rule: str) -> np.ndarray:
	d = direction / np.linalg.norm(direction)
	if rule in ("base_to_tip", "left_to_right") and d[0] < 0:
		d = -d
	elif rule == "right_to_left" and d[0] > 0:
		d = -d
	elif rule == "top_to_bottom" and d[1] < 0:
		d = -d
	elif rule == "bottom_to_top" and d[1] > 0:
		d = -d
	return d


def _ransac(points: np.ndarray, threshold: float, iterations: int, minimum: int,
			rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	best: np.ndarray | None = None
	best_error = math.inf
	for _ in range(iterations):
		i, j = rng.choice(len(points), 2, replace=False)
		d = points[j] - points[i]
		if np.linalg.norm(d) < 1e-6:
			continue
		d /= np.linalg.norm(d)
		mask = _distances(points, points[i], d) <= threshold
		error = float(_distances(points[mask], points[i], d).sum())
		if best is None or int(mask.sum()) > int(best.sum()) or (
			int(mask.sum()) == int(best.sum()) and error < best_error
		):
			best, best_error = mask, error
	if best is None or int(best.sum()) < minimum:
		return None
	anchor, direction = _fit_line(points[best])
	residuals = _distances(points, anchor, direction)
	return anchor, direction, residuals <= threshold


def initialise_reference(
	detections: list[dict[str, Any]],
	config: dict[str, Any],
	image_resolution: tuple[int, int],
	rigid_detection_indices: Iterable[int] = (),
) -> RansacResult:
	"""Fit exactly two unloaded lines and propose deterministic permanent IDs."""
	rigid_indices = set(int(i) for i in rigid_detection_indices)
	deforming_indices = [i for i in range(len(detections)) if i not in rigid_indices]
	xy = np.asarray([detections[i]["center"] for i in deforming_indices], dtype=np.float64)
	expected = [int(config["expected_points_line_a"]), int(config["expected_points_line_b"])]
	warnings: list[str] = []
	if int(config.get("expected_line_count", 2)) != 2:
		return RansacResult(False, warnings=["This initializer requires exactly two reference lines"])
	configured_total = config.get("expected_deforming_point_count")
	if configured_total is not None and int(configured_total) != sum(expected):
		return RansacResult(False, warnings=["Configured deforming total does not equal LINE_A + LINE_B"])
	if len(xy) < sum(expected):
		return RansacResult(False, warnings=["Too few deforming detections"])

	remaining_local = np.arange(len(xy))
	fits = []
	rng = np.random.default_rng(0)
	for _ in range(2):
		if len(remaining_local) < min(expected):
			return RansacResult(False, warnings=["Insufficient points for second line"])
		fit = _ransac(
			xy[remaining_local],
			float(config.get("distance_threshold_px", 10.0)),
			int(config.get("iterations", 300)),
			max(min(expected), int(config.get("minimum_inliers_per_line", 2))),
			rng,
		)
		if fit is None:
			return RansacResult(False, warnings=["RANSAC could not find two valid lines"])
		anchor, direction, mask = fit
		selected = remaining_local[mask]
		fits.append((anchor, direction, selected))
		remaining_local = remaining_local[~mask]

	# Assign line identity spatially, independent of which line RANSAC found first.
	rule = str(config.get("direction_rule", "base_to_tip"))
	fits = [(a, _normalise_direction(d, rule), ids) for a, d, ids in fits]
	normal = np.array([-fits[0][1][1], fits[0][1][0]])
	fits.sort(key=lambda fit: float(np.dot(fit[0], normal)))
	angle = math.degrees(math.acos(np.clip(abs(np.dot(fits[0][1], fits[1][1])), 0, 1)))
	offset = fits[1][0] - fits[0][0]
	separation = abs(float(offset[0] * fits[0][1][1] - offset[1] * fits[0][1][0]))
	if angle > float(config.get("maximum_angle_difference_deg", 15.0)):
		warnings.append(f"Line angle difference {angle:.1f} exceeds limit")
	if separation < float(config.get("minimum_line_separation_px", 20.0)):
		warnings.append(f"Line separation {separation:.1f}px is too small")
	max_sep = config.get("maximum_line_separation_px")
	if max_sep is not None and separation > float(max_sep):
		warnings.append(f"Line separation {separation:.1f}px is too large")
	if len(remaining_local) > int(config.get("maximum_outlier_count", 3)):
		warnings.append("Too many rejected outliers")
	elif len(remaining_local) >= int(config.get("minimum_inliers_per_line", 2)):
		extra = _ransac(
			xy[remaining_local], float(config.get("distance_threshold_px", 10.0)),
			int(config.get("iterations", 300)),
			int(config.get("minimum_inliers_per_line", 2)), rng,
		)
		if extra is not None:
			warnings.append("More than two plausible deforming-point lines were found")

	points: list[TrackedPoint] = []
	lines: list[ReferenceLine] = []
	used: set[int] = set()
	for line_index, (anchor, direction, local_ids) in enumerate(fits):
		line_id = "LINE_A" if line_index == 0 else "LINE_B"
		required = expected[line_index]
		if len(local_ids) != required:
			warnings.append(f"{line_id} has {len(local_ids)} points; expected {required}")
			continue
		projections = (xy[local_ids] - anchor) @ direction
		order = np.argsort(projections)
		sorted_ids = local_ids[order]
		sorted_proj = projections[order]
		if len(sorted_proj) > 1 and np.min(np.diff(sorted_proj)) < float(config.get("minimum_order_spacing_px", 2.0)):
			warnings.append(f"{line_id} point ordering is ambiguous")
		residuals = _distances(xy[sorted_ids], anchor, direction)
		if float(np.mean(residuals)) > float(config.get("maximum_mean_residual_px", 5.0)):
			warnings.append(f"{line_id} residual is too large")
		names = []
		for line_order, (local_id, residual) in enumerate(zip(sorted_ids, residuals)):
			original_id = deforming_indices[int(local_id)]
			if original_id in used:
				warnings.append("A detection belongs to both lines")
				continue
			used.add(original_id)
			name = f"{line_id}_{line_order:02d}"
			pos = tuple(float(v) for v in detections[original_id]["center"])
			names.append(name)
			points.append(TrackedPoint(name, name, PointGroup.DEFORMING, line_id, line_order,
									   pos, pos, pos, pos, pos,
									   reference_line_residual=float(residual)))
		lines.append(ReferenceLine(
			line_id, tuple(map(float, anchor)), tuple(map(float, direction)), names,
			[float(v) for v in residuals], float(np.mean(residuals)),
			float(np.ptp(sorted_proj)) if len(sorted_proj) else 0.0,
		))

	for rigid_order, detection_index in enumerate(sorted(rigid_indices)):
		if detection_index >= len(detections):
			warnings.append("Rigid reference marker index is invalid")
			continue
		name = f"RIGID_{rigid_order:02d}"
		pos = tuple(float(v) for v in detections[detection_index]["center"])
		points.append(TrackedPoint(name, name, PointGroup.RIGID_REFERENCE, None, rigid_order,
								   pos, pos, pos, pos, pos))
	min_rigid = int(config.get("minimum_valid_rigid_reference_points", 2))
	if len(rigid_indices) < min_rigid:
		warnings.append(f"At least {min_rigid} rigid reference markers are required")
	rejected = [deforming_indices[int(i)] for i in remaining_local]
	mean_residual = np.mean([line.mean_residual for line in lines]) if lines else math.inf
	quality = float(np.clip(1.0 - mean_residual / max(float(config.get("distance_threshold_px", 10)), 1), 0, 1))
	return RansacResult(not warnings, points, lines, rejected, quality, warnings)


def make_reference_profile(result: RansacResult, image_resolution: tuple[int, int],
						   config: dict[str, Any], geometry_config: dict[str, Any]) -> ReferenceProfile:
	if not result.valid:
		raise ValueError("Cannot accept an invalid reference proposal")
	now = datetime.now(timezone.utc)
	return ReferenceProfile(
		f"finray-{now.strftime('%Y%m%dT%H%M%SZ')}", now.isoformat(), image_resolution,
		{"LINE_A": int(config["expected_points_line_a"]),
		 "LINE_B": int(config["expected_points_line_b"])},
		result.points, result.lines,
		[p.id for p in result.points if p.point_group == PointGroup.RIGID_REFERENCE],
		str(geometry_config.get("structural_connection_version", "1")),
		str(config.get("configuration_version", "1")), result.quality,
	)


def _encode(value: Any) -> Any:
	if isinstance(value, Enum):
		return value.value
	if hasattr(value, "__dataclass_fields__"):
		return {key: _encode(item) for key, item in asdict(value).items()}
	if isinstance(value, (list, tuple)):
		return [_encode(item) for item in value]
	if isinstance(value, dict):
		return {key: _encode(item) for key, item in value.items()}
	return value


def save_reference_profile(profile: ReferenceProfile, path: str | Path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(_encode(profile), indent=2), encoding="utf-8")


def load_reference_profile(path: str | Path) -> ReferenceProfile:
	raw = json.loads(Path(path).read_text(encoding="utf-8"))
	raw["points"] = [TrackedPoint(
		**{**p, "point_group": PointGroup(p["point_group"]), "status": PointStatus(p["status"]),
		   "reference_position": tuple(p["reference_position"]), "previous_position": tuple(p["previous_position"]),
		   "current_position": tuple(p["current_position"]), "predicted_position": tuple(p["predicted_position"]),
		   "detected_position": None if p["detected_position"] is None else tuple(p["detected_position"])}
	) for p in raw["points"]]
	raw["lines"] = [ReferenceLine(**{**line, "anchor": tuple(line["anchor"]),
									 "direction": tuple(line["direction"])}) for line in raw["lines"]]
	raw["image_resolution"] = tuple(raw["image_resolution"])
	return ReferenceProfile(**raw)


class PointTracker:
	"""Lucas–Kanade prediction followed by detector-corrected global assignment."""

	def __init__(self, profile: ReferenceProfile, config: dict[str, Any]):
		self.points = profile.points
		self.config = config
		self.previous_gray: np.ndarray | None = None
		self.valid = True
		self.quality = 1.0

	def update(self, gray: np.ndarray, detections: list[dict[str, Any]]) -> list[TrackedPoint]:
		previous = np.asarray([p.current_position for p in self.points], np.float32).reshape(-1, 1, 2)
		flow_ok = np.zeros(len(self.points), dtype=bool)
		flow_error = np.full(len(self.points), np.inf)
		predicted = previous.reshape(-1, 2).copy()
		if self.previous_gray is not None and len(previous):
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
						int(self.config.get("termination_count", 20)),
						float(self.config.get("termination_epsilon", 0.03)))
			next_pts, status, error = cv2.calcOpticalFlowPyrLK(
				self.previous_gray, gray, previous, None,
				winSize=tuple(self.config.get("window_size", [21, 21])),
				maxLevel=int(self.config.get("maximum_pyramid_level", 3)), criteria=criteria,
			)
			if next_pts is not None:
				predicted = next_pts.reshape(-1, 2)
				flow_ok = status.reshape(-1).astype(bool)
				flow_error = error.reshape(-1)

		centres = np.asarray([d["center"] for d in detections], dtype=float)
		assigned: dict[int, int] = {}
		max_distance = float(self.config.get("maximum_assignment_distance", 35.0))
		if len(centres) and len(predicted):
			cost = np.linalg.norm(predicted[:, None, :] - centres[None, :, :], axis=2)
			rows, cols = _linear_sum_assignment(cost)
			assigned = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] <= max_distance}

		max_flow_error = float(self.config.get("maximum_optical_flow_error", 25.0))
		max_missing = int(self.config.get("maximum_missing_frames", 5))
		for i, point in enumerate(self.points):
			was_missing = point.missing_frame_count > 0
			point.previous_position = point.current_position
			point.predicted_position = tuple(map(float, predicted[i]))
			if i in assigned and float(detections[assigned[i]].get("detection_quality", 1.0)) > 0:
				pos = tuple(map(float, centres[assigned[i]]))
				point.current_position = pos
				point.detected_position = pos
				point.status = PointStatus.RECOVERED if was_missing else PointStatus.DETECTED
				point.missing_frame_count = 0
				distance_q = 1.0 - min(np.linalg.norm(predicted[i] - centres[assigned[i]]) / max_distance, 1.0)
				point.tracking_quality = float(distance_q * detections[assigned[i]].get("detection_quality", 1.0))
			elif flow_ok[i] and flow_error[i] <= max_flow_error and point.missing_frame_count < max_missing:
				point.current_position = point.predicted_position
				point.detected_position = None
				point.status = PointStatus.TRACKED_ONLY
				point.missing_frame_count += 1
				point.tracking_quality = float(max(0.0, 1.0 - flow_error[i] / max_flow_error) * 0.5)
			else:
				point.detected_position = None
				point.missing_frame_count += 1
				point.status = PointStatus.MISSING if point.missing_frame_count <= max_missing else PointStatus.INVALID
				point.tracking_quality = 0.0
		# A very large adjacent-point stretch is an identity/association failure,
		# not normal departure from the unloaded reference line.
		max_ratio = float(self.config.get("maximum_adjacent_distance_ratio", 3.0))
		for line_id in ("LINE_A", "LINE_B"):
			line_points = sorted(
				(p for p in self.points if p.line_id == line_id),
				key=lambda p: p.line_order if p.line_order is not None else -1,
			)
			for first, second in zip(line_points, line_points[1:]):
				ref_distance = np.linalg.norm(
					np.subtract(second.reference_position, first.reference_position)
				)
				now_distance = np.linalg.norm(
					np.subtract(second.current_position, first.current_position)
				)
				if ref_distance > 1e-6 and now_distance / ref_distance > max_ratio:
					first.status = second.status = PointStatus.INVALID
					first.tracking_quality = second.tracking_quality = 0.0
		self.previous_gray = gray.copy()
		valid_points = sum(p.status in (PointStatus.DETECTED, PointStatus.RECOVERED) for p in self.points)
		self.quality = float(np.mean([p.tracking_quality for p in self.points])) if self.points else 0.0
		self.valid = valid_points >= int(self.config.get("minimum_valid_points", len(self.points)))
		return self.points


def draw_tracking(frame: np.ndarray, points: list[TrackedPoint]) -> np.ndarray:
	colors = {PointStatus.DETECTED: (0, 255, 0), PointStatus.RECOVERED: (0, 255, 255),
			  PointStatus.TRACKED_ONLY: (255, 180, 0), PointStatus.MISSING: (0, 0, 255),
			  PointStatus.INVALID: (80, 80, 80)}
	for point in points:
		x, y = (int(round(v)) for v in point.current_position)
		cv2.circle(frame, (x, y), 5, colors[point.status], 2)
		cv2.putText(frame, point.name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
					0.35, colors[point.status], 1, cv2.LINE_AA)
	return frame
