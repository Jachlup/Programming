"""Reference assignment and permanent-ID runtime tracking."""

from __future__ import annotations

import copy
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
	REFERENCE_ORIGIN = "REFERENCE_ORIGIN"
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
	fatal_errors: list[str] = field(default_factory=list)
	reference_origin_detection_index: int | None = None

	@property
	def reference_origin_point(self) -> TrackedPoint | None:
		"""Return the proposed real origin marker, if one was selected."""
		origins = [point for point in self.points if point.point_group == PointGroup.REFERENCE_ORIGIN]
		return origins[0] if len(origins) == 1 else None

	@property
	def origin_point(self) -> TrackedPoint | None:
		return self.reference_origin_point

	@property
	def origin_detection_index(self) -> int | None:
		return self.reference_origin_detection_index


@dataclass
class ReferenceProfile:
	profile_id: str
	created_at: str
	image_resolution: tuple[int, int]
	expected_points: dict[str, int]
	points: list[TrackedPoint]
	lines: list[ReferenceLine]
	reference_origin_point_id: str | None
	reference_origin_position: tuple[float, float] | None
	rigid_reference_point_ids: list[str]
	structural_connection_version: str
	configuration_version: str
	quality: float

	@property
	def reference_origin_reference_position(self) -> tuple[float, float] | None:
		return self.reference_origin_position

	@property
	def unloaded_reference_origin_position(self) -> tuple[float, float] | None:
		return self.reference_origin_position


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
	epsilon = 1e-8
	if rule in ("base_to_tip", "left_to_right"):
		if d[0] < -epsilon or (abs(d[0]) <= epsilon and d[1] < 0):
			d = -d
	elif rule == "right_to_left":
		if d[0] > epsilon or (abs(d[0]) <= epsilon and d[1] > 0):
			d = -d
	elif rule == "top_to_bottom":
		if d[1] < -epsilon or (abs(d[1]) <= epsilon and d[0] < 0):
			d = -d
	elif rule == "bottom_to_top":
		if d[1] > epsilon or (abs(d[1]) <= epsilon and d[0] > 0):
			d = -d
	elif d[0] < -epsilon or (abs(d[0]) <= epsilon and d[1] < 0):
		# Unknown rules still receive a deterministic SVD sign.
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
		reference_origin_detection_index: int | None = None,
	) -> RansacResult:
	"""Fit two unloaded lines while assigning all marker roles permanent IDs.

	Detection indices are setup-only identifiers.  The returned points use stable
	IDs and the selected origin is never offered to either RANSAC fit.
	"""
	warnings: list[str] = []
	fatal_errors: list[str] = []
	points: list[TrackedPoint] = []
	lines: list[ReferenceLine] = []

	try:
		expected = [
			int(config["expected_points_line_a"]),
			int(config["expected_points_line_b"]),
		]
	except (KeyError, TypeError, ValueError):
		return RansacResult(
			False,
			fatal_errors=["Expected LINE_A and LINE_B point counts are not configured"],
			reference_origin_detection_index=reference_origin_detection_index,
		)

	raw_rigid_indices = {int(index) for index in rigid_detection_indices}
	invalid_rigid_indices = sorted(
		index for index in raw_rigid_indices if index < 0 or index >= len(detections)
	)
	if invalid_rigid_indices:
		fatal_errors.append(
			f"Rigid reference detection indices are invalid: {invalid_rigid_indices}"
		)
	rigid_indices = {
		index for index in raw_rigid_indices if 0 <= index < len(detections)
	}

	origin_index: int | None = None
	if reference_origin_detection_index is None:
		fatal_errors.append("Exactly one reference-origin detection must be selected")
	else:
		candidate = int(reference_origin_detection_index)
		if candidate < 0 or candidate >= len(detections):
			fatal_errors.append("Reference-origin detection index is invalid")
		else:
			origin_index = candidate
			if origin_index in rigid_indices:
				fatal_errors.append(
					"Reference origin and rigid reference must be separate detections"
				)
				rigid_indices.discard(origin_index)
			position = tuple(float(value) for value in detections[origin_index]["center"])
			origin_quality = float(
				np.clip(detections[origin_index].get("detection_quality", 1.0), 0.0, 1.0)
			)
			points.append(TrackedPoint(
				id="REFERENCE_ORIGIN",
				name="REFERENCE_ORIGIN",
				point_group=PointGroup.REFERENCE_ORIGIN,
				line_id=None,
				line_order=None,
				reference_position=position,
				previous_position=position,
				current_position=position,
				predicted_position=position,
				detected_position=position,
				tracking_quality=origin_quality,
			))

	excluded_indices = set(rigid_indices)
	if origin_index is not None:
		excluded_indices.add(origin_index)
	deforming_indices = [
		index for index in range(len(detections)) if index not in excluded_indices
	]
	xy = np.asarray(
		[detections[index]["center"] for index in deforming_indices],
		dtype=np.float64,
	).reshape(-1, 2)

	if int(config.get("expected_line_count", 2)) != 2:
		fatal_errors.append("This initializer requires exactly two reference lines")
	configured_total = config.get("expected_deforming_point_count")
	if configured_total is not None and int(configured_total) != sum(expected):
		fatal_errors.append("Configured deforming total does not equal LINE_A + LINE_B")
	if len(xy) < sum(expected):
		fatal_errors.append(
			f"Too few deforming detections: found {len(xy)}, expected {sum(expected)}"
		)
		return RansacResult(
			False, points=points, warnings=warnings, fatal_errors=fatal_errors,
			reference_origin_detection_index=origin_index,
		)

	remaining_local = np.arange(len(xy))
	fits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
	rng = np.random.default_rng(0)
	for _ in range(2):
		if len(remaining_local) < min(expected):
			fatal_errors.append("Insufficient points for second line")
			return RansacResult(
				False, points=points, warnings=warnings, fatal_errors=fatal_errors,
				reference_origin_detection_index=origin_index,
			)
		fit = _ransac(
			xy[remaining_local],
			float(config.get("distance_threshold_px", 10.0)),
			int(config.get("iterations", 300)),
			max(min(expected), int(config.get("minimum_inliers_per_line", 2))),
			rng,
		)
		if fit is None:
			fatal_errors.append("RANSAC could not find two valid lines")
			return RansacResult(
				False, points=points, warnings=warnings, fatal_errors=fatal_errors,
				reference_origin_detection_index=origin_index,
			)
		anchor, direction, mask = fit
		selected = remaining_local[mask]
		fits.append((anchor, direction, selected))
		remaining_local = remaining_local[~mask]

	# Assign line identity spatially, independent of which line RANSAC found first.
	rule = str(config.get("direction_rule", "base_to_tip"))
	fits = [(a, _normalise_direction(d, rule), ids) for a, d, ids in fits]
	mean_direction = fits[0][1] + fits[1][1]
	if np.linalg.norm(mean_direction) < 1e-9:
		mean_direction = fits[0][1]
	mean_direction = _normalise_direction(mean_direction, rule)
	normal = np.array([-mean_direction[1], mean_direction[0]])
	fits.sort(key=lambda fit: float(np.dot(fit[0], normal)))
	angle = math.degrees(math.acos(np.clip(abs(np.dot(fits[0][1], fits[1][1])), 0, 1)))
	offset = fits[1][0] - fits[0][0]
	separation = abs(float(offset[0] * fits[0][1][1] - offset[1] * fits[0][1][0]))
	if angle > float(config.get("maximum_angle_difference_deg", 15.0)):
		fatal_errors.append(f"Line angle difference {angle:.1f} exceeds limit")
	if separation < float(config.get("minimum_line_separation_px", 20.0)):
		fatal_errors.append(f"Line separation {separation:.1f}px is too small")
	max_sep = config.get("maximum_line_separation_px")
	if max_sep is not None and separation > float(max_sep):
		fatal_errors.append(f"Line separation {separation:.1f}px is too large")
	if len(remaining_local) > int(config.get("maximum_outlier_count", 3)):
		fatal_errors.append("Too many rejected outliers")
	elif len(remaining_local) >= int(config.get("minimum_inliers_per_line", 2)):
		extra = _ransac(
			xy[remaining_local], float(config.get("distance_threshold_px", 10.0)),
			int(config.get("iterations", 300)),
			int(config.get("minimum_inliers_per_line", 2)), rng,
		)
		if extra is not None:
			fatal_errors.append("More than two plausible deforming-point lines were found")
	if len(remaining_local):
		warnings.append(f"Rejected {len(remaining_local)} RANSAC outlier detection(s)")

	used: set[int] = set()
	for line_index, (anchor, direction, local_ids) in enumerate(fits):
		line_id = "LINE_A" if line_index == 0 else "LINE_B"
		required = expected[line_index]
		if len(local_ids) != required:
			fatal_errors.append(
				f"{line_id} has {len(local_ids)} points; expected {required}"
			)
		projections = (xy[local_ids] - anchor) @ direction
		order = np.argsort(projections, kind="stable")
		sorted_ids = local_ids[order]
		sorted_proj = projections[order]
		if len(sorted_proj) > 1 and np.min(np.diff(sorted_proj)) < float(config.get("minimum_order_spacing_px", 2.0)):
			fatal_errors.append(f"{line_id} point ordering is ambiguous")
		residuals = _distances(xy[sorted_ids], anchor, direction)
		if float(np.mean(residuals)) > float(config.get("maximum_mean_residual_px", 5.0)):
			fatal_errors.append(f"{line_id} residual is too large")
		names = []
		for line_order, (local_id, residual) in enumerate(zip(sorted_ids, residuals)):
			original_id = deforming_indices[int(local_id)]
			if original_id in used:
				fatal_errors.append("A detection belongs to both lines")
				continue
			used.add(original_id)
			name = f"{line_id}_{line_order:02d}"
			pos = tuple(float(v) for v in detections[original_id]["center"])
			names.append(name)
			points.append(TrackedPoint(
				id=name,
				name=name,
				point_group=PointGroup.DEFORMING,
				line_id=line_id,
				line_order=line_order,
				reference_position=pos,
				previous_position=pos,
				current_position=pos,
				predicted_position=pos,
				detected_position=pos,
				tracking_quality=float(np.clip(
					detections[original_id].get("detection_quality", 1.0), 0.0, 1.0
				)),
				reference_line_residual=float(residual),
			))
		lines.append(ReferenceLine(
			line_id, tuple(map(float, anchor)), tuple(map(float, direction)), names,
			[float(v) for v in residuals], float(np.mean(residuals)),
			float(np.ptp(sorted_proj)) if len(sorted_proj) else 0.0,
		))

	ordered_rigid_indices = sorted(
		rigid_indices,
		key=lambda index: (
			float(detections[index]["center"][0]),
			float(detections[index]["center"][1]),
		),
	)
	for rigid_order, detection_index in enumerate(ordered_rigid_indices):
		name = f"RIGID_{rigid_order:02d}"
		pos = tuple(float(v) for v in detections[detection_index]["center"])
		points.append(TrackedPoint(
			id=name,
			name=name,
			point_group=PointGroup.RIGID_REFERENCE,
			line_id=None,
			line_order=rigid_order,
			reference_position=pos,
			previous_position=pos,
			current_position=pos,
			predicted_position=pos,
			detected_position=pos,
			tracking_quality=float(np.clip(
				detections[detection_index].get("detection_quality", 1.0), 0.0, 1.0
			)),
		))

	mode = str(config.get("geometry_reference_mode", "translation_only")).lower().replace("-", "_")
	min_rigid = int(config.get("minimum_valid_rigid_reference_points", 2))
	if mode in ("translation_and_rotation", "rigid", "rigid_compensated") and len(rigid_indices) < min_rigid:
		fatal_errors.append(
			f"At least {min_rigid} rigid reference markers are required in {mode} mode"
		)
	rejected = [deforming_indices[int(i)] for i in remaining_local]
	mean_residual = np.mean([line.mean_residual for line in lines]) if lines else math.inf
	quality = float(np.clip(1.0 - mean_residual / max(float(config.get("distance_threshold_px", 10)), 1), 0, 1))
	return RansacResult(
		valid=not fatal_errors,
		points=points,
		lines=lines,
		rejected_detection_indices=rejected,
		quality=quality,
		warnings=warnings,
		fatal_errors=fatal_errors,
		reference_origin_detection_index=origin_index,
	)


def make_reference_profile(result: RansacResult, image_resolution: tuple[int, int],
						   config: dict[str, Any], geometry_config: dict[str, Any]) -> ReferenceProfile:
	if not result.valid:
		detail = "; ".join(result.fatal_errors) or "proposal validation failed"
		raise ValueError(f"Cannot accept an invalid reference proposal: {detail}")
	origin_points = [
		point for point in result.points
		if point.point_group == PointGroup.REFERENCE_ORIGIN
	]
	if len(origin_points) != 1:
		raise ValueError("Reference profile requires exactly one reference-origin marker")
	origin = origin_points[0]
	if origin.id != "REFERENCE_ORIGIN":
		raise ValueError("Reference-origin marker must use permanent ID REFERENCE_ORIGIN")
	now = datetime.now(timezone.utc)
	return ReferenceProfile(
		f"finray-{now.strftime('%Y%m%dT%H%M%SZ')}", now.isoformat(), image_resolution,
		{"LINE_A": int(config["expected_points_line_a"]),
		 "LINE_B": int(config["expected_points_line_b"])},
		result.points, result.lines,
		origin.id, origin.reference_position,
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
	if not isinstance(raw, dict):
		raise ValueError("Invalid reference profile: root must be a JSON object")
	try:
		raw_points = raw.get("points", [])
		explicit_origins = [
			point for point in raw_points
			if point.get("point_group") == PointGroup.REFERENCE_ORIGIN.value
		]
		if "reference_origin_point_id" not in raw:
			raw["reference_origin_point_id"] = (
				explicit_origins[0].get("id") if len(explicit_origins) == 1 else None
			)
		if "reference_origin_position" not in raw:
			origin_id = raw.get("reference_origin_point_id")
			explicit_match = next(
				(
					point for point in raw_points
					if origin_id is not None
					and point.get("id") == origin_id
					and point.get("point_group") == PointGroup.REFERENCE_ORIGIN.value
				),
				None,
			)
			raw["reference_origin_position"] = (
				explicit_match.get("reference_position")
				if explicit_match is not None else None
			)
		raw["points"] = [TrackedPoint(
			**{
				**point,
				"point_group": PointGroup(point["point_group"]),
				"status": PointStatus(point["status"]),
				"reference_position": tuple(point["reference_position"]),
				"previous_position": tuple(point["previous_position"]),
				"current_position": tuple(point["current_position"]),
				"predicted_position": tuple(point["predicted_position"]),
				"detected_position": (
					None if point["detected_position"] is None
					else tuple(point["detected_position"])
				),
			}
		) for point in raw["points"]]
		raw["lines"] = [ReferenceLine(**{
			**line,
			"anchor": tuple(line["anchor"]),
			"direction": tuple(line["direction"]),
		}) for line in raw["lines"]]
		raw["image_resolution"] = tuple(raw["image_resolution"])
		if raw["reference_origin_position"] is not None:
			raw["reference_origin_position"] = tuple(raw["reference_origin_position"])
		return ReferenceProfile(**raw)
	except (AttributeError, KeyError, IndexError, TypeError) as exc:
		raise ValueError(f"Invalid reference profile structure: {exc}") from exc


class PointTracker:
	"""Lucas–Kanade prediction followed by detector-corrected global assignment."""

	def __init__(self, profile: ReferenceProfile, config: dict[str, Any]):
		self.profile = profile
		self.points = copy.deepcopy(profile.points)
		for point in self.points:
			position = tuple(float(value) for value in point.reference_position)
			point.previous_position = position
			point.current_position = position
			point.predicted_position = position
			point.detected_position = position
			point.status = PointStatus.DETECTED
			point.missing_frame_count = 0
		self.origin_point_id = profile.reference_origin_point_id
		self._points_by_id = {point.id: point for point in self.points}
		origin = self.get_point(self.origin_point_id)
		if (
			self.origin_point_id is None
			or origin is None
			or origin.point_group != PointGroup.REFERENCE_ORIGIN
		):
			raise ValueError("Reference profile has no explicit reference-origin marker")
		self.required_point_ids = [
			point.id for point in self.points
			if point.point_group in (PointGroup.REFERENCE_ORIGIN, PointGroup.DEFORMING)
		]
		self.config = config
		reacquisition_policy = str(
			config.get("origin_reacquisition_policy", "manual_confirmation")
		).strip().lower()
		if reacquisition_policy not in {
			"manual",
			"manual_confirmation",
			"click_confirmation",
			"automatic",
			"automatic_proximity",
		}:
			raise ValueError(
				f"Unknown origin_reacquisition_policy: {reacquisition_policy}"
			)
		self.previous_gray: np.ndarray | None = None
		self.valid = True
		self.quality = 1.0

	def get_point(self, point_id: str | None) -> TrackedPoint | None:
		"""Return a tracked point by permanent ID without positional assumptions."""
		if point_id is None:
			return None
		return self._points_by_id.get(point_id)

	def _manual_origin_confirmation_required(self, point: TrackedPoint) -> bool:
		policy = str(
			self.config.get("origin_reacquisition_policy", "manual_confirmation")
		).strip().lower()
		return (
			policy in {"manual", "manual_confirmation", "click_confirmation"}
			and point.point_group == PointGroup.REFERENCE_ORIGIN
			and point.status in (PointStatus.MISSING, PointStatus.INVALID)
		)

	@property
	def origin_reacquisition_required(self) -> bool:
		origin = self.get_point(self.origin_point_id)
		return (
			origin is not None
			and self._manual_origin_confirmation_required(origin)
		)

	def _assignment_gate(self, point: TrackedPoint, default: float) -> float:
		if point.point_group != PointGroup.REFERENCE_ORIGIN:
			return default
		origin_default = float(self.config.get(
			"origin_reacquisition_max_distance_px", default
		))
		if point.status in (PointStatus.MISSING, PointStatus.INVALID):
			gate = float(self.config.get(
				"origin_reacquisition_max_distance_px",
				self.config.get("origin_maximum_assignment_distance", default),
			))
			identity_lock = float(
				self.config.get("origin_identity_lock_distance_px", gate)
			)
			return min(gate, identity_lock)
		return float(self.config.get(
			"origin_maximum_assignment_distance", origin_default
		))

	def _point_is_required_and_valid(self, point: TrackedPoint) -> bool:
		if point.status in (PointStatus.MISSING, PointStatus.INVALID):
			return False
		if point.point_group == PointGroup.REFERENCE_ORIGIN:
			minimum_quality = float(
				self.config.get("minimum_origin_tracking_quality", 0.0)
			)
			if point.tracking_quality < minimum_quality:
				return False
			if point.status == PointStatus.TRACKED_ONLY:
				return bool(self.config.get("allow_tracked_only_origin", False))
			return True
		if point.status == PointStatus.TRACKED_ONLY:
			if not bool(self.config.get("allow_tracked_only_points", False)):
				return False
			return point.tracking_quality >= float(
				self.config.get("minimum_tracked_only_quality", 0.0)
			)
		return True

	def _refresh_validity(self) -> None:
		required_points = [
			self._points_by_id[point_id] for point_id in self.required_point_ids
			if point_id in self._points_by_id
		]
		self.valid = (
			len(required_points) == len(self.required_point_ids)
			and all(self._point_is_required_and_valid(point) for point in required_points)
		)
		self.quality = (
			float(np.mean([point.tracking_quality for point in required_points]))
			if required_points else 0.0
		)

	def confirm_origin_reacquisition(
		self,
		detection: dict[str, Any],
	) -> TrackedPoint:
		"""Explicitly bind a user-confirmed detection back to the permanent origin ID."""
		origin = self.get_point(self.origin_point_id)
		if origin is None:
			raise RuntimeError("The tracker has no reference-origin point")
		if not self.origin_reacquisition_required:
			raise RuntimeError("The reference origin is not awaiting manual reacquisition")
		try:
			position_array = np.asarray(detection["center"], dtype=np.float64)
			quality = float(detection.get("detection_quality", 1.0))
		except (KeyError, TypeError, ValueError, OverflowError) as exc:
			raise ValueError("The selected origin detection is invalid") from exc
		if (
			position_array.shape != (2,)
			or not np.all(np.isfinite(position_array))
			or not math.isfinite(quality)
			or quality <= 0.0
		):
			raise ValueError("The selected origin detection is invalid")
		position = (float(position_array[0]), float(position_array[1]))
		origin.previous_position = origin.current_position
		origin.predicted_position = position
		origin.current_position = position
		origin.detected_position = position
		origin.status = PointStatus.RECOVERED
		origin.tracking_quality = float(np.clip(quality, 0.0, 1.0))
		origin.missing_frame_count = 0
		self._refresh_validity()
		return origin

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

		centres = np.asarray(
			[detection["center"] for detection in detections], dtype=float
		).reshape(-1, 2)
		assigned: dict[int, int] = {}
		max_distance = float(self.config.get("maximum_assignment_distance", 35.0))
		max_flow_error = float(self.config.get("maximum_optical_flow_error", 25.0))
		if len(centres) and len(predicted):
			distance = np.linalg.norm(predicted[:, None, :] - centres[None, :, :], axis=2)
			gates = np.asarray([
				max(self._assignment_gate(point, max_distance), 1e-6)
				for point in self.points
			])
			normalised_cost = distance / gates[:, None]
			normalised_cost[normalised_cost > 1.0] = 1e6

			origin_row = next(
				(
					index for index, point in enumerate(self.points)
					if point.point_group == PointGroup.REFERENCE_ORIGIN
				),
				None,
			)
			if origin_row is not None:
				minimum_detection_quality = float(
					self.config.get("minimum_origin_detection_quality", 0.0)
				)
				for column, detection in enumerate(detections):
					if float(detection.get("detection_quality", 1.0)) < minimum_detection_quality:
						normalised_cost[origin_row, column] = 1e6
				origin = self.points[origin_row]
				if self._manual_origin_confirmation_required(origin):
					normalised_cost[origin_row, :] = 1e6
				elif self.previous_gray is None:
					initial_gate = float(self.config.get(
						"origin_initialization_max_distance_px",
						self.config.get("origin_identity_lock_distance_px", 2.0),
					))
					normalised_cost[
						origin_row,
						distance[origin_row] > initial_gate,
					] = 1e6
				elif (
					not flow_ok[origin_row]
					or flow_error[origin_row] > max_flow_error
				):
					# Without LK continuity, accept only detector jitter at the
					# unchanged predicted location. Any actual displacement
					# transitions to the explicit-confirmation lost state.
					no_flow_tolerance = float(self.config.get(
						"origin_no_flow_max_distance_px", 0.25
					))
					normalised_cost[
						origin_row,
						distance[origin_row] > no_flow_tolerance,
					] = 1e6
				elif (
					origin.missing_frame_count > 0
					and bool(self.config.get(
						"origin_reacquisition_requires_unambiguous_match", True
					))
					and len(self.points) > 1
				):
					margin = float(
						self.config.get("origin_reacquisition_ambiguity_margin_px", 2.0)
					)
					other_rows = [
						index for index in range(len(self.points)) if index != origin_row
					]
					for column in range(len(centres)):
						other_distance = float(np.min(distance[other_rows, column]))
						if other_distance + margin < distance[origin_row, column]:
							normalised_cost[origin_row, column] = 1e6

			# Dummy columns let the global optimizer choose "unmatched" instead of
			# forcing a bad detection-to-ID pairing.
			dummy_cost = np.full((len(self.points), len(self.points)), 1.05)
			assignment_cost = np.concatenate((normalised_cost, dummy_cost), axis=1)
			rows, cols = _linear_sum_assignment(assignment_cost)
			assigned = {
				int(row): int(column)
				for row, column in zip(rows, cols)
				if column < len(centres) and normalised_cost[row, column] <= 1.0
			}

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
				gate = float(gates[i])
				quality_gate = (
					max(
						gate,
						float(self.config.get(
							"origin_maximum_assignment_distance", gate
						)),
					)
					if point.point_group == PointGroup.REFERENCE_ORIGIN
					else gate
				)
				distance_q = 1.0 - min(
					np.linalg.norm(predicted[i] - centres[assigned[i]])
					/ quality_gate,
					1.0,
				)
				point.tracking_quality = float(
					distance_q * detections[assigned[i]].get("detection_quality", 1.0)
				)
			else:
				point_max_missing = (
					int(self.config.get("origin_maximum_missing_frames", max_missing))
					if point.point_group == PointGroup.REFERENCE_ORIGIN
					else max_missing
				)
				can_track_only = (
					flow_ok[i]
					and flow_error[i] <= max_flow_error
					and point.missing_frame_count < point_max_missing
				)
				if can_track_only:
					point.current_position = point.predicted_position
					point.detected_position = None
					point.status = PointStatus.TRACKED_ONLY
					point.missing_frame_count += 1
					point.tracking_quality = float(
						max(0.0, 1.0 - flow_error[i] / max_flow_error) * 0.5
					)
				else:
					point.detected_position = None
					point.missing_frame_count += 1
					point.status = (
						PointStatus.MISSING
						if point.missing_frame_count <= point_max_missing
						else PointStatus.INVALID
					)
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
		self._refresh_validity()
		return self.points


def draw_tracking(frame: np.ndarray, points: list[TrackedPoint]) -> np.ndarray:
	role_colors = {
		PointGroup.REFERENCE_ORIGIN: (255, 0, 255),
		PointGroup.DEFORMING: (0, 255, 0),
		PointGroup.RIGID_REFERENCE: (255, 255, 0),
	}
	status_colors = {
		PointStatus.TRACKED_ONLY: (255, 180, 0),
		PointStatus.RECOVERED: (0, 255, 255),
		PointStatus.MISSING: (0, 0, 255),
		PointStatus.INVALID: (80, 80, 80),
	}
	for point in points:
		x, y = (int(round(v)) for v in point.current_position)
		role_color = role_colors[point.point_group]
		color = (
			status_colors[point.status]
			if point.status in (PointStatus.MISSING, PointStatus.INVALID)
			else role_color
		)
		label = "ORIGIN" if point.point_group == PointGroup.REFERENCE_ORIGIN else point.name
		radius = 8 if point.point_group == PointGroup.REFERENCE_ORIGIN else 5
		cv2.circle(frame, (x, y), radius, color, 2)
		if point.status in (PointStatus.TRACKED_ONLY, PointStatus.RECOVERED):
			cv2.circle(frame, (x, y), 2, status_colors[point.status], -1)
		cv2.putText(frame, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
					0.35, color, 1, cv2.LINE_AA)
	return frame
