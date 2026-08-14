"""Image processing helpers for red/blue blob detection and line fitting."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
import yaml
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HsvPreset:
	red_lower_1: np.ndarray
	red_upper_1: np.ndarray
	red_lower_2: np.ndarray
	red_upper_2: np.ndarray

	def __post_init__(self):
		self.red_lower_1 = np.array(self.red_lower_1)
		self.red_upper_1 = np.array(self.red_upper_1)
		self.red_lower_2 = np.array(self.red_lower_2)
		self.red_upper_2 = np.array(self.red_upper_2)


@dataclass
class CalibrationConfig:
	min_area: int
	max_area: Optional[int]
	area_low_ratio: float
	area_up_ratio: float
	hue_margin: int
	sat_margin: int
	val_margin: int
	preset: str
	red_lower_1: np.ndarray
	red_upper_1: np.ndarray
	red_lower_2: np.ndarray
	red_upper_2: np.ndarray

	def __post_init__(self):
		self.red_lower_1 = np.array(self.red_lower_1)
		self.red_upper_1 = np.array(self.red_upper_1)
		self.red_lower_2 = np.array(self.red_lower_2)
		self.red_upper_2 = np.array(self.red_upper_2)


@dataclass
class BlueCalibrationConfig:
	min_area: int
	max_area: Optional[int]
	area_low_ratio: float
	area_up_ratio: float
	hue_margin: int
	sat_margin: int
	val_margin: int
	blue_lower_1: np.ndarray
	blue_upper_1: np.ndarray
	blue_lower_2: np.ndarray
	blue_upper_2: np.ndarray

	def __post_init__(self):
		self.blue_lower_1 = np.array(self.blue_lower_1)
		self.blue_upper_1 = np.array(self.blue_upper_1)
		self.blue_lower_2 = np.array(self.blue_lower_2)
		self.blue_upper_2 = np.array(self.blue_upper_2)


@dataclass
class CircleDetectionConfig:
	dp: int
	min_dist: int
	param1: int
	param2: int
	min_radius: int
	max_radius: int
	require_validation_for_reference: bool = False
	require_validation_for_tracking: bool = False

@dataclass
class ForcePointConfig:
	node_number: int
	force_magnitude: float

@dataclass
class AppConfig:
	calibration: CalibrationConfig
	blue_calibration: BlueCalibrationConfig
	circles: CircleDetectionConfig
	presets: dict
	force_node: ForcePointConfig
	camera: dict = field(default_factory=dict)
	reference_ransac: dict = field(default_factory=dict)
	tracking: dict = field(default_factory=dict)
	blue_tracking: dict = field(default_factory=dict)
	reference: dict = field(default_factory=dict)
	geometry: dict = field(default_factory=dict)
	dataset: dict = field(default_factory=dict)
	force_model: dict = field(default_factory=dict)
	display: dict = field(default_factory=dict)
	training: dict = field(default_factory=dict)
	def active_preset(self) -> HsvPreset:
		"""Return the currently selected HsvPreset."""
		return self.presets[self.calibration.preset]


DEFAULT_BLUE_CALIBRATION: dict[str, Any] = {
	"min_area": 50,
	"max_area": 5_000,
	"area_low_ratio": 0.6,
	"area_up_ratio": 0.3,
	"hue_margin": 10,
	"sat_margin": 60,
	"val_margin": 60,
	"blue_lower_1": [90, 40, 30],
	"blue_upper_1": [140, 255, 255],
	"blue_lower_2": [0, 0, 0],
	"blue_upper_2": [0, 0, 0],
}

DEFAULT_BLUE_TRACKING: dict[str, Any] = {
	"maximum_assignment_distance_px": 40.0,
	"maximum_missing_frames": 5,
	"maximum_optical_flow_error": 25.0,
	"maximum_pyramid_level": 3,
	"minimum_tracking_quality": 0.2,
	"allow_tracked_only": True,
	"termination_count": 20,
	"termination_epsilon": 0.03,
	"window_size": [21, 21],
}


# ---------------------------------------------------------------------------
# Config loading / saving
# ---------------------------------------------------------------------------

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera-config.yaml")
_default_config_path = os.path.join(
	os.path.dirname(os.path.abspath(__file__)), "camera-config.defaults.yaml"
)


def config_from_mapping(raw: dict[str, Any]) -> AppConfig:
	"""Build and validate an application configuration from a mapping."""
	if not isinstance(raw, dict):
		raise ValueError("Configuration root must be a YAML mapping")

	# Handle missing key seamlessly for backwards-compatibility
	force_raw = raw.get("force_node", {"node_number": 1, "force_magnitude": 0.0})
	blue_calibration_raw = {
		**DEFAULT_BLUE_CALIBRATION,
		**raw.get("blue_calibration", {}),
	}

	result = AppConfig(
		calibration=CalibrationConfig(**raw["calibration"]),
		blue_calibration=BlueCalibrationConfig(**blue_calibration_raw),
		circles=CircleDetectionConfig(**raw["hough_circles"]),
		presets={name: HsvPreset(**values) for name, values in raw["presets"].items()},
		force_node=ForcePointConfig(**force_raw),
		camera=raw.get("camera", {}),
		reference_ransac=raw.get("reference_ransac", {}),
		tracking=raw.get("tracking", {}),
		reference=raw.get("reference", {}),
		geometry=raw.get("geometry", {}),
		dataset=raw.get("dataset", {}),
		force_model=raw.get("force_model", {}),
		display=raw.get("display", {}),
		training=raw.get("training", {}),
		blue_tracking={**DEFAULT_BLUE_TRACKING, **raw.get("blue_tracking", {})},
	)
	validate_config(result)
	return result


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
	config_path = os.fspath(path or _config_path)
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with open(config_path, "r", encoding="utf-8") as f:
		raw = yaml.safe_load(f)
	return config_from_mapping(raw)


def config_to_mapping(config: AppConfig) -> dict[str, Any]:
	"""Return a detached, YAML-safe representation of an ``AppConfig``."""
	cal = config.calibration
	blue = config.blue_calibration
	data: dict[str, Any] = {
		"calibration": {
			"min_area":       cal.min_area,
			"max_area":       cal.max_area,
			"area_low_ratio": cal.area_low_ratio,
			"area_up_ratio":  cal.area_up_ratio,
			"hue_margin":     cal.hue_margin,
			"sat_margin":     cal.sat_margin,
			"val_margin":     cal.val_margin,
			"preset":         cal.preset,
			"red_lower_1":    cal.red_lower_1.tolist(),
			"red_upper_1":    cal.red_upper_1.tolist(),
			"red_lower_2":    cal.red_lower_2.tolist(),
			"red_upper_2":    cal.red_upper_2.tolist(),
		},
		"blue_calibration": {
			"min_area":       blue.min_area,
			"max_area":       blue.max_area,
			"area_low_ratio": blue.area_low_ratio,
			"area_up_ratio":  blue.area_up_ratio,
			"hue_margin":     blue.hue_margin,
			"sat_margin":     blue.sat_margin,
			"val_margin":     blue.val_margin,
			"blue_lower_1":   blue.blue_lower_1.tolist(),
			"blue_upper_1":   blue.blue_upper_1.tolist(),
			"blue_lower_2":   blue.blue_lower_2.tolist(),
			"blue_upper_2":   blue.blue_upper_2.tolist(),
		},
		"hough_circles": asdict(config.circles),
		"presets": {
			name: {
				"red_lower_1": preset.red_lower_1.tolist(),
				"red_upper_1": preset.red_upper_1.tolist(),
				"red_lower_2": preset.red_lower_2.tolist(),
				"red_upper_2": preset.red_upper_2.tolist(),
			}
			for name, preset in config.presets.items()
		},
		"force_node": asdict(config.force_node),
	}
	for section in ("camera", "reference_ransac", "tracking", "blue_tracking", "reference",
					"geometry", "dataset", "force_model", "display", "training"):
		data[section] = copy.deepcopy(getattr(config, section))
	return data


def validate_config(config: AppConfig) -> None:
	"""Validate cross-field constraints used by the live vision pipeline."""
	cal = config.calibration
	if int(cal.min_area) < 0:
		raise ValueError("calibration.min_area must be non-negative")
	if cal.max_area is not None and int(cal.max_area) < int(cal.min_area):
		raise ValueError("calibration.max_area must be at least min_area")
	for name in ("area_low_ratio", "area_up_ratio"):
		if float(getattr(cal, name)) < 0.0:
			raise ValueError(f"calibration.{name} must be non-negative")
	for name, maximum in (("hue_margin", 180), ("sat_margin", 255), ("val_margin", 255)):
		value = int(getattr(cal, name))
		if value < 0 or value > maximum:
			raise ValueError(f"calibration.{name} must be in 0..{maximum}")
	if cal.preset not in config.presets:
		raise ValueError(f"Unknown calibration preset: {cal.preset}")
	for name in ("red_lower_1", "red_upper_1", "red_lower_2", "red_upper_2"):
		values = np.asarray(getattr(cal, name))
		if values.shape != (3,):
			raise ValueError(f"calibration.{name} must contain three values")
		if not (0 <= int(values[0]) <= 180 and np.all((values[1:] >= 0) & (values[1:] <= 255))):
			raise ValueError(f"calibration.{name} contains an invalid HSV value")

	blue = config.blue_calibration
	if int(blue.min_area) < 0:
		raise ValueError("blue_calibration.min_area must be non-negative")
	if blue.max_area is not None and int(blue.max_area) < int(blue.min_area):
		raise ValueError("blue_calibration.max_area must be at least min_area")
	for name in ("area_low_ratio", "area_up_ratio"):
		if float(getattr(blue, name)) < 0.0:
			raise ValueError(f"blue_calibration.{name} must be non-negative")
	for name, maximum in (("hue_margin", 180), ("sat_margin", 255), ("val_margin", 255)):
		value = int(getattr(blue, name))
		if value < 0 or value > maximum:
			raise ValueError(f"blue_calibration.{name} must be in 0..{maximum}")
	for name in ("blue_lower_1", "blue_upper_1", "blue_lower_2", "blue_upper_2"):
		values = np.asarray(getattr(blue, name))
		if values.shape != (3,):
			raise ValueError(f"blue_calibration.{name} must contain three values")
		if not (0 <= int(values[0]) <= 180 and np.all((values[1:] >= 0) & (values[1:] <= 255))):
			raise ValueError(f"blue_calibration.{name} contains an invalid HSV value")

	blue_tracking = config.blue_tracking
	for name in (
		"maximum_assignment_distance_px", "maximum_optical_flow_error",
		"termination_epsilon",
	):
		if float(blue_tracking.get(name, 0.0)) <= 0.0:
			raise ValueError(f"blue_tracking.{name} must be positive")
	for name in ("maximum_missing_frames", "maximum_pyramid_level"):
		if int(blue_tracking.get(name, -1)) < 0:
			raise ValueError(f"blue_tracking.{name} must be non-negative")
	if int(blue_tracking.get("termination_count", 0)) < 1:
		raise ValueError("blue_tracking.termination_count must be positive")
	minimum_blue_quality = float(blue_tracking.get("minimum_tracking_quality", 0.0))
	if minimum_blue_quality < 0.0 or minimum_blue_quality > 1.0:
		raise ValueError("blue_tracking.minimum_tracking_quality must be in 0..1")
	window_size = blue_tracking.get("window_size", [])
	if (
		not isinstance(window_size, (list, tuple))
		or len(window_size) != 2
		or any(int(value) < 1 for value in window_size)
	):
		raise ValueError("blue_tracking.window_size must contain two positive values")

	circles = config.circles
	for name in ("dp", "min_dist", "param1", "param2"):
		if int(getattr(circles, name)) < 1:
			raise ValueError(f"hough_circles.{name} must be positive")
	if circles.min_radius < 0 or circles.max_radius < circles.min_radius:
		raise ValueError("Hough circle radius limits are invalid")

	for name in ("width", "height", "fps"):
		if int(config.camera.get(name, 0)) < 1:
			raise ValueError(f"camera.{name} must be positive")

	ransac = config.reference_ransac
	for name in ("expected_points_line_a", "expected_points_line_b", "iterations",
				 "minimum_inliers_per_line"):
		if int(ransac.get(name, 0)) < 1:
			raise ValueError(f"reference_ransac.{name} must be positive")
	expected_total = int(ransac["expected_points_line_a"]) + int(
		ransac["expected_points_line_b"]
	)
	if int(ransac["minimum_inliers_per_line"]) > min(
		int(ransac["expected_points_line_a"]),
		int(ransac["expected_points_line_b"]),
	):
		raise ValueError("minimum_inliers_per_line exceeds an expected line count")
	configured_total = int(config.tracking.get("expected_deforming_point_count", expected_total))
	if configured_total != expected_total:
		raise ValueError(
			"tracking.expected_deforming_point_count must equal the two RANSAC line counts"
		)
	if int(config.tracking.get("expected_reference_origin_point_count", 1)) != 1:
		raise ValueError("Exactly one reference-origin point must be configured")
	for section, names in (
		(config.tracking, ("minimum_tracked_only_quality",)),
		(config.reference, ("minimum_origin_tracking_quality",)),
		(config.geometry, (
			"minimum_deforming_tracking_quality",
			"minimum_rigid_reference_tracking_quality",
		)),
	):
		for name in names:
			value = float(section.get(name, 0.0))
			if value < 0.0 or value > 1.0:
				raise ValueError(f"{name} must be in 0..1")
	mode = str(config.geometry.get("geometry_reference_mode", "translation_only"))
	if mode not in {"translation_only", "translation_and_rotation"}:
		raise ValueError("geometry.geometry_reference_mode is invalid")
	if int(config.dataset.get("frames_per_sample", 1)) < 1:
		raise ValueError("dataset.frames_per_sample must be positive")
	if int(config.dataset.get("sample_interval_ms", 0)) < 0:
		raise ValueError("dataset.sample_interval_ms must be non-negative")
	if int(config.dataset.get("maximum_invalid_frame_retries", 1)) < 1:
		raise ValueError("dataset.maximum_invalid_frame_retries must be positive")
	for name in (
		"save_images",
		"save_raw_coordinates",
		"save_compensated_coordinates",
		"save_origin_relative_coordinates",
		"save_feature_names",
	):
		if name in config.dataset and not isinstance(config.dataset[name], bool):
			raise ValueError(f"dataset.{name} must be true or false")
	if not bool(config.dataset.get("save_feature_names", True)):
		raise ValueError(
			"dataset.save_feature_names must remain true because it is part of the "
			"training compatibility contract"
		)

	connections = config.geometry.get("structural_connections", [])
	if not isinstance(connections, list):
		raise ValueError("geometry.structural_connections must be a list")
	seen_connections: set[tuple[str, str]] = set()
	for connection in connections:
		if (
			not isinstance(connection, (list, tuple))
			or len(connection) != 2
			or not all(isinstance(value, str) and value for value in connection)
		):
			raise ValueError(
				"Each structural connection must contain two non-empty point IDs"
			)
		pair = tuple(connection)
		if pair[0] == pair[1] or pair in seen_connections or pair[::-1] in seen_connections:
			raise ValueError("Structural connections must be unique and join two points")
		seen_connections.add(pair)

	force_model = config.force_model
	for name in ("model_path", "metadata_path", "model_type"):
		if force_model.get(name) is not None and not isinstance(force_model[name], str):
			raise ValueError(f"force_model.{name} must be text or null")
	minimum_force = float(force_model.get("minimum_calibrated_force_N", 5.0))
	maximum_force = float(force_model.get("maximum_calibrated_force_N", 40.0))
	if (
		not np.isfinite(minimum_force)
		or not np.isfinite(maximum_force)
		or minimum_force > maximum_force
	):
		raise ValueError("force_model calibrated range is invalid")


def configuration_compatibility_notes(config: AppConfig) -> list[str]:
	"""Explain retained legacy keys that intentionally do not drive runtime data."""
	notes: list[str] = []
	if "minimum_valid_points" in config.tracking:
		notes.append(
			"tracking.minimum_valid_points is compatibility-only: permanent required "
			"point IDs and geometry validity determine recordability."
		)
	if any(
		name in config.force_model
		for name in (
			"minimum_calibrated_force_N", "maximum_calibrated_force_N", "model_type"
		)
	):
		notes.append(
			"force_model calibrated bounds and model_type are compatibility-only; "
			"validated model metadata is authoritative."
		)
	return notes


def update_config_in_place(target: AppConfig, source: AppConfig) -> None:
	"""Copy configuration values while preserving the shared object identity."""
	validate_config(source)
	target.calibration = copy.deepcopy(source.calibration)
	target.blue_calibration = copy.deepcopy(source.blue_calibration)
	target.circles = copy.deepcopy(source.circles)
	target.presets = copy.deepcopy(source.presets)
	target.force_node = copy.deepcopy(source.force_node)
	for section in ("camera", "reference_ransac", "tracking", "blue_tracking", "reference",
					"geometry", "dataset", "force_model", "display", "training"):
		setattr(target, section, copy.deepcopy(getattr(source, section)))


def load_default_config() -> AppConfig:
	"""Load the packaged factory configuration used by the GUI reset action."""
	return load_config(_default_config_path)


def save_config(
	config: AppConfig,
	path: str | os.PathLike[str] | None = None,
) -> None:
	"""Atomically save a validated configuration to YAML."""
	validate_config(config)
	target = Path(path or _config_path)
	target.parent.mkdir(parents=True, exist_ok=True)
	data = config_to_mapping(config)
	temporary_path: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			"w", encoding="utf-8", dir=target.parent,
			prefix=f".{target.name}.", suffix=".tmp", delete=False,
		) as stream:
			temporary_path = Path(stream.name)
			yaml.safe_dump(data, stream, default_flow_style=False, sort_keys=False)
			stream.flush()
			os.fsync(stream.fileno())
		# Reject a malformed serialization before replacing the user's file.
		load_config(temporary_path)
		os.replace(temporary_path, target)
	except Exception:
		if temporary_path is not None:
			try:
				temporary_path.unlink(missing_ok=True)
			except OSError:
				pass
		raise

# ---------------------------------------------------------------------------
# Module-level config instance (loaded once at import time)
# ---------------------------------------------------------------------------

cfg = load_config()


def _resolve_threshold(value, fallback):
	return fallback if value is None else value


def _build_hue_range(hue, hue_margin):
	lower = hue - hue_margin
	upper = hue + hue_margin
	if lower < 0:
		return (0, upper), (180 + lower, 180)
	if upper > 180:
		return (lower, 180), (0, upper - 180)
	return (lower, upper), (lower, upper)


def apply_calibration_preset(config: AppConfig | None = None) -> dict:
	"""Copy the active preset's HSV ranges into calibration and return them."""
	target = cfg if config is None else config
	preset = target.active_preset()
	target.calibration.red_lower_1 = preset.red_lower_1.copy()
	target.calibration.red_upper_1 = preset.red_upper_1.copy()
	target.calibration.red_lower_2 = preset.red_lower_2.copy()
	target.calibration.red_upper_2 = preset.red_upper_2.copy()
	return {
		"lower_red1": target.calibration.red_lower_1.copy(),
		"upper_red1": target.calibration.red_upper_1.copy(),
		"lower_red2": target.calibration.red_lower_2.copy(),
		"upper_red2": target.calibration.red_upper_2.copy(),
	}


def _format_hsv_range(lower, upper):
	return f"[{int(lower[0])}, {int(lower[1])}, {int(lower[2])}] .. [{int(upper[0])}, {int(upper[1])}, {int(upper[2])}]"


def sample_hsv_at_point(frame, point):
	"""Return the HSV value of a single clicked pixel in a BGR frame."""
	if frame is None:
		return None

	x, y = point
	height, width = frame.shape[:2]
	if x < 0 or y < 0 or x >= width or y >= height:
		return None

	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	return tuple(int(value) for value in hsv_frame[y, x])


def sample_blob_hsv_near_point(frame, blob, point):
	"""Sample the nearest currently-red pixel inside a blob's outer contour.

	This handles hollow markers: their centre is inside the outer contour but is
	usually background, so sampling the clicked pixel itself gives a bad colour.
	"""
	if frame is None or blob is None:
		return None

	height, width = frame.shape[:2]
	x, y = point
	if x < 0 or y < 0 or x >= width or y >= height:
		return None

	red_mask = create_red_mask(frame)
	contour_mask = np.zeros((height, width), dtype=np.uint8)
	cv2.drawContours(contour_mask, [blob["contour"]], -1, 255, thickness=cv2.FILLED)
	candidate_mask = cv2.bitwise_and(red_mask, contour_mask)
	candidate_y, candidate_x = np.nonzero(candidate_mask)
	if candidate_x.size == 0:
		return None

	distances = (candidate_x.astype(np.int64) - x) ** 2 + (candidate_y.astype(np.int64) - y) ** 2
	nearest = int(np.argmin(distances))
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	return tuple(int(value) for value in hsv_frame[candidate_y[nearest], candidate_x[nearest]])


def create_red_mask(
	frame,
	lower_red1=None,
	upper_red1=None,
	lower_red2=None,
	upper_red2=None,
):
	"""Create a binary mask for red pixels in a BGR frame."""
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_red1 = _resolve_threshold(lower_red1, cfg.calibration.red_lower_1)
	upper_red1 = _resolve_threshold(upper_red1, cfg.calibration.red_upper_1)
	lower_red2 = _resolve_threshold(lower_red2, cfg.calibration.red_lower_2)
	upper_red2 = _resolve_threshold(upper_red2, cfg.calibration.red_upper_2)
	masks = [cv2.inRange(hsv, lower_red1, upper_red1)]
	if not (np.all(lower_red2 == 0) and np.all(upper_red2 == 0)):
		masks.append(cv2.inRange(hsv, lower_red2, upper_red2))

	mask = masks[0]
	for extra_mask in masks[1:]:
		mask = cv2.bitwise_or(mask, extra_mask)
	return mask


def create_blue_mask(
	frame,
	lower_blue1=None,
	upper_blue1=None,
	lower_blue2=None,
	upper_blue2=None,
	config: AppConfig | None = None,
):
	"""Create a binary mask for blue pixels in a BGR frame."""
	target = cfg if config is None else config
	blue = target.blue_calibration
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_blue1 = np.asarray(_resolve_threshold(lower_blue1, blue.blue_lower_1))
	upper_blue1 = np.asarray(_resolve_threshold(upper_blue1, blue.blue_upper_1))
	lower_blue2 = np.asarray(_resolve_threshold(lower_blue2, blue.blue_lower_2))
	upper_blue2 = np.asarray(_resolve_threshold(upper_blue2, blue.blue_upper_2))
	masks = [cv2.inRange(hsv, lower_blue1, upper_blue1)]
	if not (np.all(lower_blue2 == 0) and np.all(upper_blue2 == 0)):
		masks.append(cv2.inRange(hsv, lower_blue2, upper_blue2))
	mask = masks[0]
	for extra_mask in masks[1:]:
		mask = cv2.bitwise_or(mask, extra_mask)
	return mask


def sample_blue_blob_hsv_near_point(
	frame,
	blob,
	point,
	config: AppConfig | None = None,
):
	"""Sample the nearest masked blue pixel inside the selected contour."""
	if frame is None or blob is None:
		return None
	height, width = frame.shape[:2]
	x, y = point
	if x < 0 or y < 0 or x >= width or y >= height:
		return None
	blue_mask = create_blue_mask(frame, config=config)
	contour_mask = np.zeros((height, width), dtype=np.uint8)
	cv2.drawContours(contour_mask, [blob["contour"]], -1, 255, thickness=cv2.FILLED)
	candidate_mask = cv2.bitwise_and(blue_mask, contour_mask)
	candidate_y, candidate_x = np.nonzero(candidate_mask)
	if candidate_x.size == 0:
		return None
	distances = (
		(candidate_x.astype(np.int64) - x) ** 2
		+ (candidate_y.astype(np.int64) - y) ** 2
	)
	nearest = int(np.argmin(distances))
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	return tuple(
		int(value)
		for value in hsv_frame[candidate_y[nearest], candidate_x[nearest]]
	)


def calibrate_from_blob(
	blob,
	area_low_ratio=None,
	area_up_ratio=None,
	hue_margin=None,
	sat_margin=None,
	val_margin=None,
	config: AppConfig | None = None,
):
	"""Update the live calibration thresholds from a selected blob."""
	target = cfg if config is None else config
	area_low_ratio = target.calibration.area_low_ratio if area_low_ratio is None else area_low_ratio
	area_up_ratio  = target.calibration.area_up_ratio if area_up_ratio is None else area_up_ratio
	hue_margin     = target.calibration.hue_margin if hue_margin is None else hue_margin
	sat_margin     = target.calibration.sat_margin if sat_margin is None else sat_margin
	val_margin     = target.calibration.val_margin if val_margin is None else val_margin

	area = float(blob["area"])
	target.calibration.min_area = max(1, int(area * (1.0 - area_low_ratio)))
	target.calibration.max_area = max(target.calibration.min_area, int(area * (1.0 + area_up_ratio)))

	sample_hsv = blob.get("sample_hsv")

	hue, sat, val = (int(value) for value in sample_hsv)
	(s1, s2), secondary_hue_range = _build_hue_range(hue, hue_margin)
	cal_s_low = max(0, sat - sat_margin)
	cal_s_high = min(255, sat + sat_margin)
	cal_v_low = max(0, val - val_margin)
	cal_v_high = min(255, val + val_margin)

	target.calibration.red_lower_1 = np.array([int(s1), cal_s_low, cal_v_low])
	target.calibration.red_upper_1 = np.array([int(s2), cal_s_high, cal_v_high])

	low_h, high_h = secondary_hue_range
	target.calibration.red_lower_2 = np.array([int(low_h), cal_s_low, cal_v_low])
	target.calibration.red_upper_2 = np.array([int(high_h), cal_s_high, cal_v_high])

	return {
		"area_min":   target.calibration.min_area,
		"area_max":   target.calibration.max_area,
		"sample_hsv": tuple(int(value) for value in sample_hsv),
		"lower_red1": target.calibration.red_lower_1.copy(),
		"upper_red1": target.calibration.red_upper_1.copy(),
		"lower_red2": target.calibration.red_lower_2.copy(),
		"upper_red2": target.calibration.red_upper_2.copy(),
	}


def calibrate_blue_from_blob(
	blob,
	area_low_ratio=None,
	area_up_ratio=None,
	hue_margin=None,
	sat_margin=None,
	val_margin=None,
	config: AppConfig | None = None,
):
	"""Update independent blue HSV and area thresholds from a selected blob."""
	target = cfg if config is None else config
	blue = target.blue_calibration
	area_low_ratio = blue.area_low_ratio if area_low_ratio is None else area_low_ratio
	area_up_ratio = blue.area_up_ratio if area_up_ratio is None else area_up_ratio
	hue_margin = blue.hue_margin if hue_margin is None else hue_margin
	sat_margin = blue.sat_margin if sat_margin is None else sat_margin
	val_margin = blue.val_margin if val_margin is None else val_margin

	area = float(blob["area"])
	blue.min_area = max(1, int(area * (1.0 - area_low_ratio)))
	blue.max_area = max(blue.min_area, int(area * (1.0 + area_up_ratio)))
	sample_hsv = blob.get("sample_hsv")
	if sample_hsv is None:
		raise ValueError("The selected blue blob has no HSV sample")
	hue, sat, val = (int(value) for value in sample_hsv)
	(primary_low, primary_high), secondary = _build_hue_range(hue, hue_margin)
	saturation_low = max(0, sat - sat_margin)
	saturation_high = min(255, sat + sat_margin)
	value_low = max(0, val - val_margin)
	value_high = min(255, val + val_margin)
	blue.blue_lower_1 = np.array(
		[int(primary_low), saturation_low, value_low]
	)
	blue.blue_upper_1 = np.array(
		[int(primary_high), saturation_high, value_high]
	)
	secondary_low, secondary_high = secondary
	blue.blue_lower_2 = np.array(
		[int(secondary_low), saturation_low, value_low]
	)
	blue.blue_upper_2 = np.array(
		[int(secondary_high), saturation_high, value_high]
	)
	return {
		"area_min": blue.min_area,
		"area_max": blue.max_area,
		"sample_hsv": tuple(int(value) for value in sample_hsv),
		"lower_blue1": blue.blue_lower_1.copy(),
		"upper_blue1": blue.blue_upper_1.copy(),
		"lower_blue2": blue.blue_lower_2.copy(),
		"upper_blue2": blue.blue_upper_2.copy(),
	}


def find_red_blob_data(frame, min_area: int = None, max_area: int = None):
	"""Return metadata for each detected red blob."""
	min_area = _resolve_threshold(min_area, cfg.calibration.min_area)
	max_area = _resolve_threshold(max_area, cfg.calibration.max_area)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = create_red_mask(frame)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	blobs = []
	for contour in contours:
		area = float(cv2.contourArea(contour))
		if area < min_area:
			continue
		if max_area is not None and area > max_area:
			continue

		moments = cv2.moments(contour)
		if moments["m00"] == 0:
			continue

		cx = float(moments["m10"] / moments["m00"])
		cy = float(moments["m01"] / moments["m00"])
		x, y, w, h = cv2.boundingRect(contour)
		contour_mask = np.zeros(mask.shape, dtype=np.uint8)
		cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
		mean_hsv = cv2.mean(hsv, mask=contour_mask)
		center_hsv = tuple(int(value) for value in hsv[int(round(cy)), int(round(cx))])
		perimeter = float(cv2.arcLength(contour, True))
		circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
		detection_quality = float(np.clip(circularity, 0.0, 1.0))

		blobs.append(
			{
				"center": (cx, cy),
				"area": area,
				"bbox": (x, y, w, h),
				"bounding_box": (x, y, w, h),
				"contour": contour,
				"center_hsv": center_hsv,
				"mean_hsv": tuple(float(value) for value in mean_hsv[:3]),
				"detection_quality": detection_quality,
				"circle_validation": None,
			}
		)

	return blobs


def find_blue_blob_data(
	frame,
	min_area: int | None = None,
	max_area: int | None = None,
	config: AppConfig | None = None,
):
	"""Return metadata for blue candidates; target selection happens separately."""
	target = cfg if config is None else config
	blue = target.blue_calibration
	min_area = _resolve_threshold(min_area, blue.min_area)
	max_area = _resolve_threshold(max_area, blue.max_area)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = create_blue_mask(frame, config=target)
	contours, _ = cv2.findContours(
		mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)
	blobs = []
	for contour in contours:
		area = float(cv2.contourArea(contour))
		if area < min_area or (max_area is not None and area > max_area):
			continue
		moments = cv2.moments(contour)
		if moments["m00"] == 0:
			continue
		cx = float(moments["m10"] / moments["m00"])
		cy = float(moments["m01"] / moments["m00"])
		x, y, width, height = cv2.boundingRect(contour)
		contour_mask = np.zeros(mask.shape, dtype=np.uint8)
		cv2.drawContours(
			contour_mask, [contour], -1, 255, thickness=cv2.FILLED
		)
		mean_hsv = cv2.mean(hsv, mask=contour_mask)
		center_hsv = tuple(
			int(value) for value in hsv[int(round(cy)), int(round(cx))]
		)
		perimeter = float(cv2.arcLength(contour, True))
		circularity = (
			0.0 if perimeter <= 0
			else float(4.0 * np.pi * area / (perimeter * perimeter))
		)
		blobs.append({
			"center": (cx, cy),
			"area": area,
			"bbox": (x, y, width, height),
			"bounding_box": (x, y, width, height),
			"contour": contour,
			"center_hsv": center_hsv,
			"mean_hsv": tuple(float(value) for value in mean_hsv[:3]),
			"detection_quality": float(np.clip(circularity, 0.0, 1.0)),
		})
	return blobs

#make switches
def detect_circles(frame, dp=None, min_dist=None, param1=None,
                   param2=None, min_radius=None, max_radius=None):
    """Detect circles in the frame using Hough Circle Transform."""
    dp         = dp         or cfg.circles.dp
    min_dist   = min_dist   or cfg.circles.min_dist
    param1     = param1     or cfg.circles.param1
    param2     = param2     or cfg.circles.param2
    min_radius = min_radius or cfg.circles.min_radius
    max_radius = max_radius or cfg.circles.max_radius
    """Detect circles in the frame using Hough Circle Transform."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, 
                                param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return [(circle[0], circle[1], circle[2]) for circle in circles[0]]
    return []


def draw_circles(frame, circles):
	"""Draw detected circles on the frame."""
	for (x, y, r) in circles:
		cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
		cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
	return frame

def find_red_blob_centers(frame, min_area: int = None, max_area: int = None):
	"""Find red blob centers in a BGR frame."""
	return [blob["center"] for blob in find_red_blob_data(frame, min_area=min_area, max_area=max_area)]


def get_blob_at_point(blobs, point):
	"""Return the blob whose contour contains the point, if any."""
	x, y = point
	for blob in blobs:
		contour = blob["contour"]
		if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
			return blob
	return None


def find_blob_index_for_click(
	blobs: list[dict[str, Any]],
	point: tuple[int, int],
	maximum_distance_px: float,
) -> int | None:
	"""Resolve a click contour-first, then by a tightly bounded centre distance."""
	x, y = point
	for index, blob in enumerate(blobs):
		contour = blob.get("contour")
		if contour is not None and cv2.pointPolygonTest(
			contour, (float(x), float(y)), False
		) >= 0:
			return index
	if not blobs:
		return None
	distances = [
		float(np.linalg.norm(np.subtract(blob["center"], (float(x), float(y)))))
		for blob in blobs
	]
	nearest = int(np.argmin(distances))
	return nearest if distances[nearest] <= float(maximum_distance_px) else None


def overlay_red_mask_on_frame(frame, alpha: float = 0.5):
	"""Overlay the red binary mask onto the frame as a red highlight."""
	mask = create_red_mask(frame)
	mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	red_layer = np.zeros_like(frame)
	red_layer[:, :, 2] = mask
	return cv2.addWeighted(frame, 1.0, red_layer, alpha, 0)


def show_red_mask_only(frame):
	"""Create a red-on-black live view that shows only the detected mask."""
	mask = create_red_mask(frame)
	mask_view = np.zeros_like(frame)
	mask_view[:, :, 2] = mask
	return mask_view


def draw_blob_calibration_info(frame, blob):
	"""Annotate a selected blob with the clicked-pixel HSV and active calibration ranges."""
	if blob is None:
		return frame

	cx_f, cy_f = blob["center"]
	cx, cy = int(round(cx_f)), int(round(cy_f))
	area = blob["area"]
	sample_hsv = blob.get("sample_hsv", blob.get("click_hsv"))
	if sample_hsv is None:
		sample_hsv = blob.get("center_hsv")

	box_x1 = max(10, cx + 8)
	box_y1 = max(10, cy + 8)
	box_x2 = box_x1 + 380
	box_y2 = box_y1 + 160

	cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), thickness=-1)
	cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), thickness=1)
	cv2.putText(frame, f"Area: {area:.1f} px^2", (box_x1 + 10, box_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Area range: {cfg.calibration.min_area} .. {cfg.calibration.max_area}", (box_x1 + 10, box_y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	if sample_hsv is not None:
		cv2.putText(frame, f"Sampled HSV: {tuple(int(value) for value in sample_hsv)}", (box_x1 + 10, box_y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Lower red 1: {_format_hsv_range(cfg.calibration.red_lower_1, cfg.calibration.red_upper_1)}", (box_x1 + 10, box_y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Lower red 2: {_format_hsv_range(cfg.calibration.red_lower_2, cfg.calibration.red_upper_2)}", (box_x1 + 10, box_y1 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Press x to clear selection", (box_x1 + 10, box_y1 + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	return frame


def draw_centers_with_positions(frame, centers):
	"""Draw blob centers and annotate them with pixel coordinates."""
	for idx, (cx, cy) in enumerate(centers, start=1):
		draw_center = (int(round(cx)), int(round(cy)))
		cv2.circle(frame, draw_center, 6, (0, 0, 255), -1)
		cv2.putText(
			frame,
			f"{idx}: ({cx}, {cy})",
			(draw_center[0] + 8, draw_center[1] - 8),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(0, 0, 255),
			1,
			cv2.LINE_AA,
		)
	return frame


def _perp_distance(points, anchor, direction):
	diff = points - anchor
	cross = diff[:, 0] * direction[1] - diff[:, 1] * direction[0]
	return np.abs(cross)


def _fit_line(points):
	anchor = points.mean(axis=0)
	_, _, vt = np.linalg.svd(points - anchor)
	direction = vt[0]
	direction = direction / np.linalg.norm(direction)
	return anchor, direction


def _ransac_line(points, epsilon=15, iterations=200, min_inliers=2):
	count = len(points)
	if count < 2:
		return None, None, None

	rng = np.random.default_rng()
	best_mask = np.zeros(count, dtype=bool)
	best_inlier_count = 0

	for _ in range(iterations):
		idx = rng.choice(count, size=2, replace=False)
		p1, p2 = points[idx[0]], points[idx[1]]

		direction = p2 - p1
		norm = np.linalg.norm(direction)
		if norm < 1e-9:
			continue
		direction = direction / norm

		distances = _perp_distance(points, p1, direction)
		mask = distances < epsilon
		inlier_count = int(mask.sum())

		if inlier_count > best_inlier_count:
			best_inlier_count = inlier_count
			best_mask = mask

	if best_inlier_count < min_inliers:
		return None, None, None

	inlier_points = points[best_mask]
	anchor, direction = _fit_line(inlier_points)
	return anchor, direction, best_mask


def fit_two_lines_ransac(centers, epsilon=15, iterations=200):
	"""Fit up to two lines to blob centers using RANSAC."""
	if len(centers) < 4:
		return []

	points = np.asarray(centers, dtype=float)
	remaining = points.copy()
	lines = []

	for _ in range(2):
		if len(remaining) < 2:
			break

		anchor, direction, mask = _ransac_line(remaining, epsilon, iterations)
		if anchor is None:
			break

		inliers = remaining[mask]
		angle_rad = np.arctan2(abs(direction[0]), abs(direction[1]))
		angle_deg = float(np.degrees(angle_rad))

		lines.append(
			{
				"anchor": anchor,
				"direction": direction,
				"inliers": [tuple(map(int, point)) for point in inliers],
				"angle_deg": angle_deg,
			}
		)

		remaining = remaining[~mask]

	return lines


def _line_endpoints(anchor, direction, frame_shape):
	height, width = frame_shape[:2]
	vx, vy = direction
	t_values = []
	epsilon = 1e-9

	if abs(vx) > epsilon:
		t_values.extend([(-anchor[0]) / vx, (width - anchor[0]) / vx])
	if abs(vy) > epsilon:
		t_values.extend([(-anchor[1]) / vy, (height - anchor[1]) / vy])

	endpoints = []
	for t in t_values:
		x = int(anchor[0] + t * vx)
		y = int(anchor[1] + t * vy)
		if 0 <= x <= width and 0 <= y <= height:
			endpoints.append((x, y))

	if len(endpoints) < 2:
		return None, None

	return endpoints[0], endpoints[-1]


def draw_ransac_lines(frame, lines):
	"""Draw fitted RANSAC lines and their inliers."""
	colors = [(255, 200, 0), (0, 200, 255)]

	for index, line in enumerate(lines):
		color = colors[index % len(colors)]
		anchor = line["anchor"]
		direction = line["direction"]
		angle = line["angle_deg"]

		pt1, pt2 = _line_endpoints(anchor, direction, frame.shape)
		if pt1 is not None and pt2 is not None:
			cv2.line(frame, pt1, pt2, color, 2)

		for cx, cy in line["inliers"]:
			cv2.circle(frame, (cx, cy), 7, color, -1)

		ax, ay = int(anchor[0]), int(anchor[1])
		cv2.putText(
			frame,
			f"Line {index + 1}: {angle:.1f} deg",
			(ax + 10, ay - 10 + index * 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			color,
			1,
			cv2.LINE_AA,
		)

	return frame



#Functions for on-image operations
def check_distance(point1, point2):
	x1, y1 = point1
	x2, y2 = point2
	return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def draw_measurement(frame, point1, point2):
    """Draw a line between two points and annotate the distance."""
    if point1 is None or point2 is None:
        return frame
    
    distance = check_distance(point1, point2)
    mid_x = (point1[0] + point2[0]) // 2
    mid_y = (point1[1] + point2[1]) // 2
    
    cv2.line(frame, point1, point2, (0, 255, 0), 2)
    cv2.circle(frame, point1, 5, (0, 255, 0), -1)
    cv2.circle(frame, point2, 5, (0, 255, 0), -1)
    cv2.putText(frame, f"{distance:.1f} px", (mid_x + 8, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return frame



def create_circle_trackbars(window_name="Circle Tuning"):
    """Create a separate window with trackbars for Hough Circle parameters."""
    cv2.namedWindow(window_name)
    cv2.createTrackbar("dp",         window_name, cfg.circles.dp,         5,   lambda v: None)
    cv2.createTrackbar("min_dist",   window_name, cfg.circles.min_dist,   500, lambda v: None)
    cv2.createTrackbar("param1",     window_name, cfg.circles.param1,     500, lambda v: None)
    cv2.createTrackbar("param2",     window_name, cfg.circles.param2,     50, lambda v: None)
    cv2.createTrackbar("min_radius", window_name, cfg.circles.min_radius, 100, lambda v: None)
    cv2.createTrackbar("max_radius", window_name, cfg.circles.max_radius, 500, lambda v: None)


# Area ratios are displayed as whole percentages: 45 on a trackbar means 0.45.
AREA_RATIO_SCALE = 100
AREA_RATIO_TRACKBAR_MAX = 100  # Ratios are constrained to the sensible range 0.00 .. 1.00.
AREA_VALUE_TRACKBAR_MAX = 100_000
AREA_TUNING_WINDOW = "Area Tuning"


def is_area_tuning_window_open(window_name=AREA_TUNING_WINDOW):
    """Return whether the optional area tuning window currently exists."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def create_area_tuning_window(window_name=AREA_TUNING_WINDOW):
    """Open trackbars initialized from the shared live calibration config."""
    if is_area_tuning_window_open(window_name):
        return

    calibration = cfg.calibration
    min_area = max(0, min(int(calibration.min_area), AREA_VALUE_TRACKBAR_MAX))
    max_area = 0 if calibration.max_area is None else max(1, min(int(calibration.max_area), AREA_VALUE_TRACKBAR_MAX))
    low_ratio = max(0, min(round(calibration.area_low_ratio * AREA_RATIO_SCALE), AREA_RATIO_TRACKBAR_MAX))
    up_ratio = max(0, min(round(calibration.area_up_ratio * AREA_RATIO_SCALE), AREA_RATIO_TRACKBAR_MAX))

    cv2.namedWindow(window_name)
    cv2.createTrackbar("area_low_ratio x100", window_name, low_ratio, AREA_RATIO_TRACKBAR_MAX, lambda value: None)
    cv2.createTrackbar("area_up_ratio x100", window_name, up_ratio, AREA_RATIO_TRACKBAR_MAX, lambda value: None)
    cv2.createTrackbar("min_area", window_name, min_area, AREA_VALUE_TRACKBAR_MAX, lambda value: None)
    cv2.createTrackbar("max_area (0=None)", window_name, max_area, AREA_VALUE_TRACKBAR_MAX, lambda value: None)


def update_area_tuning_from_trackbars(window_name=AREA_TUNING_WINDOW):
    """Validate the sliders and copy their values into cfg.calibration immediately."""
    if not is_area_tuning_window_open(window_name):
        return None

    low_ratio_value = cv2.getTrackbarPos("area_low_ratio x100", window_name)
    up_ratio_value = cv2.getTrackbarPos("area_up_ratio x100", window_name)
    min_area = max(0, cv2.getTrackbarPos("min_area", window_name))
    max_area_value = cv2.getTrackbarPos("max_area (0=None)", window_name)

    # Zero disables the maximum. Any enabled maximum is clamped to min_area.
    max_area = None if max_area_value == 0 else max(min_area, max_area_value)
    if max_area is not None and max_area != max_area_value:
        cv2.setTrackbarPos("max_area (0=None)", window_name, max_area)

    calibration = cfg.calibration
    calibration.area_low_ratio = low_ratio_value / AREA_RATIO_SCALE
    calibration.area_up_ratio = up_ratio_value / AREA_RATIO_SCALE
    calibration.min_area = min_area
    calibration.max_area = max_area

    return {
        "area_low_ratio": calibration.area_low_ratio,
        "area_up_ratio": calibration.area_up_ratio,
        "min_area": calibration.min_area,
        "max_area": calibration.max_area,
    }


def sync_area_tuning_window_from_config(window_name=AREA_TUNING_WINDOW):
    """Refresh open area sliders after another feature, such as calibration, changes cfg."""
    if not is_area_tuning_window_open(window_name):
        return

    calibration = cfg.calibration
    cv2.setTrackbarPos("area_low_ratio x100", window_name, round(calibration.area_low_ratio * AREA_RATIO_SCALE))
    cv2.setTrackbarPos("area_up_ratio x100", window_name, round(calibration.area_up_ratio * AREA_RATIO_SCALE))
    cv2.setTrackbarPos("min_area", window_name, min(calibration.min_area, AREA_VALUE_TRACKBAR_MAX))
    max_area = 0 if calibration.max_area is None else min(calibration.max_area, AREA_VALUE_TRACKBAR_MAX)
    cv2.setTrackbarPos("max_area (0=None)", window_name, max_area)


def get_circle_params_from_trackbars(window_name="Circle Tuning"):
    """Read current trackbar values and return them as a dict.
    Falls back to cfg values if the window does not exist yet."""
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return {
            "dp":         cfg.circles.dp,
            "min_dist":   cfg.circles.min_dist,
            "param1":     cfg.circles.param1,
            "param2":     cfg.circles.param2,
            "min_radius": cfg.circles.min_radius,
            "max_radius": cfg.circles.max_radius,
        }
    return {
        "dp":         max(1, cv2.getTrackbarPos("dp",         window_name)),
        "min_dist":   max(1, cv2.getTrackbarPos("min_dist",   window_name)),
        "param1":     max(1, cv2.getTrackbarPos("param1",     window_name)),
        "param2":     max(1, cv2.getTrackbarPos("param2",     window_name)),
        "min_radius": cv2.getTrackbarPos("min_radius", window_name),
        "max_radius": cv2.getTrackbarPos("max_radius", window_name),
    }

def find_closest_blob_center(click_point, blobs):
    """Deprecated diagnostic helper; permanent tracking uses detection assignment."""
    if not blobs:
        return None
    
    x, y = click_point
    centers = [blob["center"] for blob in blobs]
    
    # Calculate Euclidean distance to all detected centers
    distances = [np.sqrt((x - c[0])**2 + (y - c[1])**2) for c in centers]
    min_idx = np.argmin(distances)
    
    return centers[min_idx]


def update_and_draw_tracked_points(frame, blobs, tracked_points, max_distance=50):
    """
    Deprecated diagnostic helper. The force-sensing runtime uses PointTracker.

    Tracks selected points frame-by-frame by linking them to the closest current 
    red blob center. Updates tracked_points in-place and draws tracking overlays.
    """
    if not tracked_points:
        return frame

    current_centers = [blob["center"] for blob in blobs]
    updated_points = []

    for pt in tracked_points:
        if not current_centers:
            # If no blobs are detected this frame, retain last known position
            updated_points.append(pt)
            continue

        # Find the closest currently alive blob to our tracked point
        distances = [np.sqrt((pt[0] - c[0])**2 + (pt[1] - c[1])**2) for c in current_centers]
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # max_distance guard prevents tracking from accidentally jumping to a completely different blob
        if min_dist < max_distance:
            matched_center = current_centers[min_idx]
            updated_points.append(matched_center)
            # Pop to ensure two tracked targets don't merge onto the exact same physical blob
            current_centers.pop(min_idx)
        else:
            # Blob temporarily occluded/lost; retain position
            updated_points.append(pt)

    # Modify the list in-place to update main.py's runtime state dictionary
    tracked_points.clear()
    tracked_points.extend(updated_points)

    # Render tracking markers onto the display image
    for idx, pt in enumerate(tracked_points):
        draw_pt = (int(round(pt[0])), int(round(pt[1])))
        # Draw target reticle (Cyan / Yellow inner ring)


        cv2.circle(frame, draw_pt, 10, (255, 255, 0), 2)
        cv2.circle(frame, draw_pt, 3, (0, 255, 255), -1)
        cv2.putText(frame, f"ID_{idx}", (draw_pt[0] + 12, draw_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    return frame

def compute_and_draw_artificial_point(frame, tracked_points, angle_deg=25):
    """
    Deprecated diagnostic helper. Artificial points are never force-sensing inputs.

    Calculates an artificial point by pivoting the vector from tracked_points[0] 
    to tracked_points[1] by `angle_deg` clockwise around tracked_points[0].
    
    Draws the geometric visualization and returns the artificial point coordinates.
    """
    # We need at least 2 tracked points to form a line and a distance vector
    if len(tracked_points) < 2:
        return None, frame

    p0 = tracked_points[0]  # Center of rotation (Pivot)
    p1 = tracked_points[1]  # Target vector point

    # Convert degrees to radians
    # In an image coordinate space (Y down), positive angle = clockwise rotation
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Get relative vector components
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    # Calculate rotated coordinates relative to the pivot (p0)
    x_art = float(p0[0] + dx * cos_t - dy * sin_t)
    y_art = float(p0[1] + dx * sin_t + dy * cos_t)
    art_point = (x_art, y_art)

    # Keep sub-pixel coordinates internally and convert only for OpenCV drawing.
    draw_p0 = (int(round(p0[0])), int(round(p0[1])))
    draw_p1 = (int(round(p1[0])), int(round(p1[1])))
    draw_art = (int(round(x_art)), int(round(y_art)))

    # --- Visualizations ---
    # 1. Draw a subtle baseline connecting the original two tracked points
    cv2.line(frame, draw_p0, draw_p1, (180, 180, 180), 1, cv2.LINE_AA)
    
    # 2. Draw a thick Magenta line indicating the pivoted offset arm
    cv2.line(frame, draw_p0, draw_art, (255, 0, 255), 2, cv2.LINE_AA)
    
    # 3. Draw the Artificial Point (Solid Magenta circle with a white border)
    cv2.circle(frame, draw_art, 6, (255, 0, 255), -1)
    cv2.circle(frame, draw_art, 10, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 4. Label the artificial point
    cv2.putText(frame, "ART_PT", (draw_art[0] + 14, draw_art[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    return art_point, frame

def filter_and_collect_points(blobs, circles, artificial_point, frame=None):
    """
    Deprecated Hough diagnostic helper. It never adds an artificial coordinate.

    Checks which regular blob centers are inside any detected circles.
    Collects matching real detected points into a single array.
    
    Optionally draws a distinct visual ring around validated points if a frame is provided.
    """
    collected_points = []
    
    # 1. Safely parse and normalize circle formats from cv2.HoughCircles
    valid_circles = []
    if circles is not None:
        if isinstance(circles, np.ndarray):
            # HoughCircles often returns shape (1, N, 3)
            valid_circles = circles[0] if len(circles.shape) == 3 else circles
        else:
            valid_circles = circles

    # 2. Extract centers from detected blobs
    centers = [blob["center"] for blob in blobs] if blobs else []

    # 3. Check each blob center against all detected circles
    for pt in centers:
        px, py = pt
        is_inside = False
        
        for circle in valid_circles:
            # HoughCircles commonly returns uint16 values. Convert before
            # subtraction so negative deltas do not wrap around and overflow.
            cx, cy, r = (float(value) for value in circle)
            px_float, py_float = float(px), float(py)
            # Compare squared distance to avoid computationally expensive square roots
            squared_dist = (px_float - cx) ** 2 + (py_float - cy) ** 2
            if squared_dist <= r ** 2:
                is_inside = True
                break  # Point is verified inside at least one circle; stop checking others
        
        if is_inside:
            point_tuple = (int(px), int(py))
            collected_points.append(point_tuple)
            
            # Optional: Visual confirmation overlay (Green double ring)
            if frame is not None:
                cv2.circle(frame, point_tuple, 14, (255, 0, 0), 1, cv2.LINE_AA)

    # Convert to a standard NumPy array for downstream data applications
    return np.array(collected_points, dtype=np.int32), frame



def create_node_trackbars(window_name="Force Node"):
    """Create a separate window with trackbars for collecting force magnitude [N] and node number associociated with this data"""
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Force Magnitude [N]", window_name, 0, 40, lambda v: None)
    cv2.createTrackbar("Node Number", window_name, 1, 20, lambda v: None)

def get_node_params_from_trackbars(window_name="Force Node"):
	"""Read current trackbar values and return them as a dict.
	Falls back to default values if the window does not exist yet."""
	if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
		return {
			"force_magnitude": 0,
			"node_number": 1,
		}
	return {
		"force_magnitude": cv2.getTrackbarPos("Force Magnitude [N]", window_name),
		"node_number": cv2.getTrackbarPos("Node Number", window_name),
	}

def _generate_unique_base_name(node_number: int, force_magnitude: float) -> str:
    """Generates a unique file base name containing node number, force, and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Replace decimal dots with underscores to ensure clean, safe file names
    force_str = str(force_magnitude).replace('.', '_')
    return f"node_{node_number}_force_{force_str}N_{timestamp}"


def save_points_data(state: dict, node_params: dict) -> Optional[str]:
    """
    Deprecated legacy TXT diagnostic; it is never used as model-training data.

    Save the displayed final points, including the artificial point, to one file.

    The node number comes from the Force Node window. The saved point count and
    ordering match ``final_points_array`` shown by the application.
    Returns the unique base name string so the image saver can mirror it.
    """
    # Extract the metadata params
    node_num = node_params.get("node_number", 1)
    force_mag = node_params.get("force_magnitude", 0)
    if state.get("artificial_point") is None:
        print("Data not saved: select two tracked points to create the artificial point.")
        return None
    
    # Generate the unified unique filename
    base_name = _generate_unique_base_name(node_num, force_mag)
    
    # Determine and create the path for the 'node forces' folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    node_forces_dir = os.path.join(current_dir, "node forces")
    os.makedirs(node_forces_dir, exist_ok=True)
    
    # Do not create a second file that omits the artificial point.
    points_file_path = os.path.join(node_forces_dir, f"{base_name}_points.txt")
    
    # Extract point collections from the application state
    all_points = state.get("final_points_array", np.array([]))
    # Standard header metadata format
    header = (
        f"Node Number: {node_num}\n"
        f"Force Magnitude [N]: {force_mag}\n"
        f"Displayed Point Count: {len(all_points)}\n"
    )

    with open(points_file_path, "w") as f:
        f.write(header)
        f.write("--- Displayed Points (including Artificial Point) ---\n")
        for point_number, pt in enumerate(all_points, start=1):
            f.write(f"Point {point_number}: {pt[0]},{pt[1]}\n")

    print(
        "Data saved successfully:\n"
        f"  -> node forces/{os.path.basename(points_file_path)}"
    )
    return base_name


def save_frame_image(frame: np.ndarray, base_name: str) -> None:
    """
    Saves the provided frame as a PNG in a 'Photos' subfolder,
    matching the exact unique base name used by the companion text logs.
    """
    if frame is None:
        print("Error: Image frame is empty. Cannot save.")
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "Photos")
    
    # Automatically create the 'Photos' subdirectory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)
    
    # Build complete destination image path
    image_path = os.path.join(images_dir, f"{base_name}.png")
    
    # Commit frame to disk using OpenCV serialization
    cv2.imwrite(image_path, frame)
    print(f"Image saved successfully:\n  -> Photos/{os.path.basename(image_path)}")
