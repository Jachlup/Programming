"""Image processing helpers for red blob detection and line fitting."""

from __future__ import annotations

import os
import cv2
import numpy as np
import yaml
from dataclasses import dataclass, field, asdict


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
	max_area: int
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
class CircleDetectionConfig:
	dp: int
	min_dist: int
	param1: int
	param2: int
	min_radius: int
	max_radius: int


@dataclass
class AppConfig:
	calibration: CalibrationConfig
	circles: CircleDetectionConfig
	presets: dict

	def active_preset(self) -> HsvPreset:
		"""Return the currently selected HsvPreset."""
		return self.presets[self.calibration.preset]


# ---------------------------------------------------------------------------
# Config loading / saving
# ---------------------------------------------------------------------------

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera-config.yaml")


def load_config() -> AppConfig:
	"""Load AppConfig from camera-config.yaml."""
	if not os.path.exists(_config_path):
		raise FileNotFoundError(f"Config file not found: {_config_path}")
	with open(_config_path, "r") as f:
		raw = yaml.safe_load(f)
	return AppConfig(
		calibration=CalibrationConfig(**raw["calibration"]),
		circles=CircleDetectionConfig(**raw["hough_circles"]),
		presets={name: HsvPreset(**values) for name, values in raw["presets"].items()},
	)


def save_config(cfg: AppConfig) -> None:
	"""Save current AppConfig back to camera-config.yaml."""
	cal = cfg.calibration
	data = {
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
		"hough_circles": asdict(cfg.circles),
		"presets": {
			name: {
				"red_lower_1": preset.red_lower_1.tolist(),
				"red_upper_1": preset.red_upper_1.tolist(),
				"red_lower_2": preset.red_lower_2.tolist(),
				"red_upper_2": preset.red_upper_2.tolist(),
			}
			for name, preset in cfg.presets.items()
		},
	}
	with open(_config_path, "w") as f:
		yaml.dump(data, f, default_flow_style=False)


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


def apply_calibration_preset() -> dict:
	"""Copy the active preset's HSV ranges into calibration and return them."""
	preset = cfg.active_preset()
	cfg.calibration.red_lower_1 = preset.red_lower_1.copy()
	cfg.calibration.red_upper_1 = preset.red_upper_1.copy()
	cfg.calibration.red_lower_2 = preset.red_lower_2.copy()
	cfg.calibration.red_upper_2 = preset.red_upper_2.copy()
	return {
		"lower_red1": cfg.calibration.red_lower_1.copy(),
		"upper_red1": cfg.calibration.red_upper_1.copy(),
		"lower_red2": cfg.calibration.red_lower_2.copy(),
		"upper_red2": cfg.calibration.red_upper_2.copy(),
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


def calibrate_from_blob(
	blob,
	area_low_ratio=None,
	area_up_ratio=None,
	hue_margin=None,
	sat_margin=None,
	val_margin=None,
):
	"""Update the live calibration thresholds from a selected blob."""
	area_low_ratio = area_low_ratio or cfg.calibration.area_low_ratio
	area_up_ratio  = area_up_ratio  or cfg.calibration.area_up_ratio
	hue_margin     = hue_margin     or cfg.calibration.hue_margin
	sat_margin     = sat_margin     or cfg.calibration.sat_margin
	val_margin     = val_margin     or cfg.calibration.val_margin

	area = float(blob["area"])
	cfg.calibration.min_area = max(1, int(area * (1.0 - area_low_ratio)))
	cfg.calibration.max_area = max(cfg.calibration.min_area, int(area * (1.0 + area_up_ratio)))

	sample_hsv = blob.get("sample_hsv")

	hue, sat, val = (int(value) for value in sample_hsv)
	(s1, s2), secondary_hue_range = _build_hue_range(hue, hue_margin)
	cal_s_low = max(0, sat - sat_margin)
	cal_s_high = min(255, sat + sat_margin)
	cal_v_low = max(0, val - val_margin)
	cal_v_high = min(255, val + val_margin)

	cfg.calibration.red_lower_1 = np.array([int(s1), cal_s_low, cal_v_low])
	cfg.calibration.red_upper_1 = np.array([int(s2), cal_s_high, cal_v_high])

	low_h, high_h = secondary_hue_range
	cfg.calibration.red_lower_2 = np.array([int(low_h), cal_s_low, cal_v_low])
	cfg.calibration.red_upper_2 = np.array([int(high_h), cal_s_high, cal_v_high])

	return {
		"area_min":   cfg.calibration.min_area,
		"area_max":   cfg.calibration.max_area,
		"sample_hsv": tuple(int(value) for value in sample_hsv),
		"lower_red1": cfg.calibration.red_lower_1.copy(),
		"upper_red1": cfg.calibration.red_upper_1.copy(),
		"lower_red2": cfg.calibration.red_lower_2.copy(),
		"upper_red2": cfg.calibration.red_upper_2.copy(),
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

		cx = int(moments["m10"] / moments["m00"])
		cy = int(moments["m01"] / moments["m00"])
		x, y, w, h = cv2.boundingRect(contour)
		contour_mask = np.zeros(mask.shape, dtype=np.uint8)
		cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
		mean_hsv = cv2.mean(hsv, mask=contour_mask)
		center_hsv = tuple(int(value) for value in hsv[cy, cx])

		blobs.append(
			{
				"center": (cx, cy),
				"area": area,
				"bbox": (x, y, w, h),
				"contour": contour,
				"center_hsv": center_hsv,
				"mean_hsv": tuple(float(value) for value in mean_hsv[:3]),
			}
		)

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

	cx, cy = blob["center"]
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
		cv2.putText(frame, f"Clicked HSV: {tuple(int(value) for value in sample_hsv)}", (box_x1 + 10, box_y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Lower red 1: {_format_hsv_range(cfg.calibration.red_lower_1, cfg.calibration.red_upper_1)}", (box_x1 + 10, box_y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Lower red 2: {_format_hsv_range(cfg.calibration.red_lower_2, cfg.calibration.red_upper_2)}", (box_x1 + 10, box_y1 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f"Press x to clear selection", (box_x1 + 10, box_y1 + 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	return frame


def draw_centers_with_positions(frame, centers):
	"""Draw blob centers and annotate them with pixel coordinates."""
	for idx, (cx, cy) in enumerate(centers, start=1):
		cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
		cv2.putText(
			frame,
			f"{idx}: ({cx}, {cy})",
			(cx + 8, cy - 8),
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
    cv2.createTrackbar("param2",     window_name, cfg.circles.param2,     500, lambda v: None)
    cv2.createTrackbar("min_radius", window_name, cfg.circles.min_radius, 500, lambda v: None)
    cv2.createTrackbar("max_radius", window_name, cfg.circles.max_radius, 500, lambda v: None)


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