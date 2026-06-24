import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
from datetime import datetime


mouse_pos = {"x": 0, "y": 0}


def on_mouse_move(event, x, y, flags, param):
    """Updates the tracked mouse position on any mouse event."""
    mouse_pos["x"] = x
    mouse_pos["y"] = y

def create_pipeline():
    """
    Sets up and starts the RealSense camera pipeline (color stream only).
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280 , 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def find_red_blob_centers(frame):
    """
    Detects red blobs in a BGR frame and returns their geometric centers.

    Red wraps around in HSV, so we combine two masks to catch it fully.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,   120,  70])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 120,  70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter out small noise blobs
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers


def draw_centers(frame, centers):
    for (cx, cy) in centers:
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    return frame

_save_counter = 0  # place this at module level, outside the function

def save_centers_to_csv(centers, folder="CSV"):
    global _save_counter

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "centers.csv")

    file_exists = os.path.isfile(filepath)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "point_index", "x", "y"])

        for (cx, cy) in centers:
            writer.writerow([timestamp, _save_counter, cx, cy])
            _save_counter += 1

    print(f"[Saved] {len(centers)} point(s) appended to {filepath}")

def draw_hsv_at_cursor(frame):
    """
    Reads the HSV value at the current mouse position
    and draws it onto the frame.
    """
    x, y = mouse_pos["x"], mouse_pos["y"]

    # Guard against coordinates outside the frame
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_val, s_val, v_val = hsv[y, x]

    text = f"HSV: ({h_val}, {s_val}, {v_val})"
    cv2.putText(frame, text, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

def find_green_blob_center(frame):
    """
    Detects a single green blob and returns its center, or None if not found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40,  70,  70])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Pick the largest green blob
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def get_intrinsics(pipeline):
    profile = pipeline.get_active_profile()
    stream = profile.get_stream(rs.stream.color)
    return stream.as_video_stream_profile().get_intrinsics()

CALIBRATION_FACTOR = 100 / 77.7  # real mm / measured mm

def pixel_to_mm(cx, cy, intrinsics, Z=85):
    x_mm = (cx - intrinsics.ppx) * Z / intrinsics.fx * CALIBRATION_FACTOR
    y_mm = (cy - intrinsics.ppy) * Z / intrinsics.fy * CALIBRATION_FACTOR
    return x_mm, y_mm

def draw_distances(frame, green_center, red_centers, intrinsics, Z=100):
    if green_center is None:
        return frame

    base_x_mm, base_y_mm = pixel_to_mm(*green_center, intrinsics, Z)

    for (cx, cy) in red_centers:
        x_mm, y_mm = pixel_to_mm(cx, cy, intrinsics, Z)
        dx = round(x_mm - base_x_mm, 1)
        dy = round(y_mm - base_y_mm, 1)

        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"({dx}, {dy}) mm", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# RANSAC line fitting
# ---------------------------------------------------------------------------

def _perp_distance(points, anchor, direction):
    """
    Perpendicular distance from each point to a line defined by:
      anchor    -- any point on the line (np.array [x, y])
      direction -- unit direction vector  (np.array [vx, vy])

    Uses the 2-D cross product:  d = |(P - anchor) x direction|
    This works for ALL angles including vertical lines.
    """
    diff = points - anchor                          # shape (N, 2)
    # cross product in 2D: vx*(dy) - vy*(dx)
    cross = diff[:, 0] * direction[1] - diff[:, 1] * direction[0]
    return np.abs(cross)


def _fit_line(points):
    """
    Fits a line to a set of points using PCA (Total Least Squares).
    Returns (anchor, direction) where both are np.arrays of shape (2,).

    PCA finds the direction of maximum variance, which is the best-fit
    line direction. It minimises perpendicular (not vertical) residuals,
    so it works correctly for vertical and near-vertical lines.
    """
    anchor = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - anchor)
    direction = Vt[0]                               # first principal component
    direction = direction / np.linalg.norm(direction)
    return anchor, direction


def ransac_line(points, epsilon=15, iterations=200, min_inliers=2):
    """
    Fits a single line to `points` using RANSAC.

    Parameters
    ----------
    points      : np.array of shape (N, 2)
    epsilon     : inlier distance threshold in pixels
    iterations  : number of random trials
    min_inliers : minimum inliers required to accept a candidate

    Returns
    -------
    best_anchor    : np.array (2,) -- a point on the best line
    best_direction : np.array (2,) -- unit direction vector of the best line
    inlier_mask    : boolean np.array (N,) -- True for inlier points
    None, None, None if too few points or no consensus found.
    """
    n = len(points)
    if n < 2:
        return None, None, None

    best_inlier_count = 0
    best_mask = np.zeros(n, dtype=bool)

    rng = np.random.default_rng()

    for _ in range(iterations):
        # 1. Pick 2 random points to define a candidate line
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]

        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue                                # coincident points, skip
        direction = direction / norm

        # 2. Count inliers (points within epsilon of the candidate line)
        distances = _perp_distance(points, p1, direction)
        mask = distances < epsilon
        count = mask.sum()

        # 3. Keep the best candidate
        if count > best_inlier_count:
            best_inlier_count = count
            best_mask = mask

    if best_inlier_count < min_inliers:
        return None, None, None

    # 4. Refit using ALL inliers (PCA / Total Least Squares)
    inlier_points = points[best_mask]
    best_anchor, best_direction = _fit_line(inlier_points)

    return best_anchor, best_direction, best_mask


def fit_two_lines_ransac(centers, epsilon=15, iterations=200):
    """
    Detects two lines in a list of (cx, cy) centers using RANSAC.

    Strategy: run RANSAC once, remove inliers, run RANSAC again.

    Parameters
    ----------
    centers    : list of (cx, cy) tuples from find_red_blob_centers()
    epsilon    : inlier distance threshold in pixels  ← tune this
    iterations : RANSAC iterations per line

    Returns
    -------
    lines : list of dicts, each containing:
        "anchor"    -- np.array (2,) point on line
        "direction" -- np.array (2,) unit direction vector
        "inliers"   -- list of (cx, cy) inlier points
        "angle_deg" -- angle from vertical in degrees
    Returns [] if fewer than 4 points given.
    """
    if len(centers) < 4:
        return []

    points = np.array(centers, dtype=float)
    lines = []
    remaining = points.copy()
    remaining_idx = list(range(len(points)))

    for _ in range(2):
        if len(remaining) < 2:
            break

        anchor, direction, mask = ransac_line(remaining, epsilon, iterations)
        if anchor is None:
            break

        inlier_coords = remaining[mask]

        # Angle from vertical: vertical = 0°, horizontal = 90°
        angle_rad = np.arctan2(abs(direction[0]), abs(direction[1]))
        angle_deg = np.degrees(angle_rad)

        lines.append({
            "anchor":    anchor,
            "direction": direction,
            "inliers":   [tuple(p.astype(int)) for p in inlier_coords],
            "angle_deg": angle_deg,
        })

        # Remove inliers before fitting the second line
        remaining = remaining[~mask]

    return lines


def _line_endpoints(anchor, direction, frame_shape):
    """
    Extends a line (anchor + direction) to the frame boundaries.
    Returns two endpoint pixel tuples for cv2.line().
    """
    h, w = frame_shape[:2]
    vx, vy = direction

    # Parametric: P(t) = anchor + t * direction
    # Find t values where the line hits each frame edge
    t_values = []
    eps = 1e-9

    if abs(vx) > eps:
        t_values += [(-anchor[0]) / vx, (w - anchor[0]) / vx]
    if abs(vy) > eps:
        t_values += [(-anchor[1]) / vy, (h - anchor[1]) / vy]

    # Keep only t values where the point is inside the frame
    endpoints = []
    for t in t_values:
        x = int(anchor[0] + t * vx)
        y = int(anchor[1] + t * vy)
        if 0 <= x <= w and 0 <= y <= h:
            endpoints.append((x, y))

    if len(endpoints) < 2:
        return None, None

    return endpoints[0], endpoints[-1]


def draw_ransac_lines(frame, lines):
    """
    Draws the two fitted RANSAC lines and their inlier points onto the frame.

    Line 1 → cyan,   inliers shown as filled circles
    Line 2 → yellow, inliers shown as filled circles
    """
    colors = [(255, 200, 0), (0, 200, 255)]   # yellow, cyan  (BGR)

    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        anchor    = line["anchor"]
        direction = line["direction"]
        angle     = line["angle_deg"]

        # Draw the full line across the frame
        pt1, pt2 = _line_endpoints(anchor, direction, frame.shape)
        if pt1 and pt2:
            cv2.line(frame, pt1, pt2, color, 2)

        # Draw inlier points
        for (cx, cy) in line["inliers"]:
            cv2.circle(frame, (cx, cy), 7, color, -1)

        # Label: angle from vertical
        ax, ay = int(anchor[0]), int(anchor[1])
        cv2.putText(frame, f"Line {i+1}: {angle:.1f} deg",
                    (ax + 10, ay - 10 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
