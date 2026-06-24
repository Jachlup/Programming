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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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
# force sensor calibration
# model calibration
# 3d design
# 