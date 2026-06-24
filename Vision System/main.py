"""Main application loop for RealSense capture and red-blob line detection."""

from __future__ import annotations

import cv2
import numpy as np

from cameraV2 import create_pipeline, get_frame, stop_pipeline
from processing import (
	create_red_mask,
	draw_blob_calibration_info,
	draw_centers_with_positions,
 	calibrate_from_blob,
	apply_calibration_preset,
	find_red_blob_data,
	get_blob_at_point,
	sample_hsv_at_point,
	show_red_mask_only,
	detect_circles,
	draw_circles,
	create_circle_trackbars,
	get_circle_params_from_trackbars,
    # --- New Tracking Imports ---
    find_closest_blob_center,
    update_and_draw_tracked_points
)
from commands import execute_command

# Default runtime switches
SHOW_MASK_ONLY = False
CALIBRATION_MODE = False
MEASURE_MODE = False


def on_mouse_click(event, x, y, flags, state):
    """Pick the blob under the cursor and store it for calibration, measure distance, or track points."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # 1. Measurement mode handling
    if state["measure_mode"]:
        points = state["measure_points"]
        if len(points) < 2:
            points.append((x, y))
        else:
            state["measure_points"] = [(x, y)]
        return

    # 2. Calibration mode handling
    if state["calibration_mode"]:
        blob = get_blob_at_point(state["blobs"], (x, y))
        if blob is None:
            state["selected_blob"] = None
            return

        sample_hsv = sample_hsv_at_point(state.get("frame"), (x, y))
        selected_blob = dict(blob)
        selected_blob["click_point"] = (x, y)
        selected_blob["sample_hsv"] = sample_hsv
        state["selected_blob"] = selected_blob
        calibrate_from_blob(selected_blob)
        return

    # 3. Default Point-Tracking Mode (Active when Calibration/Measure are OFF)
    closest_center = find_closest_blob_center((x, y), state["blobs"])
    if closest_center is not None:
        if closest_center not in state["tracked_points"]:
            state["tracked_points"].append(closest_center)
            print(f"Added tracking target at: {closest_center}")
        else:
            state["tracked_points"].remove(closest_center)
            print(f"Removed tracking target at: {closest_center}")


def main():
	global SHOW_MASK_ONLY, CALIBRATION_MODE, MEASURE_MODE
	pipeline = create_pipeline()
	
	# Added "tracked_points" list to hold our locked targets
	state = {"blobs": [],
			"selected_blob": None,
			"calibration_mode": False,
			"measure_mode": False,
			"measure_points": [],
			"tracked_points": [] } 
			
	cv2.namedWindow("Red Blob Detector")
	cv2.setMouseCallback("Red Blob Detector", on_mouse_click, state)
	create_circle_trackbars()

	try:
		while True:
			color_frame = get_frame(pipeline)
			if color_frame is None:
				continue

			raw_frame = np.asanyarray(color_frame.get_data())
			state["frame"] = raw_frame
			blobs = find_red_blob_data(raw_frame)
			state["blobs"] = blobs
			red_mask_frame = create_red_mask(raw_frame)
			
			if SHOW_MASK_ONLY:
				display_frame = show_red_mask_only(raw_frame)
			else:
				display_frame = raw_frame.copy()

			centers = [blob["center"] for blob in blobs]
			display_frame = draw_centers_with_positions(display_frame, centers)
			display_frame = draw_blob_calibration_info(display_frame, state["selected_blob"])
			
			circle_params = get_circle_params_from_trackbars()
			state["circle_params"] = circle_params
			display_frame = draw_circles(display_frame, detect_circles(red_mask_frame, **circle_params))
			
			# --- Real-time tracking pipeline step ---
			display_frame = update_and_draw_tracked_points(display_frame, blobs, state["tracked_points"])
			
			# Future applications can read the tracked list right here!
			# e.g., coordinate_data = state["tracked_points"]

			cv2.imshow("Red Blob Detector", display_frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break
			if key == ord("m"):
				SHOW_MASK_ONLY = not SHOW_MASK_ONLY
			if key == ord("c"):
				CALIBRATION_MODE = not CALIBRATION_MODE
				state["calibration_mode"] = CALIBRATION_MODE
				if CALIBRATION_MODE:
					preset = apply_calibration_preset()
				print(f"Calibration mode: {'ON' if CALIBRATION_MODE else 'OFF'}")
			if key == ord("x"):
				state["selected_blob"] = None
				state["tracked_points"] = []  # Clear tracking targets on 'x' key press
				print("Cleared selections and tracking points.")
			if key == ord("i"):
				MEASURE_MODE = not MEASURE_MODE
				state["measure_mode"] = MEASURE_MODE
				state["measure_points"] = []
				print(f"Measuring mode: {'ON' if MEASURE_MODE else 'OFF'}")
			if key == ord("t"):
				command = input("Command: ")
				execute_command(command, state)

	finally:
		stop_pipeline(pipeline)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()