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
	get_circle_params_from_trackbars
)
from commands import execute_command
# Default runtime switches (can be toggled with keys)
# Press 'm' in the window to toggle mask-only view on/off
SHOW_MASK_ONLY = False
CALIBRATION_MODE = False
MEASURE_MODE = False


def on_mouse_click(event, x, y, flags, state):
    """Pick the blob under the cursor and store it for calibration, or measure distance."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # measurement mode — handled independently from calibration
    if state["measure_mode"]:
        points = state["measure_points"]
        if len(points) < 2:
            points.append((x, y))
        else:
            state["measure_points"] = [(x, y)]  # reset and start fresh
        return

    # existing calibration logic
    if not state["calibration_mode"]:
        return

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

def main():
	global SHOW_MASK_ONLY, CALIBRATION_MODE, MEASURE_MODE
	pipeline = create_pipeline()
	state = {"blobs": [],
			"selected_blob": None,
			"calibration_mode": False,
			"measure_mode": False,
			"measure_points": [] }
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
			
		
			# choose view based on the runtime switch
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
			


			cv2.imshow("Red Blob Detector", display_frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break
			# toggle mask-only view
			if key == ord("m"):
				SHOW_MASK_ONLY = not SHOW_MASK_ONLY
			if key == ord("c"):
				CALIBRATION_MODE = not CALIBRATION_MODE
				state["calibration_mode"] = CALIBRATION_MODE
				if CALIBRATION_MODE:
					preset = apply_calibration_preset()
					print(
						"Wide red preset loaded: "
						f"red1={preset['lower_red1']}..{preset['upper_red1']} "
						f"red2={preset['lower_red2']}..{preset['upper_red2']}"
					)
				print(f"Calibration mode: {'ON' if CALIBRATION_MODE else 'OFF'}")
			if key == ord("x"):
				state["selected_blob"] = None
			if key == ord("i"):
				MEASURE_MODE = not MEASURE_MODE
				state["measure_mode"] = MEASURE_MODE
				state["measure_points"] = []  # reset measurement points when toggling
				print(f"Measuring mode: {'ON' if MEASURE_MODE else 'OFF'}")
			if key == ord("t"):
				command = input("Command: ")
				execute_command(command, state)



	finally:
		stop_pipeline(pipeline)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

