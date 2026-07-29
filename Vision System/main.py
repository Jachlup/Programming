"""Main application loop for RealSense capture and red-blob line detection."""


import os

import cv2
import numpy as np

from camera import create_pipeline, get_frame, stop_pipeline
from processing import *
from commands import execute_command

# Default runtime switches
SHOW_MASK_ONLY = False
CALIBRATION_MODE = False
MEASURE_MODE = False


def print_keyboard_controls():
	"""Print every keyboard command handled by the application."""
	print(
		"Keyboard controls:\n"
		"  q / Esc - quit the program\n"
		"  a       - open the area tuning window\n"
		"  m       - toggle mask-only view\n"
		"  c       - toggle calibration mode\n"
		"  x       - clear calibration selection and tracked points\n"
		"  i       - toggle distance measurement mode\n"
		"  t       - enter a text command\n"
		"  s       - save configuration, point data, and the current frame"
	)


def configure_qt_font_directory():
	"""Use an installed font directory instead of OpenCV's missing wheel path."""
	font_directories = (
		"/usr/share/fonts/truetype/dejavu",
		"/usr/share/fonts/truetype/ubuntu",
		"/usr/share/fonts/truetype/liberation2",
	)
	for directory in font_directories:
		if os.path.isdir(directory):
			os.environ["QT_QPA_FONTDIR"] = directory
			return


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
            print(f"Calibration ignored: no detected red blob at ({x}, {y}).")
            return

        sample_hsv = sample_blob_hsv_near_point(state.get("frame"), blob, (x, y))
        if sample_hsv is None:
            state["selected_blob"] = None
            print(f"Calibration ignored: could not sample a red pixel near ({x}, {y}).")
            return

        selected_blob = dict(blob)
        selected_blob["click_point"] = (x, y)
        selected_blob["sample_hsv"] = sample_hsv
        state["selected_blob"] = selected_blob
        calibration = calibrate_from_blob(selected_blob)
        sync_area_tuning_window_from_config()
        print(
            f"Calibrated blob at ({x}, {y}): HSV={calibration['sample_hsv']}, "
            f"area={calibration['area_min']}..{calibration['area_max']}"
        )
        return

    # 3. Default Point-Tracking Mode (Active when Calibration/Measure are OFF)
    closest_center = find_closest_blob_center((x, y), state["blobs"])
    if closest_center is not None:
        if closest_center not in state["tracked_points"]:
            if len(state["tracked_points"]) >= 2:
                print("Two tracking targets are already selected. Remove one before adding another.")
                return
            state["tracked_points"].append(closest_center)
            print(f"Added tracking target at: {closest_center}")
        else:
            state["tracked_points"].remove(closest_center)
            print(f"Removed tracking target at: {closest_center}")


def main():
	global SHOW_MASK_ONLY, CALIBRATION_MODE, MEASURE_MODE
	configure_qt_font_directory()
	print_keyboard_controls()
	pipeline = create_pipeline()
	
	# Added "tracked_points" list to hold our locked targets
	state = {"blobs": [],
			"selected_blob": None,
			"calibration_mode": False,
			"measure_mode": False,
			"measure_points": [],
			"tracked_points": [],
			"artificial_point": None }
			
	cv2.namedWindow("Red Blob Detector")
	cv2.setMouseCallback("Red Blob Detector", on_mouse_click, state)
	# create_circle_trackbars()
	create_node_trackbars()
	try:
		while True:
			# When open, these sliders are the live source for cfg.calibration.
			update_area_tuning_from_trackbars()

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

			art_pt, display_frame = compute_and_draw_artificial_point(display_frame, state["tracked_points"], angle_deg=-25)
			state["artificial_point"] = art_pt
			# Future applications can read the tracked list right here!
			# e.g., coordinate_data = state["tracked_points"]

			detected_circles_list = detect_circles(red_mask_frame, **circle_params)
			
			final_array, display_frame = filter_and_collect_points(
				blobs=state["blobs"], 
				circles=detected_circles_list, 
				artificial_point=state["artificial_point"],
				frame=display_frame  # Pass display_frame to get visual green rings on screen
			)

			node_params = get_node_params_from_trackbars()


			
			state["final_points_array"] = final_array

			cv2.imshow("Red Blob Detector", display_frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break
			if key == ord("m"):
				SHOW_MASK_ONLY = not SHOW_MASK_ONLY
			if key == ord("a"):
				create_area_tuning_window()
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
			if key == ord("s"):
				# 1. Grab the latest trackbar data
				node_params = get_node_params_from_trackbars()
				
				# 2. Update the active configuration instance directly
				from processing import cfg
				cfg.force_node.node_number = node_params["node_number"]
				cfg.force_node.force_magnitude = float(node_params["force_magnitude"])
				
				# 3. Call your newly unified save function
				from processing import save_config
				save_config(cfg)
				print(f"Configuration and Force Node data saved successfully.")

				unique_base = save_points_data(state, node_params)
				if unique_base is not None:
					save_frame_image(raw_frame, unique_base)

	finally:
		stop_pipeline(pipeline)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
