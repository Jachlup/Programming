"""RealSense startup and the high-level visual force-sensing event loop."""

from __future__ import annotations

from datetime import datetime
import os

import cv2
import numpy as np

from application import ApplicationState
from camera import create_pipeline, get_frame, stop_pipeline
from commands import execute_command
from processing import (
	apply_calibration_preset,
	cfg,
	create_area_tuning_window,
	create_node_trackbars,
	get_node_params_from_trackbars,
	save_config,
	save_frame_image,
	update_area_tuning_from_trackbars,
)


WINDOW_NAME = "Red Blob Detector"


def print_keyboard_controls() -> None:
	"""Print every keyboard command handled by the application."""
	print(
		"Keyboard controls:\n"
		"  q / Esc - quit\n"
		"  a       - open area tuning\n"
		"  m       - toggle mask-only view\n"
		"  c       - toggle HSV calibration mode\n"
		"  x       - clear the active mouse-mode selection\n"
		"  i       - toggle distance measurement mode\n"
		"  t       - enter a text command (type 'help' for the workflow)\n"
		"  s       - save configuration and a diagnostic raw-frame snapshot"
	)


def configure_qt_font_directory() -> None:
	"""Use an installed font directory instead of OpenCV's missing wheel path."""
	for directory in (
		"/usr/share/fonts/truetype/dejavu",
		"/usr/share/fonts/truetype/ubuntu",
		"/usr/share/fonts/truetype/liberation2",
	):
		if os.path.isdir(directory):
			os.environ["QT_QPA_FONTDIR"] = directory
			return


def on_mouse_click(
	event: int,
	x: int,
	y: int,
	flags: int,
	state: ApplicationState,
) -> None:
	"""Delegate all mouse-mode behavior to the application coordinator."""
	state.handle_mouse_click(event, x, y, flags)


def _clear_active_selection(state: ApplicationState) -> None:
	state.selected_calibration_blob = None
	state.measure_points = []
	state.cancel_origin_reacquisition()
	if state.reference_origin_selection_mode or state.selected_origin_detection_index is not None:
		state.clear_reference_origin()
	print("Cleared the active mouse-mode selection.")


def _save_diagnostic_snapshot(state: ApplicationState) -> None:
	save_config(cfg)
	if state.current_raw_frame is not None:
		base = f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
		save_frame_image(state.current_raw_frame, base)
	print("Configuration and diagnostic frame saved; training data uses DatasetSession CSV only.")


def main() -> None:
	configure_qt_font_directory()
	print_keyboard_controls()
	state = ApplicationState()
	camera = cfg.camera
	pipeline = create_pipeline(
		width=int(camera.get("width", 640)),
		height=int(camera.get("height", 480)),
		fps=int(camera.get("fps", 60)),
	)
	try:
		cv2.namedWindow(WINDOW_NAME)
		cv2.setMouseCallback(WINDOW_NAME, on_mouse_click, state)
		create_node_trackbars()
		while True:
			update_area_tuning_from_trackbars()
			color_frame = get_frame(pipeline)
			if color_frame is None:
				continue
			raw_frame = np.asanyarray(color_frame.get_data())
			display_frame = state.update_frame(raw_frame)
			cv2.imshow(WINDOW_NAME, display_frame)

			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):
				break
			if key == ord("a"):
				create_area_tuning_window()
			elif key == ord("m"):
				state.display_options.show_mask_only = not state.display_options.show_mask_only
			elif key == ord("c"):
				if (
					state.measure_mode
					or state.reference_origin_selection_mode
					or state.origin_reacquisition_selection_mode
				):
					print("Calibration was not enabled because another mouse mode is active.")
				else:
					state.calibration_mode = not state.calibration_mode
					if state.calibration_mode:
						apply_calibration_preset()
					print(f"Calibration mode: {'ON' if state.calibration_mode else 'OFF'}")
			elif key == ord("x"):
				_clear_active_selection(state)
			elif key == ord("i"):
				if (
					state.calibration_mode
					or state.reference_origin_selection_mode
					or state.origin_reacquisition_selection_mode
				):
					print("Measurement was not enabled because another mouse mode is active.")
				else:
					state.measure_mode = not state.measure_mode
					state.measure_points = []
					print(f"Measurement mode: {'ON' if state.measure_mode else 'OFF'}")
			elif key == ord("t"):
				execute_command(input("Command: "), state)
			elif key == ord("s"):
				node = get_node_params_from_trackbars()
				cfg.force_node.node_number = int(node["node_number"])
				cfg.force_node.force_magnitude = float(node["force_magnitude"])
				_save_diagnostic_snapshot(state)
	finally:
		stop_pipeline(pipeline)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
