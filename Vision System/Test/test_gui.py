"""Hardware-free tests for the PySide6 interface and configuration lifecycle."""

from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QPoint, Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import application
from application import ApplicationState
import cv2
from camera import _apply_sensor_options
from commands import execute_command
from processing import (
	cfg,
	config_from_mapping,
	config_to_mapping,
	load_config,
	save_config,
	update_config_in_place,
)
from vision_gui.common import VideoLabel
from vision_gui.main_window import MainWindow
from vision_gui.runtime import ApplicationController, CameraWorker


class _DeviceWithoutColorSensor:
	def first_color_sensor(self):
		raise RuntimeError("Could not find requested sensor type!")

	def query_sensors(self):
		return []


class _ProfileWithoutColorSensor:
	def get_device(self):
		return _DeviceWithoutColorSensor()


def test_missing_configurable_color_sensor_is_a_nonfatal_warning() -> None:
	warnings: list[str] = []
	_apply_sensor_options(
		_ProfileWithoutColorSensor(),
		{"auto_exposure": True},
		warnings.append,
	)
	assert warnings
	assert "settings were skipped" in warnings[0]


def test_calibration_commands_share_the_application_state(capsys) -> None:
	original = config_to_mapping(cfg)
	state = ApplicationState()
	try:
		assert execute_command("calibration_start", state)
		assert state.calibration_mode
		assert execute_command("calibration_status", state)
		assert "Calibration mode: ON" in capsys.readouterr().out
		state.selected_calibration_blob = {"area": 250.0, "sample_hsv": (2, 200, 180)}
		assert execute_command("calibration_clear", state)
		assert state.selected_calibration_blob is None
		assert execute_command("calibration_stop", state)
		assert not state.calibration_mode
	finally:
		update_config_in_place(cfg, config_from_mapping(original))


def test_atomic_configuration_round_trip_uses_an_explicit_path(tmp_path: Path) -> None:
	config = load_config()
	config.calibration.min_area = 0
	config.calibration.max_area = 1_000
	target = tmp_path / "camera-config.yaml"
	save_config(config, target)
	reloaded = load_config(target)
	assert config_to_mapping(reloaded) == config_to_mapping(config)
	assert not list(tmp_path.glob("*.tmp"))


def test_worker_executes_registry_commands_and_emits_detached_state(qtbot) -> None:
	state = ApplicationState()
	worker = CameraWorker(state)
	states: list[dict] = []
	completed: list[tuple[str, bool]] = []
	worker.state_updated.connect(states.append)
	worker.command_completed.connect(lambda command, ok: completed.append((command, ok)))
	worker.run_command("show_points off")
	assert completed == [("show_points off", True)]
	assert states[-1]["display_options"]["show_points"] is False
	assert state.display_options.show_points is False


def test_worker_keeps_motor_telemetry_detached_from_application_state() -> None:
	state = ApplicationState()
	worker = CameraWorker(state)
	snapshot = {
		"position_rad": 1.0,
		"velocity_rad_s": 0.0,
		"target_torque_Nm": -2.0,
		"measured_torque_Nm": -1.9,
		"temperature_C": 30.0,
		"state": "HOLDING_TORQUE",
		"feedback_monotonic": 123.0,
		"configuration": {"not": "copied"},
	}
	worker.set_motor_telemetry(snapshot)
	snapshot["position_rad"] = 99.0
	assert worker._latest_motor_telemetry["position_rad"] == 1.0
	assert "configuration" not in worker._latest_motor_telemetry
	assert not hasattr(state, "motor_controller")


def test_worker_applies_live_blob_area_limits() -> None:
	original = config_to_mapping(cfg)
	state = ApplicationState()
	worker = CameraWorker(state)
	applied: list[dict] = []
	worker.configuration_applied.connect(applied.append)
	try:
		worker.set_blob_area_limits(321, 987)
		assert state.config.calibration.min_area == 321
		assert state.config.calibration.max_area == 987
		assert applied[-1]["calibration"]["min_area"] == 321
		worker.set_blob_area_limits(321, None)
		assert state.config.calibration.max_area is None
		worker.set_blob_area_limits(0, 1_000)
		assert state.config.calibration.min_area == 0
		assert state.config.calibration.max_area == 1_000
	finally:
		update_config_in_place(cfg, config_from_mapping(original))


def test_worker_applies_independent_blue_blob_area_limits() -> None:
	original = config_to_mapping(cfg)
	state = ApplicationState(config=config_from_mapping(config_to_mapping(cfg)))
	worker = CameraWorker(state)
	applied: list[dict] = []
	worker.configuration_applied.connect(applied.append)
	try:
		worker.set_blue_blob_area_limits(75, 650)
		assert state.config.blue_calibration.min_area == 75
		assert state.config.blue_calibration.max_area == 650
		assert applied[-1]["blue_calibration"]["min_area"] == 75
		worker.set_blue_blob_area_limits(75, None)
		assert state.config.blue_calibration.max_area is None
	finally:
		update_config_in_place(cfg, config_from_mapping(original))


def test_frame_processing_uses_application_blob_area_limits(monkeypatch) -> None:
	state = ApplicationState(config=config_from_mapping(config_to_mapping(cfg)))
	state.config.calibration.min_area = 123
	state.config.calibration.max_area = 789
	captured: list[tuple[int, int | None]] = []

	def capture(_frame, min_area=None, max_area=None):
		captured.append((min_area, max_area))
		return []

	monkeypatch.setattr(application, "find_red_blob_data", capture)
	state.update_frame(np.zeros((32, 32, 3), dtype=np.uint8))
	assert captured == [(123, 789)]


def test_calibration_click_explains_red_mask_failure(capsys) -> None:
	state = ApplicationState(config=config_from_mapping(config_to_mapping(cfg)))
	state.calibration_mode = True
	state.current_raw_frame = np.zeros((60, 80, 3), dtype=np.uint8)
	state.current_detections = []

	assert not state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 30, 20)
	message = capsys.readouterr().err
	assert "Red-mask check: FAIL" in message
	assert "pixel HSV=" in message
	assert "Active HSV ranges" in message


def test_calibration_click_explains_blob_area_failure(capsys) -> None:
	state = ApplicationState(config=config_from_mapping(config_to_mapping(cfg)))
	state.calibration_mode = True
	state.config.calibration.min_area = 200
	state.config.calibration.max_area = 1_000
	frame = np.zeros((60, 80, 3), dtype=np.uint8)
	red_bgr = cv2.cvtColor(
		np.array([[[5, 220, 100]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
	)[0, 0]
	cv2.rectangle(
		frame, (10, 10), (20, 20), tuple(int(v) for v in red_bgr), thickness=-1
	)
	state.current_raw_frame = frame
	state.current_detections = []

	assert not state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 15, 15)
	message = capsys.readouterr().err
	assert "Red-mask check: PASS" in message
	assert "contour area 100.0 px²" in message
	assert "below the minimum 200 px²" in message


def test_video_label_maps_letterboxed_click_to_frame_coordinates(qtbot) -> None:
	label = VideoLabel()
	label.resize(600, 400)
	label.set_bgr_frame(np.zeros((100, 200, 3), dtype=np.uint8))
	label.show()
	qtbot.waitExposed(label)
	clicks: list[tuple[int, int]] = []
	label.frame_clicked.connect(lambda x, y: clicks.append((x, y)))
	qtbot.mouseClick(label, Qt.MouseButton.LeftButton, pos=QPoint(300, 200))
	assert clicks
	assert abs(clicks[-1][0] - 100) <= 1
	assert abs(clicks[-1][1] - 50) <= 1


def test_main_window_embeds_reference_with_camera_and_stops_worker(qtbot) -> None:
	controller = ApplicationController()
	window = MainWindow(controller, auto_start_camera=False)
	qtbot.addWidget(window)
	window.show()
	qtbot.waitUntil(lambda: window.tuning_panel._applied is not None)
	assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
		"Camera", "Tuning", "Dataset", "Motor", "Collection", "Training",
	]
	assert window.reference_panel is window.camera_panel.reference_panel
	assert window.camera_preview_dock.widget() is window.camera_panel.video
	assert window.camera_preview_dock.isVisible()
	assert not (
		window.camera_preview_dock.features()
		& window.camera_preview_dock.DockWidgetFeature.DockWidgetClosable
	)
	assert window.camera_panel.control_tabs.tabText(2) == "Blue blob"
	assert window.camera_panel.control_tabs.tabText(3) == "Reference setup"
	assert window.camera_panel.blue_blob_panel.mask_only.isCheckable()
	assert window.camera_panel.calibration_mask_only.isCheckable()
	assert window.tuning_panel.mask_only.isCheckable()
	controller.frame_ready.emit(np.zeros((60, 80, 3), dtype=np.uint8))
	qtbot.waitUntil(lambda: window.camera_panel.video._image is not None)
	window.tabs.setCurrentWidget(window.tuning_panel)
	assert window.camera_preview_dock.isVisible()
	assert window.camera_panel.video._image.width() == 80
	assert window.camera_panel.video._image.height() == 60
	mask_state = {
		"display_options": {"show_mask_only": True},
		"mode": "DIAGNOSTIC",
		"runtime_warnings": [],
		"calibration_min_area": 250,
		"calibration_max_area": 900,
	}
	window.camera_panel.update_state(mask_state)
	window.tuning_panel.update_runtime_state(mask_state)
	assert window.camera_panel.calibration_mask_only.isChecked()
	assert window.tuning_panel.mask_only.isChecked()
	assert window.camera_panel.minimum_area_slider.value() == 250
	assert window.camera_panel.maximum_area_slider.value() == 900
	blue_state = {
		**mask_state,
		"display_options": {"show_blue_mask_only": True},
		"blue_target_selected": True,
		"blue_tracking_valid": True,
		"blue_tracking_status": "DETECTED",
		"blue_tracking_quality": 0.8,
		"blue_tracking_reason": "",
		"blue_position": (200.0, 100.0),
		"blue_origin_relative_position": (30.0, -10.0),
		"blue_calibration_min_area": 50,
		"blue_calibration_max_area": 500,
	}
	window.camera_panel.update_state(blue_state)
	blue_panel = window.camera_panel.blue_blob_panel
	assert blue_panel.mask_only.isChecked()
	assert blue_panel.image_position.text() == "(200.00, 100.00) px"
	assert blue_panel.relative_position.text() == "(+30.00, -10.00) px"
	invalid_blue_state = {
		**blue_state,
		"blue_tracking_valid": False,
		"blue_tracking_status": "INVALID",
		"blue_tracking_reason": "Maximum missing-frame limit was exceeded",
		"blue_origin_relative_position": None,
		"blue_relative_unavailable_reason": "Maximum missing-frame limit was exceeded",
	}
	window.camera_panel.update_state(invalid_blue_state)
	assert "missing-frame limit" in blue_panel.tracking_reason.text()
	window.camera_panel.minimum_area_slider.setValue(300)
	window.camera_panel.update_state(mask_state)
	assert window.camera_panel.minimum_area_slider.value() == 300
	assert window.camera_panel.minimum_area_spin.value() == 300
	qtbot.waitUntil(
		lambda: window.camera_panel._pending_area_limits is None,
		timeout=2_000,
	)
	minimum_area = next(
		binding.widget for binding in window.tuning_panel._bindings
		if binding.path == ("calibration", "min_area")
	)
	minimum_area.setValue(minimum_area.value() + 1)
	assert window.tuning_panel.apply.isEnabled()
	window.tuning_panel.revert.click()
	assert not window.tuning_panel.apply.isEnabled()
	window.dataset_panel.update_camera_state(True, False)
	window.dataset_panel.update_state({
		"dataset_active": True,
		"dataset_force_N": 4.0,
		"dataset_pending": 2,
		"reference_configured": True,
		"tracking_active": True,
		"recording_valid": True,
	})
	assert not window.dataset_panel.change_force.isEnabled()
	assert window.dataset_panel.abort.isEnabled()
	window.close()
	assert not controller._thread.isRunning()
