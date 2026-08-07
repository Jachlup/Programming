"""Hardware-free tests for the PySide6 interface and configuration lifecycle."""

from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QPoint, Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application import ApplicationState
from processing import config_to_mapping, load_config, save_config
from vision_gui.common import VideoLabel
from vision_gui.main_window import MainWindow
from vision_gui.runtime import ApplicationController, CameraWorker


def test_atomic_configuration_round_trip_uses_an_explicit_path(tmp_path: Path) -> None:
	config = load_config()
	config.calibration.min_area += 1
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


def test_main_window_has_four_tabs_and_stops_worker(qtbot) -> None:
	controller = ApplicationController()
	window = MainWindow(controller, auto_start_camera=False)
	qtbot.addWidget(window)
	window.show()
	qtbot.waitUntil(lambda: window.tuning_panel._applied is not None)
	assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
		"Camera", "Reference setup", "Tuning", "Dataset",
	]
	minimum_area = next(
		binding.widget for binding in window.tuning_panel._bindings
		if binding.path == ("calibration", "min_area")
	)
	minimum_area.setValue(minimum_area.value() + 1)
	assert window.tuning_panel.apply.isEnabled()
	window.tuning_panel.revert.click()
	assert not window.tuning_panel.apply.isEnabled()
	window.close()
	assert not controller._thread.isRunning()
