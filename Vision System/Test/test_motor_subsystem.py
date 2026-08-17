"""Hardware-free tests for motor validation, state transitions, and watchdogs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import pytest
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motor_control import (
	CppMotorBackend,
	MockMotorBackend,
	MotorConfig,
	MotorController,
	MotorPanel,
	MotorState,
	MotorWorker,
	load_motor_config,
	save_motor_config,
)
from vision_gui.main_window import MainWindow
from vision_gui.runtime import ApplicationController


def _config(**changes) -> MotorConfig:
	values = {
		"feedback_interval_ms": 5,
		"homing_timeout_s": 0.3,
		"movement_timeout_s": 0.4,
		"stall_duration_s": 0.02,
		"stall_epsilon_rad": 0.005,
		"position_tolerance_rad": 0.01,
		**changes,
	}
	return replace(MotorConfig(), **values)


class _Factory:
	def __init__(self, backend=None):
		self.backend = backend
		self.created = []

	def __call__(self, config):
		backend = self.backend or MockMotorBackend(
			minimum_position_rad=config.minimum_position_rad,
			maximum_position_rad=config.maximum_position_rad,
			initial_position_rad=0.0,
		)
		self.created.append(backend)
		return backend


class _StuckPositionBackend(MockMotorBackend):
	def _advance(self) -> None:
		self._last_update = time.monotonic()
		self.velocity_rad_s = 0.0
		if self.mode == "RAW_TORQUE":
			self.measured_torque_Nm = self.commanded_torque_Nm


@pytest.fixture
def worker(qtbot):
	factory = _Factory()
	instance = MotorWorker(_config(), factory)
	instance.initialize()
	yield instance, factory
	instance._timer.stop()


def _connect_enable_home(qtbot, worker, factory):
	worker.connect_motor()
	assert worker.state == MotorState.IDLE
	backend = factory.created[-1]
	assert backend.current_limit_A == 0.5
	assert backend.maximum_torque_Nm == 10.0
	assert not backend.enabled
	worker.enable_drive()
	worker.start_homing()
	qtbot.waitUntil(lambda: worker.state == MotorState.IDLE and worker._homed, timeout=800)
	return backend


def test_confirmed_configuration_and_force_conversion_are_validated() -> None:
	config = MotorConfig()
	assert config.backend == "mock"
	assert config.closing_direction == -1
	assert config.profile_velocity_rad_s == 5.0
	assert config.profile_acceleration_rad_s2 == 20.0
	assert config.maximum_closing_torque_Nm == 4.0
	assert config.maximum_controller_torque_Nm == 10.0
	assert config.current_limit_A == 0.5
	assert config.minimum_position_rad == -1.0
	assert config.maximum_position_rad == 2.2
	assert config.maximum_temperature_C == 50.0
	with pytest.raises(ValueError, match="pulley_radius"):
		config.torque_for_finger_force(5.0)
	converted = replace(config, pulley_radius_m=0.02, efficiency=0.5)
	assert converted.torque_for_finger_force(-25.0) == -1.0
	with pytest.raises(ValueError, match="fixed safety limit"):
		replace(config, current_limit_A=0.6)
	with pytest.raises(ValueError, match="must be -1"):
		replace(config, closing_direction=1, homing_torque_Nm=0.6)
	assert replace(config, minimum_position_rad=0.0).minimum_position_rad == 0.0
	with pytest.raises(ValueError, match="fixed safety limit -1"):
		replace(config, minimum_position_rad=-1.001)
	with pytest.raises(ValueError, match="homed 0 rad origin"):
		replace(config, minimum_position_rad=0.1)


def test_motor_range_configuration_round_trips_atomically(tmp_path) -> None:
	configured = replace(
		MotorConfig(),
		minimum_position_rad=-1.0,
		maximum_position_rad=1.9,
		open_position_rad=1.8,
		maximum_closing_torque_Nm=2.5,
	)
	target = tmp_path / "motor-config.yaml"
	save_motor_config(configured, target)
	assert load_motor_config(target) == configured
	assert not list(tmp_path.glob("*.tmp"))


def test_cpp_hardware_backend_is_explicit_and_requires_built_bridge(tmp_path) -> None:
	missing = tmp_path / "gripper_bridge"
	backend = CppMotorBackend(executable_path=missing)
	assert backend.backend_type == "hardware-cpp"
	assert CppMotorBackend.default_executable().name == "gripper_bridge"
	with pytest.raises(FileNotFoundError, match=r"C\+\+ motor bridge"):
		backend.connect()
	assert not backend.connected
	assert not backend.enabled


def test_cpp_backend_debug_trace_contains_native_output_and_exact_protocol(tmp_path) -> None:
	bridge = tmp_path / "fake_motor_bridge"
	bridge.write_text(
		"""#!/usr/bin/env python3
import sys

print("fake native SDK startup", flush=True)
print("MOTOR_BRIDGE OK READY 1", flush=True)
for raw_line in sys.stdin:
    command = raw_line.strip()
    if command == "FEEDBACK":
        print("MOTOR_BRIDGE OK 1.25 0.5 -0.75 26", flush=True)
    else:
        print("MOTOR_BRIDGE OK", flush=True)
    if command == "QUIT":
        break
""",
		encoding="utf-8",
	)
	bridge.chmod(0o755)
	backend = CppMotorBackend(executable_path=bridge)
	backend.connect()
	try:
		backend.apply_current_limit(0.5)
		backend.apply_maximum_torque(10.0)
		backend.stop_idle()
		assert backend.read_position() == pytest.approx(1.25)
		assert backend.read_velocity() == pytest.approx(0.5)
		assert backend.read_measured_torque() == pytest.approx(-0.75)
		assert backend.read_temperature() == pytest.approx(26.0)
	finally:
		backend.disconnect()

	events = backend.drain_debug_events()
	formatted = [
		f"{event['category']} {event['message']}"
		for event in events
	]
	assert any(line == "NATIVE fake native SDK startup" for line in formatted)
	assert any(line.startswith("TX CONFIGURE 0.5 10 -1 2.2000000000000002") for line in formatted)
	assert "TX CURRENT_LIMIT 0.5" in formatted
	assert "TX MAX_TORQUE 10" in formatted
	assert "TX FEEDBACK" in formatted
	assert "RX OK 1.25 0.5 -0.75 26" in formatted
	assert "TX QUIT" in formatted
	assert any("euid=" in line and "groups=" in line for line in formatted)


def test_connect_disconnect_enable_disable_and_no_implicit_enable(worker) -> None:
	instance, factory = worker
	states = []
	instance.state_updated.connect(states.append)
	instance.connect_motor()
	backend = factory.created[-1]
	assert instance.state == MotorState.IDLE
	assert backend.connected and not backend.enabled
	instance.enable_drive()
	assert backend.enabled
	instance.disable_drive()
	assert not backend.enabled
	instance.disconnect_motor()
	assert instance.state == MotorState.DISCONNECTED
	instance.disconnect_motor()
	assert states


def test_ranges_update_only_while_disconnected(worker) -> None:
	instance, factory = worker
	completed = []
	instance.command_completed.connect(
		lambda command, ok, message: completed.append((command, ok, message))
	)
	instance.configure_ranges(-1.0, 1.8, 1.6, -2.0, 3.0, 12.0)
	assert instance.config.minimum_position_rad == pytest.approx(-1.0)
	assert instance.config.maximum_position_rad == pytest.approx(1.8)
	assert instance.config.open_position_rad == pytest.approx(1.6)
	assert instance.config.maximum_closing_torque_Nm == pytest.approx(2.0)
	assert instance.config.profile_velocity_rad_s == pytest.approx(3.0)
	assert instance.config.profile_acceleration_rad_s2 == pytest.approx(12.0)
	assert completed[-1][0:2] == ("configure_ranges", True)

	instance.connect_motor()
	backend = factory.created[-1]
	assert backend.minimum_position_rad == pytest.approx(-1.0)
	assert backend.maximum_position_rad == pytest.approx(1.8)
	instance.configure_ranges(-0.5, 1.5, 1.4, -1.0, 2.0, 8.0)
	assert instance.config.minimum_position_rad == pytest.approx(-1.0)
	assert instance.config.maximum_position_rad == pytest.approx(1.8)
	assert completed[-1][0:2] == ("configure_ranges", False)
	assert "Disconnect" in completed[-1][2]
	instance.disconnect_motor()


def test_connection_failure_enters_fault_without_hardware(qtbot) -> None:
	backend = MockMotorBackend()
	backend.fail_operations.add("connect")
	worker = MotorWorker(_config(), _Factory(backend))
	worker.initialize()
	try:
		worker.connect_motor()
		assert worker.state == MotorState.FAULT
		assert not worker._connected
	finally:
		worker._timer.stop()


def test_homing_success_and_cancellation_are_incremental(qtbot, worker) -> None:
	instance, factory = worker
	backend = _connect_enable_home(qtbot, instance, factory)
	assert backend.position_rad == 0.0
	assert backend.mode == "IDLE"

	instance.start_homing()
	assert instance.state == MotorState.HOMING
	started = time.monotonic()
	instance.stop_motor()
	assert time.monotonic() - started < 0.1
	assert instance.state == MotorState.IDLE
	assert not backend.enabled
	assert backend.commanded_torque_Nm == 0.0


def test_unzeroed_position_below_command_minimum_can_connect_and_home(qtbot) -> None:
	backend = MockMotorBackend(
		minimum_position_rad=-1.0,
		maximum_position_rad=2.2,
		initial_position_rad=-1.5,
	)
	worker = MotorWorker(_config(minimum_position_rad=-1.0), _Factory(backend))
	worker.initialize()
	try:
		worker.connect_motor()
		assert worker.state == MotorState.IDLE
		assert worker._position_rad == pytest.approx(-1.5)
		worker.enable_drive()
		worker.start_homing()
		qtbot.waitUntil(
			lambda: worker.state == MotorState.IDLE and worker._homed,
			timeout=800,
		)
		assert worker._position_rad == pytest.approx(0.0)

		worker.move_to_position(-1.1, 5.0, 20.0)
		assert worker.state == MotorState.IDLE
		assert "outside -1..2.2" in worker._latest_warning
	finally:
		worker._timer.stop()


def test_homing_timeout_enters_fault_and_disables(qtbot) -> None:
	config = _config(homing_timeout_s=0.03, stall_duration_s=1.0)
	backend = MockMotorBackend(initial_position_rad=1.0)
	worker = MotorWorker(config, _Factory(backend))
	worker.initialize()
	try:
		worker.connect_motor()
		worker.enable_drive()
		worker.start_homing()
		qtbot.waitUntil(lambda: worker.state == MotorState.FAULT, timeout=500)
		assert not backend.enabled
		assert backend.commanded_torque_Nm == 0.0
		assert "timed out" in worker._latest_error.lower()
	finally:
		worker._timer.stop()


def test_position_completion_timeout_and_limit_rejection(qtbot, worker) -> None:
	instance, factory = worker
	backend = _connect_enable_home(qtbot, instance, factory)
	instance.move_to_position(0.2, 5.0, 20.0)
	assert instance.state == MotorState.MOVING
	qtbot.waitUntil(lambda: instance.state == MotorState.IDLE, timeout=800)
	assert backend.position_rad == pytest.approx(0.2, abs=0.01)

	instance.move_to_position(3.0, 5.0, 20.0)
	assert instance.state == MotorState.IDLE
	assert "outside" in instance._latest_warning
	instance.start_torque_hold(-4.1)
	assert instance.state == MotorState.IDLE
	assert "exceeds" in instance._latest_warning


def test_position_timeout_and_feedback_failure_enter_fault(qtbot) -> None:
	backend = _StuckPositionBackend(initial_position_rad=0.0)
	worker = MotorWorker(_config(movement_timeout_s=0.03), _Factory(backend))
	worker.initialize()
	try:
		worker.connect_motor()
		worker.enable_drive()
		worker._homed = True
		worker.move_to_position(1.0, 5.0, 20.0)
		qtbot.waitUntil(lambda: worker.state == MotorState.FAULT, timeout=500)
		assert "timed out" in worker._latest_error.lower()
		assert not backend.enabled
	finally:
		worker._timer.stop()

	feedback = MockMotorBackend(initial_position_rad=0.0)
	feedback_worker = MotorWorker(_config(), _Factory(feedback))
	feedback_worker.initialize()
	try:
		feedback_worker.connect_motor()
		feedback.fail_operations.add("read_position")
		qtbot.waitUntil(lambda: feedback_worker.state == MotorState.FAULT, timeout=500)
		assert "read_position" in feedback_worker._latest_error
	finally:
		feedback_worker._timer.stop()


def test_torque_ramp_watchdog_backoff_and_stop(qtbot, worker) -> None:
	instance, factory = worker
	backend = _connect_enable_home(qtbot, instance, factory)
	instance.start_torque_hold(-1.0)
	assert instance.state == MotorState.HOLDING_TORQUE
	qtbot.waitUntil(lambda: instance._commanded_torque_Nm < -0.02, timeout=500)
	assert any(call[0] == "refresh" for call in backend.calls)

	position_before = backend.position_rad
	instance.update_target_torque(-0.5)
	assert instance.state == MotorState.MOVING
	assert instance._target_position_rad == pytest.approx(
		min(position_before + 0.3, 2.2), abs=0.02
	)
	instance.stop_motor()
	assert instance.state == MotorState.IDLE
	assert not backend.enabled
	assert backend.mode == "IDLE"
	instance.stop_motor()
	assert instance.state == MotorState.IDLE


def test_torque_success_waits_for_ramp_and_measured_feedback(qtbot, worker) -> None:
	instance, factory = worker
	_connect_enable_home(qtbot, instance, factory)
	completed = []
	trace = []
	instance.command_completed.connect(
		lambda command, ok, message: completed.append((command, ok, message))
	)
	instance.debug_events_emitted.connect(trace.extend)

	instance.start_torque_hold(-0.05)
	assert instance.state == MotorState.HOLDING_TORQUE
	assert not instance._torque_confirmed
	assert not any(command == "torque" and ok for command, ok, _ in completed)
	assert any(event["category"] == "ACCEPTED" for event in trace)

	qtbot.waitUntil(lambda: instance._torque_confirmed, timeout=1000)
	assert any(command == "torque" and ok for command, ok, _ in completed)
	assert any(event["category"] == "FEEDBACK" for event in trace)
	assert any(
		event["category"] == "RESULT" and "torque: CONFIRMED" in event["message"]
		for event in trace
	)


def test_watchdog_feedback_and_temperature_failures_stop_drive(qtbot, worker) -> None:
	instance, factory = worker
	backend = _connect_enable_home(qtbot, instance, factory)
	instance.start_torque_hold(-0.5)
	backend.fail_operations.add("refresh_torque")
	qtbot.waitUntil(lambda: instance.state == MotorState.FAULT, timeout=500)
	assert not backend.enabled

	# A fresh worker verifies the independent temperature safety path.
	hot = MockMotorBackend(initial_position_rad=0.0)
	hot_worker = MotorWorker(_config(), _Factory(hot))
	hot_worker.initialize()
	try:
		hot_worker.connect_motor()
		hot.temperature_C = 51.0
		qtbot.waitUntil(lambda: hot_worker.state == MotorState.FAULT, timeout=500)
		assert "temperature" in hot_worker._latest_error.lower()
	finally:
		hot_worker._timer.stop()


def test_controller_delivers_signals_off_gui_thread_and_shutdown_is_idempotent(qtbot) -> None:
	controller = MotorController(_config(), _Factory())
	states = []
	controller.state_updated.connect(states.append)
	try:
		controller.connect_motor()
		qtbot.waitUntil(lambda: any(state["connected"] for state in states), timeout=800)
		assert controller._thread.isRunning()
		assert controller._worker.thread() is controller._thread
	finally:
		assert controller.shutdown()
	assert not controller._thread.isRunning()
	assert controller.shutdown()


def test_motor_panel_button_rules_and_direct_stop(qtbot) -> None:
	controller = MotorController(_config(), _Factory())
	panel = MotorPanel(controller)
	qtbot.addWidget(panel)
	panel.show()
	try:
		panel.force_mode.setCurrentIndex(panel.force_mode.findData("finger_force"))
		assert panel.torque.isEnabled()
		panel.torque.setValue(-0.75)
		panel.torque.editingFinished.emit()
		assert panel.force_mode.currentData() == "raw_torque"
		assert panel.torque.value() == pytest.approx(-0.75)

		panel.range_minimum_position.setValue(-1.0)
		panel.range_maximum_position.setValue(1.8)
		panel.range_open_position.setValue(1.6)
		panel.range_minimum_torque.setValue(-2.0)
		panel.range_maximum_velocity.setValue(3.0)
		panel.range_maximum_acceleration.setValue(12.0)
		panel._confirm_action = lambda *_args: True
		panel.apply_ranges_button.click()
		qtbot.waitUntil(
			lambda: panel.config.maximum_position_rad == pytest.approx(1.8),
			timeout=800,
		)
		assert panel.config.minimum_position_rad == pytest.approx(-1.0)
		assert panel.config.open_position_rad == pytest.approx(1.6)
		assert panel.config.maximum_closing_torque_Nm == pytest.approx(2.0)
		assert panel.config.profile_velocity_rad_s == pytest.approx(3.0)
		assert panel.config.profile_acceleration_rad_s2 == pytest.approx(12.0)
		assert panel.position.maximum() == pytest.approx(1.8)
		assert panel.position.minimum() == pytest.approx(-1.0)
		assert panel.torque.minimum() == pytest.approx(-2.0)
		assert panel.velocity.maximum() == pytest.approx(3.0)
		assert panel.acceleration.maximum() == pytest.approx(12.0)

		panel.update_state({
			"backend_type": "mock",
			"state": "DISCONNECTED",
			"connected": False,
			"enabled": False,
			"homed": False,
		})
		assert panel.connect_button.isEnabled()
		assert not panel.stop_button.isEnabled()

		panel.update_state({
			"backend_type": "mock",
			"state": "HOLDING_TORQUE",
			"connected": True,
			"enabled": True,
			"homed": True,
			"commanded_torque_Nm": -0.2,
		})
		assert panel.stop_button.isEnabled()
		assert panel.release_button.isEnabled()
		assert panel.torque_button.isEnabled()
		assert not panel.range_box.isEnabled()

		assert not panel.debug_box.isVisible()
		panel.debug_toggle.setChecked(True)
		assert panel.debug_box.isVisible()
		panel.append_debug_events([{
			"wall_time": 1.0,
			"source": "C++ bridge client",
			"category": "TX",
			"message": "TORQUE -0.25",
		}])
		assert "[C++ bridge client] [TX] TORQUE -0.25" in panel.debug_trace.toPlainText()
		panel.debug_copy_button.click()
		assert "TORQUE -0.25" in panel.debug_trace.toPlainText()
		panel.debug_clear_button.click()
		assert panel.debug_trace.toPlainText() == ""
		qtbot.mouseClick(panel.stop_button, Qt.MouseButton.LeftButton)
	finally:
		assert controller.shutdown()


def test_main_window_closes_both_workers_while_torque_is_active(qtbot) -> None:
	application_controller = ApplicationController()
	motor_controller = MotorController(_config(), _Factory())
	window = MainWindow(
		application_controller,
		motor_controller,
		auto_start_camera=False,
	)
	qtbot.addWidget(window)
	window.show()
	states = []
	motor_controller.state_updated.connect(states.append)
	motor_controller.connect_motor()
	qtbot.waitUntil(lambda: any(state["connected"] for state in states), timeout=800)
	motor_controller.enable_drive()
	qtbot.waitUntil(lambda: any(state["enabled"] for state in states), timeout=800)
	motor_controller.home()
	qtbot.waitUntil(lambda: any(state["homed"] for state in states), timeout=800)
	motor_controller.hold_torque(-0.5)
	qtbot.waitUntil(
		lambda: any(state["state"] == "HOLDING_TORQUE" for state in states),
		timeout=800,
	)
	window.close()
	assert not motor_controller._thread.isRunning()
	assert not application_controller._thread.isRunning()
