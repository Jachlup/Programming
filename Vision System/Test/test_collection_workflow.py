"""Hardware-free tests for collection protocols, stability, and coordination."""

from pathlib import Path
from types import SimpleNamespace
import sys
import time

from PySide6.QtCore import QObject, Signal
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collection_protocol import protocol_from_mapping
from experiment_control import ExperimentController, ExperimentState, StabilityMonitor


def _mapping(*, steps=None, confirmations=True, duration=0.0):
	protocol = {
		"experiment_id": "collection-test",
		"samples_per_step": 1,
		"settling_time_s": 0.0,
		"sample_interval_ms": 25,
		"required_stability_duration_s": duration,
		"require_operator_confirmation": confirmations,
		"use_motor": True,
		"maximum_motor_telemetry_age_s": 0.5,
		"maximum_motor_velocity_rad_s": 0.05,
		"maximum_torque_variation_Nm": 0.1,
		"minimum_tracking_quality": 0.5,
		"minimum_origin_quality": 0.5,
		"minimum_geometry_quality": 0.5,
		"maximum_feature_movement": 0.2,
		"require_zero_force_baseline": False,
	}
	if steps is None:
		protocol.update({
			"force_levels_N": [0.0, 5.0],
			"repetitions": 2,
			"sequence": "loading_unloading",
		})
	else:
		protocol["steps"] = steps
	return {"collection_protocol": protocol}


def _vision(*, active=False, accepted=0, pending=0, root=""):
	return {
		"frame_number": accepted + 1,
		"reference_configured": True,
		"tracking_active": True,
		"tracking_valid": True,
		"tracking_quality": 0.9,
		"origin_valid": True,
		"origin_tracking_quality": 0.9,
		"geometry_valid": True,
		"geometry_quality": 0.9,
		"blue_tracking_valid": True,
		"detected_marker_count": 7,
		"expected_marker_count": 7,
		"feature_names": ("x", "y"),
		"feature_values": (1.0, 2.0),
		"dataset_active": active,
		"dataset_accepted": accepted,
		"dataset_pending": pending,
		"dataset_root": root,
		"dataset_directory": root,
		"nearest_red_point_to_blue": "LINE_A_01",
	}


def _motor(**updates):
	snapshot = {
		"connected": True,
		"enabled": True,
		"homed": True,
		"state": "HOLDING_TORQUE",
		"position_rad": 1.0,
		"velocity_rad_s": 0.0,
		"target_torque_Nm": -2.0,
		"measured_torque_Nm": -2.0,
		"temperature_C": 30.0,
		"feedback_monotonic": time.monotonic(),
		"torque_within_tolerance": True,
	}
	snapshot.update(updates)
	return snapshot


class _ApplicationController(QObject):
	state_updated = Signal(object)
	camera_state_changed = Signal(bool, bool)
	command_completed = Signal(str, bool)
	experiment_batch_queued = Signal(str, bool, str)

	def __init__(self):
		super().__init__()
		self.commands = []
		self.batches = []
		self.aborts = 0

	def execute(self, command): self.commands.append(command)
	def queue_experiment_batch(self, batch): self.batches.append(batch)
	def abort_experiment_batch(self): self.aborts += 1


class _MotorController(QObject):
	state_updated = Signal(object)

	def __init__(self):
		super().__init__()
		self.actions = []
		self.config = SimpleNamespace(open_position_rad=2.0, position_tolerance_rad=0.5)

	def stop_motor(self): self.actions.append(("stop",))
	def enable_drive(self): self.actions.append(("enable",))
	def open_gripper(self): self.actions.append(("open",))
	def hold_torque(self, torque): self.actions.append(("torque", torque))


def test_protocol_generates_unique_loading_and_unloading_steps() -> None:
	protocol = protocol_from_mapping(_mapping())
	assert len(protocol.steps) == 8
	assert len({step.force_step_id for step in protocol.steps}) == 8
	assert {step.direction for step in protocol.steps} == {
		"baseline", "loading", "unloading",
	}
	assert {step.repetition for step in protocol.steps} == {1, 2}


def test_stability_requires_a_consecutive_low_motion_window() -> None:
	protocol = protocol_from_mapping(_mapping(duration=0.2))
	monitor = StabilityMonitor(protocol)
	vision = _vision()
	motor = _motor(feedback_monotonic=10.0)
	assert not monitor.update(vision, motor, now=10.0).stable
	vision["frame_number"] = 2
	vision["feature_values"] = (1.05, 2.0)
	assert not monitor.update(vision, motor, now=10.1).stable
	vision["frame_number"] = 3
	assert monitor.update(vision, motor, now=10.2).stable
	vision["frame_number"] = 4
	vision["feature_values"] = (2.0, 2.0)
	result = monitor.update(vision, motor, now=10.3)
	assert not result.stable
	assert "movement" in result.reason.lower()


def test_operator_guided_step_queues_confirmed_force_and_writes_progress(
	tmp_path: Path,
) -> None:
	protocol_path = tmp_path / "protocol.yaml"
	protocol_path.write_text(yaml.safe_dump(_mapping(steps=[{
		"force_N": 5.0,
		"direction": "loading",
		"repetition": 1,
		"samples": 1,
		"force_step_id": "loading-r01-s001",
	}])), encoding="utf-8")
	dataset_root = tmp_path / "dataset"
	dataset_root.mkdir()
	application = _ApplicationController()
	motor = _MotorController()
	runner = ExperimentController(application, motor)
	try:
		runner.load_protocol(protocol_path)
		application.camera_state_changed.emit(True, False)
		application.state_updated.emit(_vision(root=str(dataset_root)))
		motor.state_updated.emit(_motor())
		assert runner.preflight_errors() == []
		runner.start()
		assert application.commands[-1].startswith("dataset_start collection-test")

		application.state_updated.emit(_vision(active=True, root=str(dataset_root)))
		assert runner.state == ExperimentState.WAITING_FOR_OPERATOR
		runner.confirm_step(5.25)
		assert runner.state == ExperimentState.WAITING_FOR_VISION_STABILITY
		application.state_updated.emit(_vision(active=True, root=str(dataset_root)))
		assert runner.state == ExperimentState.SETTLING
		runner._tick()
		assert runner.state == ExperimentState.RECORDING
		assert application.batches[0]["known_force_N"] == 5.25
		assert application.batches[0]["force_step_id"] == "loading-r01-s001"
		assert application.batches[0]["sample_interval_ms"] == 25
		assert application.batches[0]["require_motor_telemetry"] is True

		application.experiment_batch_queued.emit(
			"loading-r01-s001", True, "queued"
		)
		application.state_updated.emit(
			_vision(active=True, accepted=1, pending=0, root=str(dataset_root))
		)
		assert runner.state == ExperimentState.REVIEWING_STEP
		runner.accept_review()
		assert runner.state == ExperimentState.COMPLETE
		progress = yaml.safe_load(
			(dataset_root / "collection-progress.yaml").read_text(encoding="utf-8")
		)
		assert progress["completed_step_ids"] == ["loading-r01-s001"]
	finally:
		runner.shutdown()


def test_abort_stops_pending_collection_without_erasing_rows(tmp_path: Path) -> None:
	protocol_path = tmp_path / "protocol.yaml"
	protocol_path.write_text(yaml.safe_dump(_mapping(steps=[{
		"force_N": 3.0, "direction": "loading", "repetition": 1,
	}])), encoding="utf-8")
	application = _ApplicationController()
	motor = _MotorController()
	runner = ExperimentController(application, motor)
	try:
		runner.load_protocol(protocol_path)
		runner._vision = _vision(active=True, root=str(tmp_path))
		runner._motor = _motor()
		runner._set_state(ExperimentState.RECORDING, "recording")
		runner.abort()
		assert runner.state == ExperimentState.ABORTED
		assert application.aborts == 1
		assert ("stop",) in motor.actions
		assert "dataset_stop" in application.commands
	finally:
		runner.shutdown()


def test_confirmed_repeat_opens_to_two_radians_and_reapplies_prior_torque(
	tmp_path: Path,
) -> None:
	protocol_path = tmp_path / "protocol.yaml"
	protocol_path.write_text(yaml.safe_dump(_mapping(steps=[
		{"force_N": 4.0, "direction": "loading", "repetition": 1},
		{"force_N": 4.0, "direction": "loading", "repetition": 2},
	])), encoding="utf-8")
	application = _ApplicationController()
	motor = _MotorController()
	runner = ExperimentController(application, motor)
	try:
		runner.load_protocol(protocol_path)
		runner._motor = _motor()
		runner._confirmed_force_N = 4.0
		runner._captured_torque_Nm = -2.0
		runner._last_step_accepted = 1
		runner._set_state(ExperimentState.REVIEWING_STEP, "review")
		runner.accept_review()
		assert runner.state == ExperimentState.WAITING_FOR_OPERATOR
		assert runner._pending_action == "repeat_reset"

		runner.confirm_step(4.0)
		assert runner.state == ExperimentState.MOVING
		assert motor.actions[-1] == ("stop",)
		motor.state_updated.emit(_motor(state="IDLE", enabled=False))
		assert motor.actions[-1] == ("enable",)
		motor.state_updated.emit(_motor(state="IDLE", enabled=True, position_rad=1.0))
		assert motor.actions[-1] == ("open",)
		motor.state_updated.emit(_motor(state="IDLE", enabled=True, position_rad=2.0))
		assert motor.actions[-1] == ("torque", -2.0)
		motor.state_updated.emit(_motor())
		assert runner.state == ExperimentState.WAITING_FOR_OPERATOR
		assert "reapplied" in runner._message.lower()
	finally:
		runner.shutdown()
