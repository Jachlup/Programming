"""Operator-guided collection coordinator for detached vision and motor state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
import yaml

from collection_protocol import (
	CollectionProtocol,
	CollectionStep,
	OperatorForceSource,
	load_collection_protocol,
)


class ExperimentState(str, Enum):
	PREFLIGHT = "PREFLIGHT"
	WAITING_FOR_OPERATOR = "WAITING_FOR_OPERATOR"
	MOVING = "MOVING"
	WAITING_FOR_MOTOR_STABILITY = "WAITING_FOR_MOTOR_STABILITY"
	WAITING_FOR_VISION_STABILITY = "WAITING_FOR_VISION_STABILITY"
	SETTLING = "SETTLING"
	RECORDING = "RECORDING"
	REVIEWING_STEP = "REVIEWING_STEP"
	NEXT_STEP = "NEXT_STEP"
	VALIDATING_DATASET = "VALIDATING_DATASET"
	TRAINING = "TRAINING"
	MODEL_REVIEW = "MODEL_REVIEW"
	COMPLETE = "COMPLETE"
	ABORTED = "ABORTED"
	FAULT = "FAULT"


@dataclass(frozen=True)
class StabilityResult:
	stable: bool
	elapsed_s: float
	reason: str


class StabilityMonitor:
	"""Require continuous valid, low-motion vision and motor observations."""

	def __init__(self, protocol: CollectionProtocol) -> None:
		self.protocol = protocol
		self._samples: deque[tuple[float, np.ndarray, float | None]] = deque()
		self._last_frame_number: int | None = None

	def reset(self) -> None:
		self._samples.clear()
		self._last_frame_number = None

	def _invalid(self, reason: str) -> StabilityResult:
		self.reset()
		return StabilityResult(False, 0.0, reason)

	def update(
		self,
		vision: dict[str, Any],
		motor: dict[str, Any],
		*,
		now: float | None = None,
	) -> StabilityResult:
		now_value = time.monotonic() if now is None else float(now)
		for key, message in (
			("tracking_valid", "Tracking is invalid"),
			("origin_valid", "Reference origin is invalid"),
			("geometry_valid", "Geometry is invalid"),
			("blue_tracking_valid", "Blue target is invalid"),
		):
			if not bool(vision.get(key)):
				return self._invalid(message)
		quality_checks = (
			("tracking_quality", self.protocol.minimum_tracking_quality, "tracking"),
			("origin_tracking_quality", self.protocol.minimum_origin_quality, "origin"),
			("geometry_quality", self.protocol.minimum_geometry_quality, "geometry"),
		)
		for key, threshold, label in quality_checks:
			try:
				quality = float(vision.get(key, 0.0))
			except (TypeError, ValueError, OverflowError):
				return self._invalid(f"{label.capitalize()} quality is invalid")
			if not math.isfinite(quality) or quality < threshold:
				return self._invalid(
					f"{label.capitalize()} quality {quality:g} is below {threshold:g}"
				)

		features = np.asarray(vision.get("feature_values", ()), dtype=np.float64)
		if features.ndim != 1 or features.size == 0 or not np.all(np.isfinite(features)):
			return self._invalid("Feature vector is unavailable or invalid")
		torque: float | None = None
		if self.protocol.use_motor:
			if not all(bool(motor.get(key)) for key in ("connected", "enabled", "homed")):
				return self._invalid("Motor is not connected, enabled, and homed")
			try:
				velocity = float(motor["velocity_rad_s"])
				torque = float(motor["measured_torque_Nm"])
				feedback_time = float(motor["feedback_monotonic"])
			except (KeyError, TypeError, ValueError, OverflowError):
				return self._invalid("Motor feedback is unavailable")
			if not all(math.isfinite(value) for value in (velocity, torque, feedback_time)):
				return self._invalid("Motor feedback is not finite")
			age = now_value - feedback_time
			if age < 0.0 or age > self.protocol.maximum_motor_telemetry_age_s:
				return self._invalid("Motor feedback is stale")
			if abs(velocity) > self.protocol.maximum_motor_velocity_rad_s:
				return self._invalid(
					f"Motor velocity {velocity:g} rad/s is above the stability limit"
				)
			if not bool(motor.get("torque_within_tolerance")):
				return self._invalid("Measured torque is outside the confirmed tolerance")

		frame_number = int(vision.get("frame_number", -1))
		if frame_number == self._last_frame_number:
			return self.result(now=now_value)
		self._last_frame_number = frame_number
		if self._samples and self._samples[-1][1].shape != features.shape:
			return self._invalid("Feature schema changed during the stability window")
		self._samples.append((now_value, features.copy(), torque))
		duration = self.protocol.required_stability_duration_s
		while (
			len(self._samples) > 1
			and now_value - self._samples[1][0] >= duration
		):
			self._samples.popleft()
		return self.result(now=now_value)

	def result(self, *, now: float | None = None) -> StabilityResult:
		if not self._samples:
			return StabilityResult(False, 0.0, "Waiting for a valid frame")
		now_value = time.monotonic() if now is None else float(now)
		matrix = np.vstack([sample[1] for sample in self._samples])
		movement = float(np.max(np.ptp(matrix, axis=0))) if len(matrix) > 1 else 0.0
		if movement > self.protocol.maximum_feature_movement:
			self.reset()
			return StabilityResult(
				False,
				0.0,
				f"Feature movement {movement:g} exceeds "
				f"{self.protocol.maximum_feature_movement:g}",
			)
		if self.protocol.use_motor:
			torques = [sample[2] for sample in self._samples]
			if any(value is None for value in torques):
				return self._invalid("Motor torque history is incomplete")
			torque_variation = max(torques) - min(torques)
			if torque_variation > self.protocol.maximum_torque_variation_Nm:
				self.reset()
				return StabilityResult(
					False,
					0.0,
					f"Torque variation {torque_variation:g} Nm exceeds "
					f"{self.protocol.maximum_torque_variation_Nm:g} Nm",
				)
		elapsed = max(0.0, now_value - self._samples[0][0])
		stable = elapsed + 1e-9 >= self.protocol.required_stability_duration_s
		return StabilityResult(
			stable,
			elapsed,
			"Stable" if stable else "Accumulating consecutive stable frames",
		)


class ExperimentController(QObject):
	"""Coordinate state snapshots; never own camera or motor hardware objects."""

	state_updated = Signal(object)
	protocol_loaded = Signal(object)
	log_emitted = Signal(str, str)

	def __init__(self, application_controller, motor_controller, parent=None) -> None:
		super().__init__(parent)
		self.application_controller = application_controller
		self.motor_controller = motor_controller
		self.protocol: CollectionProtocol | None = None
		self.force_source = OperatorForceSource()
		self.state = ExperimentState.PREFLIGHT
		self._vision: dict[str, Any] = {}
		self._motor: dict[str, Any] = {}
		self._camera_running = False
		self._camera_paused = False
		self._step_index = 0
		self._confirmed_force_N: float | None = None
		self._captured_torque_Nm: float | None = None
		self._accepted_before = 0
		self._recording_started = False
		self._last_step_accepted = 0
		self._completed_step_ids: list[str] = []
		self._skipped_step_ids: list[str] = []
		self._stability: StabilityMonitor | None = None
		self._stability_result = StabilityResult(False, 0.0, "Not started")
		self._settling_started: float | None = None
		self._message = "Load a collection protocol and complete preflight."
		self._awaiting_dataset_start = False
		self._pending_action = ""
		self._motion_phase = ""
		self._paused_state: ExperimentState | None = None
		self._confirmation_override: bool | None = None
		self._dataset_root = ""
		self._timer = QTimer(self)
		self._timer.setInterval(50)
		self._timer.timeout.connect(self._tick)
		self._timer.start()

		application_controller.state_updated.connect(self.update_vision_state)
		application_controller.camera_state_changed.connect(self.update_camera_state)
		application_controller.command_completed.connect(self._command_completed)
		application_controller.experiment_batch_queued.connect(self._batch_queued)
		motor_controller.state_updated.connect(self.update_motor_state)

	@property
	def current_step(self) -> CollectionStep | None:
		if self.protocol is None or not 0 <= self._step_index < len(self.protocol.steps):
			return None
		return self.protocol.steps[self._step_index]

	@property
	def require_operator_confirmation(self) -> bool:
		if self._confirmation_override is not None:
			return self._confirmation_override
		return bool(self.protocol and self.protocol.require_operator_confirmation)

	def _log(self, level: str, message: str) -> None:
		self.log_emitted.emit(level, str(message))

	def _set_state(self, state: ExperimentState, message: str) -> None:
		self.state = state
		self._message = str(message)
		self._emit_state()

	def _emit_state(self) -> None:
		step = self.current_step
		protocol = self.protocol
		self.state_updated.emit({
			"state": self.state.value,
			"message": self._message,
			"protocol_loaded": protocol is not None,
			"experiment_id": "" if protocol is None else protocol.experiment_id,
			"current_step": 0 if step is None else step.index,
			"planned_steps": 0 if protocol is None else len(protocol.steps),
			"planned_samples": 0 if protocol is None else protocol.planned_samples,
			"force_step_id": "" if step is None else step.force_step_id,
			"planned_force_N": None if step is None else step.force_N,
			"confirmed_force_N": self._confirmed_force_N,
			"direction": "" if step is None else step.direction,
			"repetition": 0 if step is None else step.repetition,
			"samples_this_step": 0 if step is None else step.samples,
			"accepted_this_step": self._last_step_accepted,
			"completed_steps": len(self._completed_step_ids),
			"skipped_steps": len(self._skipped_step_ids),
			"stability_elapsed_s": self._stability_result.elapsed_s,
			"stability_reason": self._stability_result.reason,
			"confirmation_every_step": self.require_operator_confirmation,
			"pending_action": self._pending_action,
			"paused": self._paused_state is not None,
			"nearest_red_point_id": self._vision.get("nearest_red_point_to_blue", ""),
			"preflight_errors": self.preflight_errors() if protocol else [],
			"dataset_root": (
				self._dataset_root
				or self._vision.get("dataset_directory", "")
			),
		})

	def load_protocol(self, path: str | Path) -> CollectionProtocol:
		if self.state not in {
			ExperimentState.PREFLIGHT,
			ExperimentState.COMPLETE,
			ExperimentState.ABORTED,
			ExperimentState.FAULT,
		}:
			raise RuntimeError("Cannot replace the protocol during an active experiment")
		protocol = load_collection_protocol(path)
		self.protocol = protocol
		self._stability = StabilityMonitor(protocol)
		self._step_index = 0
		self._completed_step_ids = []
		self._skipped_step_ids = []
		self._confirmed_force_N = None
		self._last_step_accepted = 0
		self._dataset_root = ""
		self._set_state(ExperimentState.PREFLIGHT, "Protocol loaded; review preflight.")
		self.protocol_loaded.emit(protocol)
		self._log("success", f"Loaded collection protocol {protocol.experiment_id}.")
		return protocol

	def set_confirmation_every_step(self, enabled: bool) -> None:
		self._confirmation_override = bool(enabled)
		self._emit_state()

	def preflight_errors(self) -> list[str]:
		if self.protocol is None:
			return ["No collection protocol is loaded"]
		errors: list[str] = []
		if not self._camera_running:
			errors.append("Camera is not running")
		if self._camera_paused:
			errors.append("Camera processing is paused")
		for key, message in (
			("reference_configured", "No accepted reference profile"),
			("tracking_active", "Tracking is not active"),
			("tracking_valid", "Tracking is invalid"),
			("origin_valid", "Reference origin is invalid"),
			("geometry_valid", "Geometry is invalid"),
			("blue_tracking_valid", "Blue target is not valid"),
		):
			if not bool(self._vision.get(key)):
				errors.append(message)
		detected = int(self._vision.get("detected_marker_count", 0))
		expected = int(self._vision.get("expected_marker_count", 0))
		if expected and detected != expected:
			errors.append(f"Detected red marker count is {detected}; expected {expected}")
		if not self._vision.get("feature_names") or not self._vision.get("feature_values"):
			errors.append("Geometry feature schema is unavailable")
		if self._vision.get("dataset_active"):
			errors.append("A dataset session is already active")
		if self.protocol.use_motor:
			if not bool(self._motor.get("connected")):
				errors.append("Motor is not connected")
			if not bool(self._motor.get("enabled")):
				errors.append("Motor drive is not enabled")
			if not bool(self._motor.get("homed")):
				errors.append("Motor has not completed homing")
			if self._motor.get("state") == "FAULT":
				errors.append("Motor is in FAULT")
		return errors

	@Slot()
	def start(self) -> None:
		if self.protocol is None:
			self._set_state(ExperimentState.FAULT, "No collection protocol is loaded.")
			return
		errors = self.preflight_errors()
		if errors:
			self._set_state(ExperimentState.PREFLIGHT, "; ".join(errors))
			return
		self._step_index = 0
		self._completed_step_ids = []
		self._skipped_step_ids = []
		self._awaiting_dataset_start = True
		first_force = self.protocol.steps[0].force_N
		self._set_state(ExperimentState.PREFLIGHT, "Creating a new dataset session…")
		self.application_controller.execute(
			f"dataset_start {self.protocol.experiment_id} {first_force:.12g}"
		)

	@Slot(float)
	def confirm_step(self, operator_force_N: float) -> None:
		if self.state != ExperimentState.WAITING_FOR_OPERATOR:
			self._log("warning", "The experiment is not waiting for operator confirmation.")
			return
		if self._paused_state is not None:
			self._log("warning", "Resume the experiment before confirming a step.")
			return
		if self._pending_action == "repeat_reset":
			self._pending_action = ""
			self._begin_repeat_reset()
			return
		step = self.current_step
		if step is None or self.protocol is None:
			self._fault("No current collection step is available")
			return
		try:
			force = self.force_source.confirm(operator_force_N)
		except ValueError as exc:
			self._log("error", str(exc))
			return
		if self.protocol.use_motor:
			if self._motor.get("state") != "HOLDING_TORQUE":
				self._log("warning", "Apply and hold the selected motor torque first.")
				return
			try:
				target = float(self._motor["target_torque_Nm"])
			except (KeyError, TypeError, ValueError, OverflowError):
				self._log("warning", "The active motor torque target is unavailable.")
				return
			if not math.isfinite(target) or target == 0.0:
				self._log("warning", "A finite nonzero closing torque must be active.")
				return
			if step.motor_torque_Nm is not None and not math.isclose(
				target, step.motor_torque_Nm, abs_tol=1e-9
			):
				self._log(
					"warning",
					f"Protocol requests {step.motor_torque_Nm:g} Nm but the motor target "
					f"is {target:g} Nm.",
				)
				return
			self._captured_torque_Nm = target
		self._confirmed_force_N = force
		assert self._stability is not None
		self._stability.reset()
		self._set_state(
			ExperimentState.WAITING_FOR_MOTOR_STABILITY,
			"Force label confirmed; waiting for motor stability.",
		)
		self._advance_motor_stability()

	def _advance_motor_stability(self) -> None:
		if self.protocol is None:
			return
		if not self.protocol.use_motor:
			self._set_state(
				ExperimentState.WAITING_FOR_VISION_STABILITY,
				"Waiting for consecutive stable vision frames.",
			)
			return
		try:
			velocity = abs(float(self._motor["velocity_rad_s"]))
		except (KeyError, TypeError, ValueError, OverflowError):
			return
		if (
			self._motor.get("state") == "HOLDING_TORQUE"
			and velocity <= self.protocol.maximum_motor_velocity_rad_s
			and bool(self._motor.get("torque_within_tolerance"))
		):
			self._set_state(
				ExperimentState.WAITING_FOR_VISION_STABILITY,
				"Motor is stable; waiting for consecutive stable vision frames.",
			)

	def _queue_current_step(self) -> None:
		step = self.current_step
		protocol = self.protocol
		if step is None or protocol is None or self._confirmed_force_N is None:
			self._fault("Cannot queue an incomplete collection step")
			return
		self._accepted_before = int(self._vision.get("dataset_accepted", 0))
		self._recording_started = False
		self._last_step_accepted = 0
		self._set_state(ExperimentState.RECORDING, "Queuing validated sample frames…")
		self.application_controller.queue_experiment_batch({
			"count": step.samples,
			"known_force_N": self._confirmed_force_N,
			"force_step_id": step.force_step_id,
			"loading_direction": step.direction,
			"repetition": step.repetition,
			"require_blue_target": True,
			"require_motor_telemetry": protocol.use_motor,
			"maximum_motor_telemetry_age_s": (
				protocol.maximum_motor_telemetry_age_s if protocol.use_motor else None
			),
			"sample_interval_ms": protocol.sample_interval_ms,
		})

	@Slot(str, bool, str)
	def _batch_queued(self, step_id: str, ok: bool, message: str) -> None:
		step = self.current_step
		if self.state != ExperimentState.RECORDING or step is None:
			return
		if step_id != step.force_step_id or not ok:
			self._fault(message or "Dataset batch could not be queued")
			return
		self._recording_started = True
		self._message = f"Recording {step.samples} validated sample(s)."
		self._emit_state()

	@Slot(object)
	def update_vision_state(self, snapshot) -> None:
		if not isinstance(snapshot, dict):
			return
		self._vision = dict(snapshot)
		if snapshot.get("dataset_root"):
			self._dataset_root = str(snapshot["dataset_root"])
		if self._awaiting_dataset_start and bool(snapshot.get("dataset_active")):
			self._awaiting_dataset_start = False
			self._confirmed_force_N = None
			self._set_state(
				ExperimentState.WAITING_FOR_OPERATOR,
				"Set motor torque, verify the known force, then confirm and save.",
			)
		elif self.state == ExperimentState.WAITING_FOR_VISION_STABILITY:
			assert self._stability is not None
			self._stability_result = self._stability.update(self._vision, self._motor)
			if self._stability_result.stable:
				self._settling_started = time.monotonic()
				self._set_state(
					ExperimentState.SETTLING,
					"Stability window passed; settling before recording.",
				)
			else:
				self._message = self._stability_result.reason
				self._emit_state()
		elif self.state == ExperimentState.SETTLING:
			assert self._stability is not None
			result = self._stability.update(self._vision, self._motor)
			if not result.stable:
				self._stability_result = result
				self._set_state(
					ExperimentState.WAITING_FOR_VISION_STABILITY,
					f"Stability was lost during settling: {result.reason}",
				)
		elif self.state == ExperimentState.RECORDING and self._recording_started:
			pending = int(snapshot.get("dataset_pending", 0))
			if pending == 0:
				accepted = int(snapshot.get("dataset_accepted", 0)) - self._accepted_before
				self._last_step_accepted = max(0, accepted)
				step = self.current_step
				assert step is not None
				if self._last_step_accepted >= step.samples:
					self._set_state(
						ExperimentState.REVIEWING_STEP,
						f"Saved {self._last_step_accepted} sample(s); review the step.",
					)
					if not self.require_operator_confirmation:
						self.accept_review()
				else:
					self._set_state(
						ExperimentState.REVIEWING_STEP,
						f"Only {self._last_step_accepted}/{step.samples} samples were accepted; "
						"retry or skip this step.",
					)
		else:
			self._emit_state()

	@Slot(bool, bool)
	def update_camera_state(self, running: bool, paused: bool) -> None:
		self._camera_running = bool(running)
		self._camera_paused = bool(paused)
		self._emit_state()

	@Slot(object)
	def update_motor_state(self, snapshot) -> None:
		if not isinstance(snapshot, dict):
			return
		self._motor = dict(snapshot)
		if self.state == ExperimentState.WAITING_FOR_MOTOR_STABILITY:
			self._advance_motor_stability()
		elif self.state == ExperimentState.MOVING:
			self._advance_repeat_motion()
		elif snapshot.get("state") == "FAULT" and self.state not in {
			ExperimentState.PREFLIGHT,
			ExperimentState.COMPLETE,
			ExperimentState.ABORTED,
			ExperimentState.FAULT,
		}:
			self._fault("Motor entered FAULT during collection")
		else:
			self._emit_state()

	@Slot(str, bool)
	def _command_completed(self, command: str, ok: bool) -> None:
		if self._awaiting_dataset_start and command.startswith("dataset_start") and not ok:
			self._awaiting_dataset_start = False
			self._fault("Dataset session could not be started")

	@Slot()
	def accept_review(self) -> None:
		if self.state != ExperimentState.REVIEWING_STEP:
			return
		step = self.current_step
		if step is None:
			return
		if self._last_step_accepted < step.samples:
			self._log("warning", "Retry or skip; the planned samples were not accepted.")
			return
		self._completed_step_ids.append(step.force_step_id)
		self._write_progress()
		self._advance_step()

	@Slot()
	def retry_step(self) -> None:
		if self.state != ExperimentState.REVIEWING_STEP:
			return
		self._confirmed_force_N = None
		self._set_state(
			ExperimentState.WAITING_FOR_OPERATOR,
			"Retry: verify force and torque, then confirm and save again.",
		)

	@Slot()
	def skip_step(self) -> None:
		if self.state not in {
			ExperimentState.WAITING_FOR_OPERATOR,
			ExperimentState.REVIEWING_STEP,
		}:
			return
		step = self.current_step
		if step is None:
			return
		self._skipped_step_ids.append(step.force_step_id)
		self._write_progress()
		self._advance_step()

	def _advance_step(self) -> None:
		previous_force = self._confirmed_force_N
		previous_torque = self._captured_torque_Nm
		self._set_state(ExperimentState.NEXT_STEP, "Advancing to the next protocol step.")
		self._step_index += 1
		self._confirmed_force_N = None
		self._last_step_accepted = 0
		if self.protocol is None or self._step_index >= len(self.protocol.steps):
			self._set_state(
				ExperimentState.VALIDATING_DATASET,
				"Collection finished; dataset audit is ready.",
			)
			self.application_controller.execute("dataset_status")
			self._set_state(ExperimentState.COMPLETE, "Collection protocol completed.")
			self._write_progress()
			self.application_controller.execute("dataset_stop")
			return
		next_step = self.current_step
		assert next_step is not None
		if (
			self.protocol.use_motor
			and previous_torque is not None
			and previous_force is not None
			and math.isclose(next_step.force_N, previous_force, abs_tol=1e-12)
		):
			self._captured_torque_Nm = previous_torque
			if self.require_operator_confirmation:
				self._pending_action = "repeat_reset"
				self._set_state(
					ExperimentState.WAITING_FOR_OPERATOR,
					"Confirm opening to 2.0 rad and reapplying the prior torque.",
				)
			else:
				self._begin_repeat_reset()
			return
		self._set_state(
			ExperimentState.WAITING_FOR_OPERATOR,
			"Set the next torque, verify force, then confirm and save.",
		)

	def _begin_repeat_reset(self) -> None:
		if self._captured_torque_Nm is None:
			self._fault("Repeat reset has no captured torque")
			return
		self._motion_phase = "stopping"
		self._set_state(
			ExperimentState.MOVING,
			"Stopping torque before reopening the gripper.",
		)
		self.motor_controller.stop_motor()

	def _advance_repeat_motion(self) -> None:
		state = self._motor.get("state")
		if state == "FAULT":
			self._fault("Motor fault during repeat reset")
			return
		if self._motion_phase == "stopping" and state == "IDLE" and not self._motor.get("enabled"):
			self._motion_phase = "enabling"
			self.motor_controller.enable_drive()
		elif self._motion_phase == "enabling" and state == "IDLE" and self._motor.get("enabled"):
			self._motion_phase = "opening"
			self._message = "Opening to the confirmed 2.0 rad position."
			self._emit_state()
			self.motor_controller.open_gripper()
		elif self._motion_phase == "opening" and state == "IDLE" and self._motor.get("enabled"):
			try:
				position = float(self._motor["position_rad"])
			except (KeyError, TypeError, ValueError, OverflowError):
				return
			if abs(position - self.motor_controller.config.open_position_rad) <= (
				self.motor_controller.config.position_tolerance_rad
			):
				self._motion_phase = "reapplying"
				self._message = f"Reapplying {self._captured_torque_Nm:g} Nm."
				self._emit_state()
				self.motor_controller.hold_torque(self._captured_torque_Nm)
		elif self._motion_phase == "reapplying" and state == "HOLDING_TORQUE":
			if bool(self._motor.get("torque_within_tolerance")):
				self._motion_phase = ""
				self._set_state(
					ExperimentState.WAITING_FOR_OPERATOR,
					"Prior torque reapplied; verify force, then confirm and save.",
				)

	@Slot()
	def pause(self) -> None:
		if self._paused_state is not None:
			resume_state = self._paused_state
			self._paused_state = None
			if resume_state == ExperimentState.SETTLING:
				assert self._stability is not None
				self._stability.reset()
				resume_state = ExperimentState.WAITING_FOR_VISION_STABILITY
			self._set_state(resume_state, "Experiment resumed; stability timing restarted.")
			return
		if self.state in {
			ExperimentState.PREFLIGHT,
			ExperimentState.COMPLETE,
			ExperimentState.ABORTED,
			ExperimentState.FAULT,
		}:
			return
		if self.state in {ExperimentState.MOVING, ExperimentState.RECORDING}:
			self._log(
				"warning",
				"Cannot pause active motor motion or an immutable recording batch; "
				"use Abort/STOP if necessary.",
			)
			return
		self._paused_state = self.state
		self._set_state(
			ExperimentState.WAITING_FOR_OPERATOR,
			"Experiment paused; press Resume to continue.",
		)

	@Slot()
	def abort(self) -> None:
		if self.state in {ExperimentState.COMPLETE, ExperimentState.ABORTED}:
			return
		self.application_controller.abort_experiment_batch()
		if bool(self._motor.get("connected")):
			self.motor_controller.stop_motor()
		self._set_state(
			ExperimentState.ABORTED,
			"Experiment aborted; accepted dataset rows were preserved.",
		)
		self._write_progress()
		if bool(self._vision.get("dataset_active")):
			self.application_controller.execute("dataset_stop")

	def _fault(self, message: str) -> None:
		self._log("error", message)
		self._set_state(ExperimentState.FAULT, message)

	def _tick(self) -> None:
		if self.state != ExperimentState.SETTLING or self.protocol is None:
			return
		if self._settling_started is None:
			self._settling_started = time.monotonic()
		if time.monotonic() - self._settling_started >= self.protocol.settling_time_s:
			self._settling_started = None
			self._queue_current_step()

	def _write_progress(self) -> None:
		root_value = self._dataset_root or self._vision.get("dataset_root")
		if not root_value or self.protocol is None:
			return
		root = Path(str(root_value))
		if not root.is_dir():
			return
		data = {
			"experiment_id": self.protocol.experiment_id,
			"state": self.state.value,
			"updated_at_unix_s": time.time(),
			"completed_step_ids": list(self._completed_step_ids),
			"skipped_step_ids": list(self._skipped_step_ids),
			"current_step_index": self._step_index,
			"planned_step_ids": [step.force_step_id for step in self.protocol.steps],
		}
		temporary_name = ""
		try:
			with tempfile.NamedTemporaryFile(
				"w", encoding="utf-8", dir=root, prefix=".progress-", delete=False
			) as stream:
				temporary_name = stream.name
				yaml.safe_dump(data, stream, sort_keys=False)
			os.replace(temporary_name, root / "collection-progress.yaml")
		except OSError as exc:
			self._log("warning", f"Could not save collection progress: {exc}")
			if temporary_name:
				try:
					Path(temporary_name).unlink(missing_ok=True)
				except OSError:
					pass

	def shutdown(self) -> None:
		self._timer.stop()
