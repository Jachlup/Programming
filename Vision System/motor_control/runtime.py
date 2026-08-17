"""Qt worker/controller pair implementing the non-blocking motor state machine."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
import math
from pathlib import Path
import time
import traceback
from typing import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from .backend import CppMotorBackend, MockMotorBackend, MotorBackend
from .config import (
	MotorConfig,
	default_motor_config_path,
	load_motor_config,
	save_motor_config,
)


class MotorState(str, Enum):
	DISCONNECTED = "DISCONNECTED"
	CONNECTING = "CONNECTING"
	IDLE = "IDLE"
	HOMING = "HOMING"
	MOVING = "MOVING"
	HOLDING_TORQUE = "HOLDING_TORQUE"
	STOPPING = "STOPPING"
	FAULT = "FAULT"


BackendFactory = Callable[[MotorConfig], MotorBackend]


def _default_backend_factory(config: MotorConfig) -> MotorBackend:
	if config.backend == "hardware":
		return CppMotorBackend(
			config.can_id,
			current_limit_A=config.current_limit_A,
			maximum_controller_torque_Nm=config.maximum_controller_torque_Nm,
			minimum_position_rad=config.minimum_position_rad,
			maximum_position_rad=config.maximum_position_rad,
			maximum_velocity_rad_s=config.profile_velocity_rad_s,
			maximum_acceleration_rad_s2=config.profile_acceleration_rad_s2,
			closing_direction=config.closing_direction,
			maximum_closing_torque_Nm=config.maximum_closing_torque_Nm,
			maximum_temperature_C=config.maximum_temperature_C,
		)
	return MockMotorBackend(
		minimum_position_rad=config.minimum_position_rad,
		maximum_position_rad=config.maximum_position_rad,
		initial_position_rad=config.open_position_rad,
	)


class MotorWorker(QObject):
	"""The sole owner and caller of the motor backend and any hardware objects."""

	state_updated = Signal(object)
	log_emitted = Signal(str, str)
	command_completed = Signal(str, bool, str)
	debug_events_emitted = Signal(object)
	shutdown_complete = Signal()

	def __init__(
		self,
		config: MotorConfig | None = None,
		backend_factory: BackendFactory | None = None,
		*,
		config_persist_path: str | Path | None = None,
	) -> None:
		super().__init__()
		self.config = config or load_motor_config()
		self._config_persist_path = (
			Path(config_persist_path) if config_persist_path is not None else None
		)
		self._backend_factory = backend_factory or _default_backend_factory
		self._backend: MotorBackend | None = None
		self._timer = QTimer(self)
		self._timer.setInterval(int(self.config.feedback_interval_ms))
		self._timer.timeout.connect(self._tick)
		self._state = MotorState.DISCONNECTED
		self._connected = False
		self._enabled = False
		self._homed = False
		self._shutdown_requested = False
		self._position_rad: float | None = None
		self._velocity_rad_s: float | None = None
		self._measured_torque_Nm: float | None = None
		self._temperature_C: float | None = None
		self._commanded_torque_Nm = 0.0
		self._target_torque_Nm = 0.0
		self._target_position_rad: float | None = None
		self._motion_started: float | None = None
		self._motion_kind = ""
		self._last_tick = time.monotonic()
		self._last_position: float | None = None
		self._stall_started: float | None = None
		self._pending_torque_after_move: float | None = None
		self._latest_warning = ""
		self._latest_error = ""
		self._feedback_monotonic: float | None = None
		self._debug_events: list[dict] = []
		self._debug_sequence = 0
		self._last_debug_state = ""
		self._torque_confirmed = False
		self._pending_torque_command = ""
		self._torque_target_sent_monotonic: float | None = None

	@property
	def state(self) -> MotorState:
		return self._state

	def _log(self, level: str, message: str) -> None:
		self._trace(f"LOG/{level.upper()}", message)
		self.log_emitted.emit(level, str(message))

	def _trace(self, category: str, message: str) -> None:
		self._debug_sequence += 1
		self._debug_events.append({
			"sequence": self._debug_sequence,
			"wall_time": time.time(),
			"monotonic": time.monotonic(),
			"source": "MotorWorker",
			"category": str(category),
			"message": str(message),
		})
		del self._debug_events[:-2000]

	def _collect_backend_debug(self, backend: MotorBackend | None = None) -> None:
		target = backend if backend is not None else self._backend
		drain = getattr(target, "drain_debug_events", None)
		if callable(drain):
			try:
				self._debug_events.extend(drain())
			except Exception as exc:
				self._trace("DEBUG", f"Could not drain backend trace: {exc}")

	def _flush_debug_events(self) -> None:
		self._collect_backend_debug()
		if not self._debug_events:
			return
		events = sorted(
			self._debug_events,
			key=lambda event: (
				float(event.get("monotonic", 0.0)),
				int(event.get("sequence", 0)),
			),
		)
		self._debug_events = []
		self.debug_events_emitted.emit(events)

	def _snapshot(self) -> dict:
		now = time.monotonic()
		command_at_target = abs(
			self._commanded_torque_Nm - self._target_torque_Nm
		) <= 1e-6
		return {
			"backend_type": self.config.backend,
			"state": self._state.value,
			"connected": self._connected,
			"enabled": self._enabled,
			"homed": self._homed,
			"position_rad": self._position_rad,
			"velocity_rad_s": self._velocity_rad_s,
			"commanded_torque_Nm": self._commanded_torque_Nm,
			"target_torque_Nm": self._target_torque_Nm,
			"measured_torque_Nm": self._measured_torque_Nm,
			"temperature_C": self._temperature_C,
			"target_position_rad": self._target_position_rad,
			"motion_elapsed_s": (
				0.0 if self._motion_started is None
				else max(0.0, now - self._motion_started)
			),
			"torque_command_at_target": command_at_target,
			"torque_within_tolerance": bool(
				self._measured_torque_Nm is not None
				and command_at_target
				and abs(self._measured_torque_Nm - self._target_torque_Nm)
				<= self.config.torque_tolerance_Nm
			),
			"torque_confirmed": self._torque_confirmed,
			"feedback_monotonic": self._feedback_monotonic,
			"latest_warning": self._latest_warning,
			"latest_error": self._latest_error,
			"configuration": self.config.to_mapping(),
		}

	def _emit_state(self) -> None:
		if self._state.value != self._last_debug_state:
			self._trace(
				"STATE",
				f"{self._last_debug_state or 'INITIAL'} -> {self._state.value}",
			)
			self._last_debug_state = self._state.value
		self._flush_debug_events()
		self.state_updated.emit(self._snapshot())

	def _complete(self, command: str, ok: bool, message: str) -> None:
		self._trace(
			"RESULT",
			f"{command}: {'CONFIRMED' if ok else 'FAILED'}: {message}",
		)
		self.command_completed.emit(command, ok, message)
		self._emit_state()

	def _acknowledge(self, command: str, message: str) -> None:
		self._trace("ACCEPTED", f"{command}: {message}")
		self._emit_state()

	def _reject(self, command: str, exc: BaseException) -> None:
		message = str(exc)
		self._latest_warning = message
		self._log("warning", f"Motor {command} rejected: {message}")
		self._complete(command, False, message)

	def _require_backend(self) -> MotorBackend:
		if not self._connected or self._backend is None:
			raise RuntimeError("Motor is not connected")
		return self._backend

	def _require_idle_enabled(self) -> MotorBackend:
		backend = self._require_backend()
		if not self._enabled:
			raise RuntimeError("Enable the drive first")
		if self._state != MotorState.IDLE:
			raise RuntimeError(f"Motor must be IDLE, not {self._state.value}")
		return backend

	@Slot()
	def initialize(self) -> None:
		self._trace(
			"INITIALIZE",
			f"backend={self.config.backend}, CAN ID={self.config.can_id}, "
			f"feedback interval={self.config.feedback_interval_ms} ms",
		)
		if not self._timer.isActive():
			self._last_tick = time.monotonic()
			self._timer.start()
		self._emit_state()

	@Slot(float, float, float, float, float, float)
	def configure_ranges(
		self,
		minimum_position_rad: float,
		maximum_position_rad: float,
		open_position_rad: float,
		minimum_closing_torque_Nm: float,
		maximum_velocity_rad_s: float,
		maximum_acceleration_rad_s2: float,
	) -> None:
		command = "configure_ranges"
		self._trace(
			"REQUEST",
			f"configure position range={minimum_position_rad:g}.."
			f"{maximum_position_rad:g} rad, "
			f"open target={open_position_rad:g} rad, "
			f"torque range={minimum_closing_torque_Nm:g}..0 Nm, "
			f"maximum velocity={maximum_velocity_rad_s:g} rad/s, "
			f"maximum acceleration={maximum_acceleration_rad_s2:g} rad/s²",
		)
		try:
			if self._connected:
				raise RuntimeError("Disconnect the motor before changing configured ranges")
			minimum_torque = float(minimum_closing_torque_Nm)
			if not math.isfinite(minimum_torque) or minimum_torque >= 0.0:
				raise ValueError("Minimum closing torque must be a finite negative value")
			updated = replace(
				self.config,
				minimum_position_rad=float(minimum_position_rad),
				maximum_position_rad=float(maximum_position_rad),
				open_position_rad=float(open_position_rad),
				maximum_closing_torque_Nm=abs(minimum_torque),
				profile_velocity_rad_s=float(maximum_velocity_rad_s),
				profile_acceleration_rad_s2=float(maximum_acceleration_rad_s2),
			)
			if self._config_persist_path is not None:
				save_motor_config(updated, self._config_persist_path)
			self.config = updated
			self._position_rad = None
			self._velocity_rad_s = None
			self._measured_torque_Nm = None
			self._temperature_C = None
			self._feedback_monotonic = None
			self._latest_warning = ""
			self._latest_error = ""
			self._state = MotorState.DISCONNECTED
			message = (
				f"Configured position range {updated.minimum_position_rad:g}.."
				f"{updated.maximum_position_rad:g} rad "
				f"(open target {updated.open_position_rad:g} rad) and closing torque "
				f"range -{updated.maximum_closing_torque_Nm:g}..0 Nm, maximum velocity "
				f"{updated.profile_velocity_rad_s:g} rad/s, and maximum acceleration "
				f"{updated.profile_acceleration_rad_s2:g} rad/s²."
			)
			if self._config_persist_path is not None:
				message += f" Saved to {self._config_persist_path}."
			self._log("success", message)
			self._complete(command, True, message)
		except Exception as exc:
			self._reject(command, exc)

	@Slot()
	def connect_motor(self) -> None:
		command = "connect"
		self._trace("REQUEST", f"connect CAN ID {self.config.can_id}")
		try:
			if self._connected:
				raise RuntimeError("Motor is already connected")
			self._latest_error = ""
			self._latest_warning = ""
			self._state = MotorState.CONNECTING
			self._emit_state()
			backend = self._backend_factory(self.config)
			self._backend = backend
			backend.connect()
			self._connected = True
			backend.apply_current_limit(self.config.current_limit_A)
			# The SDK names this a maximum and the confirmed range is -10..+10,
			# therefore the controller receives the positive magnitude.
			backend.apply_maximum_torque(self.config.maximum_controller_torque_Nm)
			backend.stop_idle()
			self._enabled = False
			self._homed = False
			self._state = MotorState.IDLE
			self._poll_feedback()
			message = (
				f"{backend.backend_type.capitalize()} motor connected at CAN ID "
				f"{self.config.can_id}; drive remains disabled."
			)
			self._log("success", message)
			self._complete(command, True, message)
		except Exception as exc:
			backend = self._backend
			if backend is not None:
				try:
					backend.disconnect()
				except Exception as cleanup_exc:
					self._log("warning", f"Connection cleanup also failed: {cleanup_exc}")
			self._collect_backend_debug(backend)
			self._backend = None
			self._connected = False
			self._enabled = False
			self._state = MotorState.FAULT
			self._report_exception("Motor connection failed", exc)
			self._complete(command, False, str(exc))

	@Slot()
	def disconnect_motor(self) -> None:
		command = "disconnect"
		self._trace("REQUEST", "disconnect")
		if not self._connected:
			self._state = MotorState.DISCONNECTED
			self._backend = None
			self._complete(command, True, "Motor is already disconnected.")
			return
		try:
			self._safe_stop(disable=True)
			assert self._backend is not None
			backend = self._backend
			backend.disconnect()
			self._collect_backend_debug(backend)
			self._backend = None
			self._connected = False
			self._enabled = False
			self._homed = False
			self._clear_motion()
			self._state = MotorState.DISCONNECTED
			self._log("success", "Motor disconnected.")
			self._complete(command, True, "Motor disconnected.")
		except Exception as exc:
			self._enter_fault("Motor disconnect failed", exc, command)

	@Slot()
	def enable_drive(self) -> None:
		command = "enable"
		self._trace("REQUEST", "enable drive")
		try:
			backend = self._require_backend()
			if self._state not in {MotorState.IDLE, MotorState.FAULT}:
				raise RuntimeError("Drive can only be enabled while idle")
			if self._enabled:
				self._complete(command, True, "Drive is already enabled.")
				return
			backend.enable()
			self._enabled = True
			self._state = MotorState.IDLE
			self._latest_error = ""
			self._log("success", "Motor drive enabled.")
			self._complete(command, True, "Motor drive enabled.")
		except Exception as exc:
			self._reject(command, exc)

	@Slot()
	def disable_drive(self) -> None:
		command = "disable"
		self._trace("REQUEST", "disable drive")
		try:
			backend = self._require_backend()
			if self._state not in {MotorState.IDLE, MotorState.FAULT}:
				self._safe_stop(disable=True)
			elif self._enabled:
				backend.stop_idle()
				backend.disable()
			self._enabled = False
			self._state = MotorState.IDLE
			self._complete(command, True, "Motor drive disabled.")
		except Exception as exc:
			self._enter_fault("Drive disable failed", exc, command)

	@Slot()
	def clear_faults(self) -> None:
		command = "clear_faults"
		self._trace("REQUEST", "clear controller faults")
		try:
			backend = self._require_backend()
			if self._enabled:
				raise RuntimeError("Disable the drive before clearing faults")
			backend.clear_faults()
			self._state = MotorState.IDLE
			self._latest_error = ""
			self._complete(command, True, "Motor faults cleared.")
		except Exception as exc:
			self._reject(command, exc)

	@Slot()
	def start_homing(self) -> None:
		command = "home"
		self._trace("REQUEST", f"home at {self.config.homing_torque_Nm:g} Nm")
		try:
			backend = self._require_idle_enabled()
			torque = self.config.validate_closing_torque(self.config.homing_torque_Nm)
			self._poll_feedback()
			backend.start_constant_torque(torque)
			self._commanded_torque_Nm = torque
			self._target_torque_Nm = torque
			self._motion_started = time.monotonic()
			self._motion_kind = "home"
			self._last_position = self._position_rad
			self._stall_started = None
			self._state = MotorState.HOMING
			self._log("info", f"Homing started at {torque:g} Nm.")
			self._acknowledge(command, "Homing command accepted; completion pending.")
		except Exception as exc:
			self._reject(command, exc)

	@Slot(float, float, float)
	def move_to_position(self, position_rad: float, velocity_rad_s: float, acceleration_rad_s2: float) -> None:
		command = "position"
		self._trace(
			"REQUEST",
			f"position={position_rad:g} rad, velocity={velocity_rad_s:g} rad/s, "
			f"acceleration={acceleration_rad_s2:g} rad/s²",
		)
		try:
			backend = self._require_idle_enabled()
			if self.config.require_homing and not self._homed:
				raise RuntimeError("Successful homing is required before position movement")
			position = self.config.validate_position(position_rad)
			velocity, acceleration = self.config.validate_profile(velocity_rad_s, acceleration_rad_s2)
			self._begin_position_move(backend, position, velocity, acceleration, "position")
			self._acknowledge(
				command,
				f"Position command accepted for {position:g} rad; completion pending.",
			)
		except Exception as exc:
			self._reject(command, exc)

	@Slot(float)
	def start_torque_hold(self, torque_Nm: float) -> None:
		command = "torque"
		self._trace("REQUEST", f"start closing torque target={torque_Nm:g} Nm")
		try:
			backend = self._require_idle_enabled()
			if self.config.require_homing and not self._homed:
				raise RuntimeError("Successful homing is required before applying torque")
			target = self.config.validate_closing_torque(torque_Nm)
			backend.start_constant_torque(0.0)
			self._target_torque_Nm = target
			self._commanded_torque_Nm = 0.0
			self._motion_started = time.monotonic()
			self._motion_kind = "torque"
			self._last_tick = self._motion_started
			self._torque_confirmed = False
			self._pending_torque_command = command
			self._torque_target_sent_monotonic = None
			self._state = MotorState.HOLDING_TORQUE
			self._log("info", f"Closing torque ramp started toward {target:g} Nm.")
			self._acknowledge(
				command,
				"Torque command accepted; ramp and measured-feedback confirmation pending.",
			)
		except Exception as exc:
			self._reject(command, exc)

	@Slot(float)
	def update_target_torque(self, torque_Nm: float) -> None:
		command = "update_torque"
		self._trace("REQUEST", f"update closing torque target={torque_Nm:g} Nm")
		try:
			backend = self._require_backend()
			if self._state != MotorState.HOLDING_TORQUE:
				raise RuntimeError("Torque can only be adjusted while holding torque")
			target = self.config.validate_closing_torque(torque_Nm)
			if abs(target) < abs(self._target_torque_Nm):
				self._poll_feedback()
				assert self._position_rad is not None
				backoff = min(
					self._position_rad + self.config.torque_reduction_backoff_rad,
					self.config.maximum_position_rad,
				)
				if backoff <= self._position_rad:
					raise RuntimeError("No safe opening travel remains for torque reduction")
				backend.release_torque()
				backend.stop_idle()
				self._commanded_torque_Nm = 0.0
				self._target_torque_Nm = target
				self._pending_torque_after_move = target
				self._begin_position_move(
					backend,
					backoff,
					self.config.profile_velocity_rad_s,
					self.config.profile_acceleration_rad_s2,
					"torque_backoff",
				)
				message = f"Opening to {backoff:g} rad before reapplying {target:g} Nm."
			else:
				self._target_torque_Nm = target
				message = f"Torque target updated to {target:g} Nm."
			self._torque_confirmed = False
			self._pending_torque_command = command
			self._torque_target_sent_monotonic = None
			self._log("info", message)
			self._acknowledge(
				command,
				message + " Measured-feedback confirmation pending.",
			)
		except Exception as exc:
			self._reject(command, exc)

	def _torque_for_finger_force(self, force_N: float, pulley_radius_m: float, efficiency: float) -> float:
		force = float(force_N)
		radius = float(pulley_radius_m)
		efficiency_value = float(efficiency)
		if not all(math.isfinite(value) for value in (force, radius, efficiency_value)):
			raise ValueError("Finger force, pulley radius, and efficiency must be finite")
		if radius <= 0.0 or efficiency_value <= 0.0:
			raise ValueError("Pulley radius and efficiency must be positive")
		torque = force * radius / efficiency_value
		return self.config.validate_closing_torque(
			torque,
			allow_zero=True,
		)

	@Slot(float, float, float)
	def start_finger_force_hold(self, force_N: float, pulley_radius_m: float, efficiency: float) -> None:
		self._trace(
			"REQUEST",
			f"start finger force={force_N:g} N, radius={pulley_radius_m:g} m, "
			f"efficiency={efficiency:g}",
		)
		try:
			torque = self._torque_for_finger_force(force_N, pulley_radius_m, efficiency)
		except Exception as exc:
			self._reject("finger_force", exc)
			return
		self.start_torque_hold(torque)

	@Slot(float, float, float)
	def update_finger_force(self, force_N: float, pulley_radius_m: float, efficiency: float) -> None:
		self._trace(
			"REQUEST",
			f"update finger force={force_N:g} N, radius={pulley_radius_m:g} m, "
			f"efficiency={efficiency:g}",
		)
		try:
			torque = self._torque_for_finger_force(force_N, pulley_radius_m, efficiency)
		except Exception as exc:
			self._reject("update_finger_force", exc)
			return
		self.update_target_torque(torque)

	@Slot()
	def release_torque(self) -> None:
		command = "release"
		self._trace("REQUEST", "release torque")
		try:
			self._require_backend()
			self._safe_stop(disable=self.config.disable_after_stop)
			self._state = MotorState.IDLE
			self._complete(command, True, "Motor torque released; drive disabled.")
		except Exception as exc:
			self._enter_fault("Torque release failed", exc, command)

	@Slot()
	def stop_motor(self) -> None:
		command = "stop"
		self._trace("REQUEST", "STOP MOTOR")
		if not self._connected:
			self._state = MotorState.DISCONNECTED
			self._complete(command, True, "Motor is disconnected.")
			return
		try:
			self._safe_stop(disable=self.config.disable_after_stop)
			self._state = MotorState.IDLE
			self._log("warning", "Motor stopped: zero torque, IDLE, drive disabled.")
			self._complete(command, True, "Motor stopped safely.")
		except Exception as exc:
			self._enter_fault("Motor stop failed", exc, command, cleanup=False)

	def _begin_position_move(self, backend: MotorBackend, position: float, velocity: float, acceleration: float, kind: str) -> None:
		backend.start_profiled_position(position, velocity, acceleration)
		self._target_position_rad = position
		self._commanded_torque_Nm = 0.0
		self._motion_started = time.monotonic()
		self._motion_kind = kind
		self._state = MotorState.MOVING

	def _poll_feedback(self) -> None:
		backend = self._require_backend()
		position = float(backend.read_position())
		velocity = float(backend.read_velocity())
		torque = float(backend.read_measured_torque())
		temperature = float(backend.read_temperature())
		if not all(math.isfinite(value) for value in (position, velocity, torque, temperature)):
			raise RuntimeError("Motor feedback contains a non-finite value")
		# The encoder is intentionally not zeroed until homing finishes, so its
		# pre-home reading may be below the configured profiled-command minimum.
		# Keep that minimum as a command limit; do not make it a feedback interlock.
		if position > self.config.maximum_position_rad:
			raise RuntimeError(
				f"Motor position {position:g} rad exceeds the configured maximum "
				f"{self.config.maximum_position_rad:g} rad"
			)
		if temperature > self.config.maximum_temperature_C:
			raise RuntimeError(
				f"Motor temperature {temperature:g} °C exceeds "
				f"{self.config.maximum_temperature_C:g} °C"
			)
		self._position_rad = position
		self._velocity_rad_s = velocity
		self._measured_torque_Nm = torque
		self._temperature_C = temperature
		self._feedback_monotonic = time.monotonic()
		self._trace(
			"FEEDBACK",
			f"position={position:.6f} rad, velocity={velocity:.6f} rad/s, "
			f"measured_torque={torque:.6f} Nm, temperature={temperature:.3f} °C, "
			f"commanded_torque={self._commanded_torque_Nm:.6f} Nm, "
			f"target_torque={self._target_torque_Nm:.6f} Nm",
		)

	@Slot()
	def _tick(self) -> None:
		if not self._connected or self._state in {
			MotorState.CONNECTING,
			MotorState.STOPPING,
			MotorState.FAULT,
			MotorState.DISCONNECTED,
		}:
			return
		now = time.monotonic()
		try:
			self._poll_feedback()
			if self._state == MotorState.HOMING:
				self._tick_homing(now)
			elif self._state == MotorState.MOVING:
				self._tick_moving(now)
			elif self._state == MotorState.HOLDING_TORQUE:
				self._tick_torque(now)
			self._emit_state()
		except Exception as exc:
			self._enter_fault("Motor safety/feedback failure", exc, "automatic")
		finally:
			self._last_tick = now

	def _tick_homing(self, now: float) -> None:
		assert self._motion_started is not None
		assert self._position_rad is not None
		if now - self._motion_started > self.config.homing_timeout_s:
			raise TimeoutError("Homing timed out before detecting a stall")
		if self._last_position is not None and abs(self._position_rad - self._last_position) < self.config.stall_epsilon_rad:
			if self._stall_started is None:
				self._stall_started = now
			elif now - self._stall_started >= self.config.stall_duration_s:
				backend = self._require_backend()
				backend.release_torque()
				backend.stop_idle()
				backend.zero_position()
				self._position_rad = 0.0
				self._homed = True
				self._clear_motion()
				self._state = MotorState.IDLE
				self._log("success", "Motor homing completed; encoder zeroed.")
				self._complete("home", True, "Homing completed.")
				return
		else:
			self._stall_started = None
		self._last_position = self._position_rad

	def _tick_moving(self, now: float) -> None:
		assert self._motion_started is not None
		assert self._target_position_rad is not None
		assert self._position_rad is not None
		if now - self._motion_started > self.config.movement_timeout_s:
			raise TimeoutError("Position movement timed out")
		if abs(self._position_rad - self._target_position_rad) > self.config.position_tolerance_rad:
			return
		kind = self._motion_kind
		backend = self._require_backend()
		backend.stop_idle()
		if kind == "torque_backoff":
			target = self._pending_torque_after_move
			if target is None:
				raise RuntimeError("Torque backoff completed without a pending target")
			backend.start_constant_torque(0.0)
			self._pending_torque_after_move = None
			self._target_torque_Nm = target
			self._commanded_torque_Nm = 0.0
			self._torque_target_sent_monotonic = None
			self._motion_started = now
			self._motion_kind = "torque"
			self._state = MotorState.HOLDING_TORQUE
			self._log("info", f"Backoff complete; reapproaching {target:g} Nm.")
		else:
			self._clear_motion()
			self._state = MotorState.IDLE
			self._log("success", "Position movement completed.")
			self._complete("position", True, "Position reached.")

	def _tick_torque(self, now: float) -> None:
		backend = self._require_backend()
		dt = max(0.0, min(now - self._last_tick, 0.25))
		maximum_step = self.config.torque_ramp_Nm_s * dt
		delta = self._target_torque_Nm - self._commanded_torque_Nm
		if abs(delta) <= maximum_step:
			command = self._target_torque_Nm
		else:
			command = self._commanded_torque_Nm + math.copysign(maximum_step, delta)
		command = self.config.validate_closing_torque(command, allow_zero=True)
		backend.refresh_torque(command)
		self._commanded_torque_Nm = command
		at_target = abs(command - self._target_torque_Nm) <= 1e-6
		if at_target and self._torque_target_sent_monotonic is None:
			# The feedback at the start of this tick predates the final target write.
			# Require at least one later hardware sample before claiming success.
			self._torque_target_sent_monotonic = time.monotonic()
		fresh_feedback_after_target = bool(
			self._feedback_monotonic is not None
			and self._torque_target_sent_monotonic is not None
			and self._feedback_monotonic > self._torque_target_sent_monotonic
		)
		measured_at_target = bool(
			fresh_feedback_after_target
			and self._measured_torque_Nm is not None
			and abs(self._measured_torque_Nm - self._target_torque_Nm)
			<= self.config.torque_tolerance_Nm
		)
		if at_target and measured_at_target and not self._torque_confirmed:
			self._torque_confirmed = True
			completion = self._pending_torque_command or "torque"
			self._pending_torque_command = ""
			message = (
				f"Measured torque {self._measured_torque_Nm:g} Nm confirmed within "
				f"±{self.config.torque_tolerance_Nm:g} Nm of target "
				f"{self._target_torque_Nm:g} Nm."
			)
			self._log("success", message)
			self._complete(completion, True, message)

	def _clear_motion(self) -> None:
		self._commanded_torque_Nm = 0.0
		self._target_torque_Nm = 0.0
		self._target_position_rad = None
		self._motion_started = None
		self._motion_kind = ""
		self._last_position = None
		self._stall_started = None
		self._pending_torque_after_move = None
		self._torque_confirmed = False
		self._pending_torque_command = ""
		self._torque_target_sent_monotonic = None

	def _safe_stop(self, *, disable: bool) -> None:
		backend = self._require_backend()
		self._state = MotorState.STOPPING
		failures: list[tuple[str, BaseException]] = []
		for label, action in (
			("zero torque", backend.release_torque),
			("IDLE", backend.stop_idle),
			("disable", backend.disable if disable and self._enabled else None),
		):
			if action is None:
				continue
			try:
				action()
			except Exception as exc:
				failures.append((label, exc))
		self._enabled = False if disable else self._enabled
		self._clear_motion()
		if failures:
			first_label, first_error = failures[0]
			for label, error in failures[1:]:
				self._log("warning", f"Motor stop also failed during {label}: {error}")
			raise RuntimeError(f"Motor stop failed during {first_label}: {first_error}") from first_error

	def _report_exception(self, context: str, exc: BaseException) -> None:
		self._latest_error = f"{context}: {exc}"
		self._log("error", self._latest_error)
		self._log("debug", "".join(traceback.format_exception(exc)).rstrip())

	def _enter_fault(self, context: str, exc: BaseException, command: str, *, cleanup: bool = True) -> None:
		# Report the triggering exception first and keep cleanup failures secondary.
		self._report_exception(context, exc)
		if cleanup and self._connected and self._backend is not None:
			try:
				self._safe_stop(disable=True)
			except Exception as cleanup_exc:
				self._log("warning", f"Motor fault cleanup also failed: {cleanup_exc}")
				self._log("debug", "".join(traceback.format_exception(cleanup_exc)).rstrip())
		self._state = MotorState.FAULT
		self._complete(command, False, str(exc))

	@Slot()
	def shutdown(self) -> None:
		if self._shutdown_requested:
			return
		self._trace("REQUEST", "motor worker shutdown")
		self._shutdown_requested = True
		failures: list[str] = []
		if self._connected and self._backend is not None:
			try:
				self._safe_stop(disable=True)
			except Exception as exc:
				failures.append(f"stop: {exc}")
			try:
				backend = self._backend
				backend.disconnect()
				self._collect_backend_debug(backend)
			except Exception as exc:
				failures.append(f"disconnect: {exc}")
		self._backend = None
		self._connected = False
		self._enabled = False
		self._homed = False
		self._clear_motion()
		self._state = MotorState.DISCONNECTED
		self._timer.stop()
		for failure in failures:
			self._log("error", f"Motor shutdown failure: {failure}")
		self._emit_state()
		self.shutdown_complete.emit()
		QThread.currentThread().quit()


class MotorController(QObject):
	"""GUI-thread facade; all backend operations are queued to MotorWorker."""

	state_updated = Signal(object)
	log_emitted = Signal(str, str)
	command_completed = Signal(str, bool, str)
	debug_events_emitted = Signal(object)

	_connect = Signal()
	_disconnect = Signal()
	_enable = Signal()
	_disable = Signal()
	_clear_faults = Signal()
	_home = Signal()
	_position = Signal(float, float, float)
	_torque = Signal(float)
	_update_torque = Signal(float)
	_finger_force = Signal(float, float, float)
	_update_finger_force = Signal(float, float, float)
	_configure_ranges = Signal(float, float, float, float, float, float)
	_release = Signal()
	_stop = Signal()
	_shutdown = Signal()

	def __init__(
		self,
		config: MotorConfig | None = None,
		backend_factory: BackendFactory | None = None,
		parent=None,
	) -> None:
		super().__init__(parent)
		persist_path = default_motor_config_path() if config is None else None
		self.config = config or load_motor_config(persist_path)
		self._thread = QThread(self)
		self._thread.setObjectName("motor-worker")
		self._worker = MotorWorker(
			self.config,
			backend_factory,
			config_persist_path=persist_path,
		)
		self._worker.moveToThread(self._thread)
		self._connect.connect(self._worker.connect_motor)
		self._disconnect.connect(self._worker.disconnect_motor)
		self._enable.connect(self._worker.enable_drive)
		self._disable.connect(self._worker.disable_drive)
		self._clear_faults.connect(self._worker.clear_faults)
		self._home.connect(self._worker.start_homing)
		self._position.connect(self._worker.move_to_position)
		self._torque.connect(self._worker.start_torque_hold)
		self._update_torque.connect(self._worker.update_target_torque)
		self._finger_force.connect(self._worker.start_finger_force_hold)
		self._update_finger_force.connect(self._worker.update_finger_force)
		self._configure_ranges.connect(self._worker.configure_ranges)
		self._release.connect(self._worker.release_torque)
		self._stop.connect(self._worker.stop_motor)
		self._shutdown.connect(self._worker.shutdown)
		self._worker.state_updated.connect(self._handle_worker_state)
		self._worker.log_emitted.connect(self.log_emitted)
		self._worker.command_completed.connect(self.command_completed)
		self._worker.debug_events_emitted.connect(self.debug_events_emitted)
		self._thread.started.connect(self._worker.initialize)
		self._thread.start()

	@Slot(object)
	def _handle_worker_state(self, state: dict) -> None:
		configuration = state.get("configuration")
		if isinstance(configuration, dict):
			self.config = MotorConfig(**configuration)
		self.state_updated.emit(state)

	def connect_motor(self) -> None: self._connect.emit()
	def disconnect_motor(self) -> None: self._disconnect.emit()
	def enable_drive(self) -> None: self._enable.emit()
	def disable_drive(self) -> None: self._disable.emit()
	def clear_faults(self) -> None: self._clear_faults.emit()
	def home(self) -> None: self._home.emit()
	def move_to_position(self, position_rad: float, velocity_rad_s: float, acceleration_rad_s2: float) -> None:
		self._position.emit(float(position_rad), float(velocity_rad_s), float(acceleration_rad_s2))
	def open_gripper(self) -> None:
		self.move_to_position(
			self.config.open_position_rad,
			self.config.profile_velocity_rad_s,
			self.config.profile_acceleration_rad_s2,
		)
	def hold_torque(self, torque_Nm: float) -> None: self._torque.emit(float(torque_Nm))
	def update_torque(self, torque_Nm: float) -> None: self._update_torque.emit(float(torque_Nm))
	def hold_finger_force(self, force_N: float, pulley_radius_m: float, efficiency: float) -> None:
		self._finger_force.emit(float(force_N), float(pulley_radius_m), float(efficiency))
	def update_finger_force(self, force_N: float, pulley_radius_m: float, efficiency: float) -> None:
		self._update_finger_force.emit(float(force_N), float(pulley_radius_m), float(efficiency))
	def configure_ranges(
		self,
		minimum_position_rad: float,
		maximum_position_rad: float,
		open_position_rad: float,
		minimum_closing_torque_Nm: float,
		maximum_velocity_rad_s: float,
		maximum_acceleration_rad_s2: float,
	) -> None:
		self._configure_ranges.emit(
			float(minimum_position_rad),
			float(maximum_position_rad),
			float(open_position_rad),
			float(minimum_closing_torque_Nm),
			float(maximum_velocity_rad_s),
			float(maximum_acceleration_rad_s2),
		)
	def release_torque(self) -> None: self._release.emit()
	def stop_motor(self) -> None: self._stop.emit()

	def shutdown(self, timeout_ms: int = 5000) -> bool:
		if not self._thread.isRunning():
			return True
		self._shutdown.emit()
		if self._thread.wait(timeout_ms):
			return True
		self.log_emitted.emit("error", "Motor worker did not stop within the timeout.")
		return False
