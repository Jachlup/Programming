"""C++ motor-process adapter and a deterministic hardware-free test backend."""

from __future__ import annotations

import math
import os
from pathlib import Path
import select
import subprocess
import time
from typing import Protocol, runtime_checkable


class _CppBridgeCommandError(RuntimeError):
	"""The bridge rejected a command but the IPC stream remains synchronized."""


@runtime_checkable
class MotorBackend(Protocol):
	"""Operations available to the motor worker; values use SI-derived units."""

	backend_type: str

	def connect(self) -> None: ...
	def disconnect(self) -> None: ...
	def enable(self) -> None: ...
	def disable(self) -> None: ...
	def clear_faults(self) -> None: ...
	def zero_position(self) -> None: ...
	def start_profiled_position(self, position_rad: float, velocity_rad_s: float, acceleration_rad_s2: float) -> None: ...
	def start_constant_torque(self, torque_Nm: float) -> None: ...
	def refresh_torque(self, torque_Nm: float) -> None: ...
	def release_torque(self) -> None: ...
	def stop_idle(self) -> None: ...
	def read_position(self) -> float: ...
	def read_velocity(self) -> float: ...
	def read_measured_torque(self) -> float: ...
	def read_temperature(self) -> float: ...
	def apply_current_limit(self, current_A: float) -> None: ...
	def apply_maximum_torque(self, torque_Nm: float) -> None: ...


class CppMotorBackend:
	"""IPC client for the persistent C++ CANdle/MD motor bridge.

	Python never loads a CANdle binding. Every SDK call, safety-limit check, and
	fail-safe drive shutdown performed by this backend happens in ``gripper_bridge``.
	"""

	backend_type = "hardware-cpp"
	_PROTOCOL_PREFIX = b"MOTOR_BRIDGE "
	_PROTOCOL_VERSION = "1"

	def __init__(
		self,
		can_id: int = 21,
		*,
		current_limit_A: float = 0.5,
		maximum_controller_torque_Nm: float = 10.0,
		minimum_position_rad: float = -1.0,
		maximum_position_rad: float = 2.2,
		maximum_velocity_rad_s: float = 5.0,
		maximum_acceleration_rad_s2: float = 20.0,
		closing_direction: int = -1,
		maximum_closing_torque_Nm: float = 4.0,
		maximum_temperature_C: float = 50.0,
		executable_path: str | Path | None = None,
		command_timeout_s: float = 5.0,
		startup_timeout_s: float = 8.0,
	) -> None:
		self.can_id = int(can_id)
		self.current_limit_A = self._finite_positive(current_limit_A, "current limit")
		self.maximum_controller_torque_Nm = self._finite_positive(
			maximum_controller_torque_Nm, "maximum controller torque"
		)
		self.minimum_position_rad = self._finite(minimum_position_rad, "minimum position")
		self.maximum_position_rad = self._finite(maximum_position_rad, "maximum position")
		self.maximum_velocity_rad_s = self._finite_positive(
			maximum_velocity_rad_s, "maximum velocity"
		)
		self.maximum_acceleration_rad_s2 = self._finite_positive(
			maximum_acceleration_rad_s2, "maximum acceleration"
		)
		self.closing_direction = int(closing_direction)
		self.maximum_closing_torque_Nm = self._finite_positive(
			maximum_closing_torque_Nm, "maximum closing torque"
		)
		self.maximum_temperature_C = self._finite_positive(
			maximum_temperature_C, "maximum temperature"
		)
		if not 1 <= self.can_id <= 127:
			raise ValueError("CAN ID must be in 1..127")
		if self.minimum_position_rad >= self.maximum_position_rad:
			raise ValueError("Motor position range is empty")
		if self.minimum_position_rad > 0.0 or self.maximum_position_rad < 0.0:
			raise ValueError("Motor position range must include the homed 0 rad origin")
		if self.closing_direction != -1:
			raise ValueError("Closing direction must be -1 for this gripper")
		if self.maximum_closing_torque_Nm > self.maximum_controller_torque_Nm:
			raise ValueError("Closing torque limit exceeds controller limit")
		fixed_ceilings = (
			(self.current_limit_A, 0.5, "current limit"),
			(self.maximum_controller_torque_Nm, 10.0, "controller torque limit"),
			(self.maximum_position_rad, 2.2, "maximum position"),
			(self.maximum_velocity_rad_s, 5.0, "maximum velocity"),
			(self.maximum_acceleration_rad_s2, 20.0, "maximum acceleration"),
			(self.maximum_closing_torque_Nm, 4.0, "closing torque limit"),
			(self.maximum_temperature_C, 50.0, "maximum temperature"),
		)
		for value, ceiling, label in fixed_ceilings:
			if value > ceiling:
				raise ValueError(f"{label} exceeds the fixed safety limit {ceiling:g}")
		if self.minimum_position_rad < -1.0:
			raise ValueError("Minimum position exceeds the fixed safety limit -1")
		self.executable_path = (
			Path(executable_path).expanduser().resolve()
			if executable_path is not None
			else self.default_executable()
		)
		self.command_timeout_s = self._finite_positive(command_timeout_s, "command timeout")
		self.startup_timeout_s = self._finite_positive(startup_timeout_s, "startup timeout")
		self.connected = False
		self.enabled = False
		self._process: subprocess.Popen[bytes] | None = None
		self._read_buffer = bytearray()
		self._recent_output: list[str] = []
		self._feedback_cache: tuple[float, float, float, float] | None = None
		self._debug_events: list[dict] = []
		self._debug_sequence = 0

	def _trace(self, category: str, message: str) -> None:
		self._debug_sequence += 1
		self._debug_events.append({
			"sequence": self._debug_sequence,
			"wall_time": time.time(),
			"monotonic": time.monotonic(),
			"source": "C++ bridge client",
			"category": str(category),
			"message": str(message),
		})
		# Connection failures can occur before MotorWorker has a chance to drain.
		del self._debug_events[:-2000]

	def drain_debug_events(self) -> list[dict]:
		events = self._debug_events
		self._debug_events = []
		return events

	@staticmethod
	def default_executable() -> Path:
		return Path(__file__).resolve().parents[2] / "Motor" / "build" / "gripper_bridge"

	@staticmethod
	def _finite(value: float, label: str) -> float:
		result = float(value)
		if not math.isfinite(result):
			raise ValueError(f"{label} must be finite")
		return result

	@classmethod
	def _finite_positive(cls, value: float, label: str) -> float:
		result = cls._finite(value, label)
		if result <= 0.0:
			raise ValueError(f"{label} must be positive")
		return result

	@staticmethod
	def _format_number(value: float) -> str:
		result = float(value)
		if not math.isfinite(result):
			raise ValueError("Motor command contains a non-finite number")
		return format(result, ".17g")

	def _remember_output(self, line: bytes) -> None:
		text = line.decode("utf-8", errors="replace").strip()
		if text:
			self._recent_output.append(text)
			del self._recent_output[:-20]
			self._trace("NATIVE", text)

	def _read_response(self, timeout_s: float) -> str:
		process = self._process
		if process is None or process.stdout is None:
			raise RuntimeError("C++ motor bridge is not running")
		deadline = time.monotonic() + timeout_s
		while True:
			newline = self._read_buffer.find(b"\n")
			if newline >= 0:
				line = bytes(self._read_buffer[:newline]).rstrip(b"\r")
				del self._read_buffer[:newline + 1]
				if not line.startswith(self._PROTOCOL_PREFIX):
					self._remember_output(line)
					continue
				response = line[len(self._PROTOCOL_PREFIX):]
				self._trace("RX", response.decode("utf-8", errors="replace"))
				status, separator, payload = response.partition(b" ")
				message = payload.decode("utf-8", errors="replace") if separator else ""
				if status == b"OK":
					return message
				if status == b"ERR":
					raise _CppBridgeCommandError(
						message or "C++ motor bridge command failed"
					)
				raise RuntimeError(f"Invalid C++ motor bridge response: {line!r}")

			remaining = deadline - time.monotonic()
			if remaining <= 0.0:
				detail = self._recent_output[-1] if self._recent_output else "no bridge output"
				raise TimeoutError(f"C++ motor bridge response timed out ({detail})")
			ready, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
			if not ready:
				continue
			chunk = os.read(process.stdout.fileno(), 4096)
			if not chunk:
				code = process.poll()
				detail = self._recent_output[-1] if self._recent_output else "no bridge output"
				self._trace("PROCESS", f"stdout closed; exit code={code}; {detail}")
				raise RuntimeError(f"C++ motor bridge exited with code {code} ({detail})")
			self._read_buffer.extend(chunk)

	def _command(self, command: str, *arguments: str, timeout_s: float | None = None) -> str:
		process = self._process
		if not self.connected or process is None or process.stdin is None:
			raise RuntimeError("Motor backend is not connected")
		if process.poll() is not None:
			raise RuntimeError(f"C++ motor bridge exited with code {process.returncode}")
		try:
			line = " ".join((command, *arguments)) + "\n"
			self._trace("TX", line.rstrip())
			process.stdin.write(line.encode("ascii"))
			process.stdin.flush()
			self._feedback_cache = None
			return self._read_response(timeout_s or self.command_timeout_s)
		except _CppBridgeCommandError:
			# The C++ side performs zero-torque, IDLE, and disable before ERR.
			self.enabled = False
			raise
		except BaseException as exc:
			# A timeout or broken stream cannot be reused: a delayed response would
			# otherwise be mistaken for the next command's response.
			self._terminate_process()
			if isinstance(exc, (BrokenPipeError, OSError)):
				raise RuntimeError("Could not communicate with C++ motor bridge") from exc
			raise

	def _configure_bridge(self) -> None:
		self._command(
			"CONFIGURE",
			self._format_number(self.current_limit_A),
			self._format_number(self.maximum_controller_torque_Nm),
			self._format_number(self.minimum_position_rad),
			self._format_number(self.maximum_position_rad),
			self._format_number(self.maximum_velocity_rad_s),
			self._format_number(self.maximum_acceleration_rad_s2),
			str(self.closing_direction),
			self._format_number(self.maximum_closing_torque_Nm),
			self._format_number(self.maximum_temperature_C),
		)

	def _close_streams(self) -> None:
		process = self._process
		if process is None:
			return
		for stream in (process.stdin, process.stdout):
			if stream is not None:
				try:
					stream.close()
				except OSError:
					pass

	def _terminate_process(self) -> None:
		process = self._process
		if process is not None and process.poll() is None:
			self._trace("PROCESS", "sending SIGTERM to C++ motor bridge")
			try:
				process.terminate()
			except ProcessLookupError:
				pass
			try:
				process.wait(timeout=5.0)
			except subprocess.TimeoutExpired:
				process.kill()
				process.wait(timeout=2.0)
		self._close_streams()
		self._process = None
		self._read_buffer.clear()
		self._feedback_cache = None
		self.connected = False
		self.enabled = False

	def connect(self) -> None:
		if self.connected or self._process is not None:
			raise RuntimeError("Motor backend is already connected")
		if not self.executable_path.is_file():
			raise FileNotFoundError(
				f"C++ motor bridge was not built: {self.executable_path}. Run Motor/build.sh."
			)
		self._recent_output.clear()
		self._read_buffer.clear()
		try:
			self._trace(
				"PROCESS",
				f"starting {self.executable_path} for CAN ID {self.can_id}; "
				f"GUI pid={os.getpid()}, euid={os.geteuid()}, egid={os.getegid()}, "
				f"groups={os.getgroups()}",
			)
			self._process = subprocess.Popen(
				[str(self.executable_path), str(self.can_id)],
				cwd=str(self.executable_path.parent),
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				bufsize=0,
			)
			self._trace("PROCESS", f"C++ bridge subprocess pid={self._process.pid}")
			ready = self._read_response(self.startup_timeout_s).split()
			if ready != ["READY", self._PROTOCOL_VERSION]:
				raise RuntimeError(f"Unsupported C++ motor bridge handshake: {' '.join(ready)}")
			self.connected = True
			self.enabled = False
			self._trace("PROCESS", "bridge handshake accepted; drive reported disabled")
			self._configure_bridge()
		except Exception:
			self._terminate_process()
			raise

	def disconnect(self) -> None:
		process = self._process
		if process is None:
			self._trace("PROCESS", "disconnect requested with no bridge process")
			self.connected = False
			self.enabled = False
			return
		error: BaseException | None = None
		if process.poll() is None and self.connected:
			try:
				self._command("QUIT", timeout_s=self.command_timeout_s)
			except BaseException as exc:
				error = exc
		try:
			process.wait(timeout=2.0)
		except subprocess.TimeoutExpired:
			self._terminate_process()
		else:
			self._close_streams()
			self._process = None
			self._read_buffer.clear()
			self._feedback_cache = None
			self.connected = False
			self.enabled = False
		if error is not None:
			raise RuntimeError(f"C++ motor bridge shutdown failed: {error}") from error

	def enable(self) -> None:
		self._command("ENABLE")
		self.enabled = True

	def disable(self) -> None:
		self._command("DISABLE")
		self.enabled = False

	def clear_faults(self) -> None:
		self._command("CLEAR_FAULTS")

	def zero_position(self) -> None:
		self._command("ZERO_POSITION")

	def start_profiled_position(self, position_rad: float, velocity_rad_s: float, acceleration_rad_s2: float) -> None:
		self._command(
			"PROFILE",
			self._format_number(position_rad),
			self._format_number(velocity_rad_s),
			self._format_number(acceleration_rad_s2),
		)

	def start_constant_torque(self, torque_Nm: float) -> None:
		self._command("START_TORQUE", self._format_number(torque_Nm))

	def refresh_torque(self, torque_Nm: float) -> None:
		self._command("TORQUE", self._format_number(torque_Nm))

	def release_torque(self) -> None:
		self._command("RELEASE_TORQUE")

	def stop_idle(self) -> None:
		self._command("STOP_IDLE")

	def _feedback(self, *, refresh: bool) -> tuple[float, float, float, float]:
		if refresh or self._feedback_cache is None:
			parts = self._command("FEEDBACK").split()
			if len(parts) != 4:
				raise RuntimeError(f"Invalid C++ motor feedback: {' '.join(parts)}")
			values = tuple(float(part) for part in parts)
			if not all(math.isfinite(value) for value in values):
				raise RuntimeError("C++ motor bridge returned non-finite feedback")
			self._feedback_cache = values  # type: ignore[assignment]
		return self._feedback_cache

	def read_position(self) -> float:
		return self._feedback(refresh=True)[0]

	def read_velocity(self) -> float:
		return self._feedback(refresh=False)[1]

	def read_measured_torque(self) -> float:
		return self._feedback(refresh=False)[2]

	def read_temperature(self) -> float:
		value = self._feedback(refresh=False)[3]
		self._feedback_cache = None
		return value

	def apply_current_limit(self, current_A: float) -> None:
		self._command("CURRENT_LIMIT", self._format_number(current_A))

	def apply_maximum_torque(self, torque_Nm: float) -> None:
		self._command("MAX_TORQUE", self._format_number(abs(float(torque_Nm))))


class MockMotorBackend:
	"""Hardware-free backend with time-based position and torque simulation."""

	backend_type = "mock"

	def __init__(
		self,
		*,
		minimum_position_rad: float = -1.0,
		maximum_position_rad: float = 2.2,
		initial_position_rad: float = 2.0,
	) -> None:
		self.minimum_position_rad = float(minimum_position_rad)
		self.maximum_position_rad = float(maximum_position_rad)
		self.position_rad = float(initial_position_rad)
		self.velocity_rad_s = 0.0
		self.commanded_torque_Nm = 0.0
		self.measured_torque_Nm = 0.0
		self.temperature_C = 25.0
		self.connected = False
		self.enabled = False
		self.mode = "IDLE"
		self.target_position_rad: float | None = None
		self.profile_velocity_rad_s = 0.0
		self.current_limit_A: float | None = None
		self.maximum_torque_Nm: float | None = None
		self.fail_operations: set[str] = set()
		self.calls: list[tuple] = []
		self._last_update = time.monotonic()

	def _fail(self, operation: str) -> None:
		self.calls.append((operation,))
		if operation in self.fail_operations:
			raise RuntimeError(f"Mock failure: {operation}")

	def _require_connected(self) -> None:
		if not self.connected:
			raise RuntimeError("Mock motor is not connected")

	def _require_enabled(self) -> None:
		self._require_connected()
		if not self.enabled:
			raise RuntimeError("Mock motor drive is disabled")

	def _advance(self) -> None:
		now = time.monotonic()
		dt = max(0.0, min(now - self._last_update, 0.1))
		self._last_update = now
		if not self.connected or not self.enabled:
			self.velocity_rad_s = 0.0
			return
		if self.mode == "PROFILE_POSITION" and self.target_position_rad is not None:
			delta = self.target_position_rad - self.position_rad
			step = min(abs(delta), self.profile_velocity_rad_s * dt)
			self.velocity_rad_s = 0.0 if dt <= 0.0 else math.copysign(step / dt, delta)
			self.position_rad += math.copysign(step, delta) if step else 0.0
			if abs(delta) <= 1e-9 or step >= abs(delta):
				self.position_rad = self.target_position_rad
				self.velocity_rad_s = 0.0
		elif self.mode == "RAW_TORQUE":
			# A modest deterministic rate is sufficient for state-machine tests.
			requested_velocity = self.commanded_torque_Nm * 2.0
			candidate = self.position_rad + requested_velocity * dt
			# Homing torque closes against the physical zero stop even when normal
			# profiled-position commands are allowed down to a negative position.
			clamped = min(max(candidate, 0.0), self.maximum_position_rad)
			self.velocity_rad_s = requested_velocity if clamped == candidate else 0.0
			self.position_rad = clamped
			self.measured_torque_Nm = self.commanded_torque_Nm
		else:
			self.velocity_rad_s = 0.0

	def connect(self) -> None:
		self._fail("connect")
		if self.connected:
			raise RuntimeError("Mock motor is already connected")
		self.connected = True
		self.enabled = False
		self.mode = "IDLE"
		self._last_update = time.monotonic()

	def disconnect(self) -> None:
		self._fail("disconnect")
		self.connected = False
		self.enabled = False
		self.mode = "IDLE"
		self.velocity_rad_s = 0.0
		self.commanded_torque_Nm = 0.0
		self.measured_torque_Nm = 0.0

	def enable(self) -> None:
		self._fail("enable")
		self._require_connected()
		self.enabled = True

	def disable(self) -> None:
		self._fail("disable")
		self._require_connected()
		self.enabled = False
		self.velocity_rad_s = 0.0

	def clear_faults(self) -> None:
		self._fail("clear_faults")
		self._require_connected()

	def zero_position(self) -> None:
		self._fail("zero_position")
		self._require_enabled()
		self.position_rad = 0.0

	def start_profiled_position(self, position_rad: float, velocity_rad_s: float, acceleration_rad_s2: float) -> None:
		self._fail("start_profiled_position")
		self._require_enabled()
		self.calls.append(("profile", position_rad, velocity_rad_s, acceleration_rad_s2))
		self.target_position_rad = float(position_rad)
		self.profile_velocity_rad_s = float(velocity_rad_s)
		self.mode = "PROFILE_POSITION"

	def start_constant_torque(self, torque_Nm: float) -> None:
		self._fail("start_constant_torque")
		self._require_enabled()
		self.calls.append(("torque", torque_Nm))
		self.commanded_torque_Nm = float(torque_Nm)
		self.measured_torque_Nm = float(torque_Nm)
		self.target_position_rad = None
		self.mode = "RAW_TORQUE"

	def refresh_torque(self, torque_Nm: float) -> None:
		self._fail("refresh_torque")
		self._require_enabled()
		if self.mode != "RAW_TORQUE":
			raise RuntimeError("Mock motor is not in torque mode")
		self.calls.append(("refresh", torque_Nm))
		self.commanded_torque_Nm = float(torque_Nm)
		self.measured_torque_Nm = float(torque_Nm)

	def release_torque(self) -> None:
		self._fail("release_torque")
		self._require_connected()
		self.commanded_torque_Nm = 0.0
		self.measured_torque_Nm = 0.0

	def stop_idle(self) -> None:
		self._fail("stop_idle")
		self._require_connected()
		self.commanded_torque_Nm = 0.0
		self.measured_torque_Nm = 0.0
		self.velocity_rad_s = 0.0
		self.target_position_rad = None
		self.mode = "IDLE"

	def read_position(self) -> float:
		self._fail("read_position")
		self._require_connected()
		self._advance()
		return self.position_rad

	def read_velocity(self) -> float:
		self._fail("read_velocity")
		self._require_connected()
		self._advance()
		return self.velocity_rad_s

	def read_measured_torque(self) -> float:
		self._fail("read_measured_torque")
		self._require_connected()
		self._advance()
		return self.measured_torque_Nm

	def read_temperature(self) -> float:
		self._fail("read_temperature")
		self._require_connected()
		return self.temperature_C

	def apply_current_limit(self, current_A: float) -> None:
		self._fail("apply_current_limit")
		self._require_connected()
		self.current_limit_A = float(current_A)

	def apply_maximum_torque(self, torque_Nm: float) -> None:
		self._fail("apply_maximum_torque")
		self._require_connected()
		self.maximum_torque_Nm = abs(float(torque_Nm))
