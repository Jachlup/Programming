"""Validated motor configuration with explicit units and conservative defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


def default_motor_config_path() -> Path:
	return Path(__file__).resolve().parents[1] / "motor-config.yaml"


@dataclass(frozen=True)
class MotorConfig:
	backend: str = "mock"
	can_id: int = 21
	closing_direction: int = -1
	homing_torque_Nm: float = -0.6
	maximum_closing_torque_Nm: float = 4.0
	current_limit_A: float = 0.5
	maximum_controller_torque_Nm: float = 10.0
	minimum_position_rad: float = -1.0
	maximum_position_rad: float = 2.2
	open_position_rad: float = 2.0
	position_tolerance_rad: float = 0.5
	profile_velocity_rad_s: float = 5.0
	profile_acceleration_rad_s2: float = 20.0
	homing_timeout_s: float = 5.0
	movement_timeout_s: float = 5.0
	stall_epsilon_rad: float = 0.01
	stall_duration_s: float = 0.1
	maximum_temperature_C: float = 50.0
	feedback_interval_ms: int = 20
	disable_after_stop: bool = True
	require_homing: bool = True
	torque_ramp_Nm_s: float = 0.5
	torque_tolerance_Nm: float = 0.1
	torque_reduction_backoff_rad: float = 0.3
	pulley_radius_m: float | None = None
	efficiency: float | None = None

	def __post_init__(self) -> None:
		if self.backend not in {"mock", "hardware"}:
			raise ValueError("motor.backend must be 'mock' or 'hardware'")
		if not 1 <= int(self.can_id) <= 127:
			raise ValueError("motor.can_id must be in 1..127")
		if self.closing_direction != -1:
			raise ValueError("motor.closing_direction must be -1 for this gripper")
		finite = (
			"homing_torque_Nm",
			"maximum_closing_torque_Nm",
			"current_limit_A",
			"maximum_controller_torque_Nm",
			"minimum_position_rad",
			"maximum_position_rad",
			"open_position_rad",
			"position_tolerance_rad",
			"profile_velocity_rad_s",
			"profile_acceleration_rad_s2",
			"homing_timeout_s",
			"movement_timeout_s",
			"stall_epsilon_rad",
			"stall_duration_s",
			"maximum_temperature_C",
			"torque_ramp_Nm_s",
			"torque_tolerance_Nm",
			"torque_reduction_backoff_rad",
		)
		for name in finite:
			if not math.isfinite(float(getattr(self, name))):
				raise ValueError(f"motor.{name} must be finite")
		if self.minimum_position_rad >= self.maximum_position_rad:
			raise ValueError("motor position range is empty")
		if self.minimum_position_rad > 0.0 or self.maximum_position_rad < 0.0:
			raise ValueError("motor position range must include the homed 0 rad origin")
		if not self.minimum_position_rad <= self.open_position_rad <= self.maximum_position_rad:
			raise ValueError("motor.open_position_rad is outside the position range")
		positive = (
			"maximum_closing_torque_Nm",
			"current_limit_A",
			"maximum_controller_torque_Nm",
			"position_tolerance_rad",
			"profile_velocity_rad_s",
			"profile_acceleration_rad_s2",
			"homing_timeout_s",
			"movement_timeout_s",
			"stall_epsilon_rad",
			"stall_duration_s",
			"maximum_temperature_C",
			"torque_ramp_Nm_s",
			"torque_tolerance_Nm",
			"torque_reduction_backoff_rad",
		)
		for name in positive:
			if float(getattr(self, name)) <= 0.0:
				raise ValueError(f"motor.{name} must be positive")
		if self.maximum_closing_torque_Nm > self.maximum_controller_torque_Nm:
			raise ValueError("closing torque limit exceeds the controller torque limit")
		fixed_ceilings = {
			"maximum_closing_torque_Nm": 4.0,
			"current_limit_A": 0.5,
			"maximum_controller_torque_Nm": 10.0,
			"maximum_position_rad": 2.2,
			"profile_velocity_rad_s": 5.0,
			"profile_acceleration_rad_s2": 20.0,
			"maximum_temperature_C": 50.0,
			"torque_ramp_Nm_s": 0.5,
			"torque_reduction_backoff_rad": 0.3,
		}
		for name, ceiling in fixed_ceilings.items():
			if float(getattr(self, name)) > ceiling:
				raise ValueError(f"motor.{name} exceeds the fixed safety limit {ceiling:g}")
		if self.minimum_position_rad < -1.0:
			raise ValueError("motor.minimum_position_rad exceeds the fixed safety limit -1")
		if self.homing_torque_Nm == 0.0 or int(math.copysign(1, self.homing_torque_Nm)) != self.closing_direction:
			raise ValueError("homing torque does not use the configured closing direction")
		if abs(self.homing_torque_Nm) > self.maximum_closing_torque_Nm:
			raise ValueError("homing torque exceeds the closing torque limit")
		if int(self.feedback_interval_ms) < 1:
			raise ValueError("motor.feedback_interval_ms must be positive")
		for name in ("pulley_radius_m", "efficiency"):
			value = getattr(self, name)
			if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
				raise ValueError(f"motor.{name} must be null or a finite positive value")

	def validate_position(self, position_rad: float) -> float:
		value = _finite(position_rad, "position")
		if not self.minimum_position_rad <= value <= self.maximum_position_rad:
			raise ValueError(
				f"Position {value:g} rad is outside {self.minimum_position_rad:g}.."
				f"{self.maximum_position_rad:g} rad"
			)
		return value

	def validate_profile(self, velocity_rad_s: float, acceleration_rad_s2: float) -> tuple[float, float]:
		velocity = _finite(velocity_rad_s, "velocity")
		acceleration = _finite(acceleration_rad_s2, "acceleration")
		if velocity <= 0.0 or velocity > self.profile_velocity_rad_s:
			raise ValueError(
				f"Velocity must be in (0, {self.profile_velocity_rad_s:g}] rad/s"
			)
		if acceleration <= 0.0 or acceleration > self.profile_acceleration_rad_s2:
			raise ValueError(
				"Acceleration must be in (0, "
				f"{self.profile_acceleration_rad_s2:g}] rad/s²"
			)
		return velocity, acceleration

	def validate_closing_torque(self, torque_Nm: float, *, allow_zero: bool = False) -> float:
		value = _finite(torque_Nm, "torque")
		if value == 0.0:
			if allow_zero:
				return 0.0
			raise ValueError("Closing torque must be nonzero")
		if int(math.copysign(1, value)) != self.closing_direction:
			raise ValueError("Torque command does not use the configured closing direction")
		if abs(value) > self.maximum_closing_torque_Nm:
			raise ValueError(
				f"Closing torque magnitude exceeds {self.maximum_closing_torque_Nm:g} Nm"
			)
		return value

	def torque_for_finger_force(self, force_N: float) -> float:
		force = _finite(force_N, "finger force")
		if self.pulley_radius_m is None or self.efficiency is None:
			raise ValueError(
				"Calculated-force mode requires positive pulley_radius_m and efficiency"
			)
		# finger_force = (motor_torque / pulley_radius) * efficiency
		torque = force * self.pulley_radius_m / self.efficiency
		return self.validate_closing_torque(torque, allow_zero=True)

	def to_mapping(self) -> dict[str, Any]:
		return asdict(self)


def _finite(value: Any, label: str) -> float:
	try:
		result = float(value)
	except (TypeError, ValueError, OverflowError) as exc:
		raise ValueError(f"{label} must be a finite number") from exc
	if not math.isfinite(result):
		raise ValueError(f"{label} must be a finite number")
	return result


def load_motor_config(path: str | Path | None = None) -> MotorConfig:
	target = Path(path) if path is not None else default_motor_config_path()
	with target.open(encoding="utf-8") as stream:
		raw = yaml.safe_load(stream) or {}
	if not isinstance(raw, dict):
		raise ValueError("Motor configuration root must be a mapping")
	section = raw.get("motor", raw)
	if not isinstance(section, dict):
		raise ValueError("motor configuration section must be a mapping")
	unknown = sorted(set(section) - set(MotorConfig.__dataclass_fields__))
	if unknown:
		raise ValueError(f"Unknown motor configuration keys: {', '.join(unknown)}")
	return MotorConfig(**section)


def save_motor_config(config: MotorConfig, path: str | Path | None = None) -> None:
	"""Atomically persist a validated motor configuration."""
	target = Path(path) if path is not None else default_motor_config_path()
	target.parent.mkdir(parents=True, exist_ok=True)
	temporary_path: Path | None = None
	try:
		with tempfile.NamedTemporaryFile(
			"w",
			encoding="utf-8",
			dir=target.parent,
			prefix=f".{target.name}.",
			suffix=".tmp",
			delete=False,
		) as stream:
			temporary_path = Path(stream.name)
			yaml.safe_dump(
				{"motor": config.to_mapping()},
				stream,
				default_flow_style=False,
				sort_keys=False,
			)
			stream.flush()
			os.fsync(stream.fileno())
		# Validate the serialization before replacing the operator's file.
		if load_motor_config(temporary_path) != config:
			raise RuntimeError("Saved motor configuration did not round-trip correctly")
		os.replace(temporary_path, target)
	except Exception:
		if temporary_path is not None:
			try:
				temporary_path.unlink(missing_ok=True)
			except OSError:
				pass
		raise
