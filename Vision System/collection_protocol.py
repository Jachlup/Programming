"""Validated protocol definitions for operator-guided dataset collection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Protocol

import yaml


class ForceSource(Protocol):
	"""Source of a confirmed ground-truth force label in newtons."""

	def confirm(self, force_N: float) -> float: ...


@dataclass
class OperatorForceSource:
	"""Force label explicitly entered and confirmed by the operator."""

	last_confirmed_force_N: float | None = None

	def confirm(self, force_N: float) -> float:
		try:
			value = float(force_N)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError("Operator force must be numeric") from exc
		if not math.isfinite(value):
			raise ValueError("Operator force must be finite")
		self.last_confirmed_force_N = value
		return value


@dataclass(frozen=True)
class CollectionStep:
	index: int
	force_N: float
	direction: str
	repetition: int
	samples: int
	force_step_id: str
	motor_torque_Nm: float | None = None


@dataclass(frozen=True)
class CollectionProtocol:
	experiment_id: str
	steps: tuple[CollectionStep, ...]
	settling_time_s: float
	sample_interval_ms: int
	required_stability_duration_s: float
	require_operator_confirmation: bool
	use_motor: bool
	maximum_motor_telemetry_age_s: float
	minimum_tracking_quality: float
	minimum_origin_quality: float
	minimum_geometry_quality: float
	maximum_feature_movement: float
	maximum_motor_velocity_rad_s: float
	maximum_torque_variation_Nm: float
	require_zero_force_baseline: bool

	@property
	def planned_samples(self) -> int:
		return sum(step.samples for step in self.steps)


def _finite_number(value: Any, label: str) -> float:
	try:
		number = float(value)
	except (TypeError, ValueError, OverflowError) as exc:
		raise ValueError(f"{label} must be numeric") from exc
	if not math.isfinite(number):
		raise ValueError(f"{label} must be finite")
	return number


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
	number = _finite_number(value, label)
	if number < 0.0 or (not allow_zero and number == 0.0):
		raise ValueError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
	return number


def _positive_integer(value: Any, label: str) -> int:
	try:
		number = int(value)
	except (TypeError, ValueError, OverflowError) as exc:
		raise ValueError(f"{label} must be an integer") from exc
	if number < 1:
		raise ValueError(f"{label} must be positive")
	return number


def _direction(value: Any) -> str:
	direction = str(value).strip().lower()
	if direction not in {"loading", "unloading", "baseline"}:
		raise ValueError("Step direction must be loading, unloading, or baseline")
	return direction


def _step_id(direction: str, repetition: int, index: int) -> str:
	return f"{direction}-r{repetition:02d}-s{index:03d}"


def _explicit_steps(
	items: list[Any],
	default_samples: int,
) -> tuple[CollectionStep, ...]:
	steps: list[CollectionStep] = []
	identifiers: set[str] = set()
	for index, raw in enumerate(items, start=1):
		if not isinstance(raw, dict):
			raise ValueError("Every protocol step must be a mapping")
		force = _finite_number(raw.get("force_N"), f"step {index} force_N")
		direction = _direction(raw.get("direction", "baseline" if force == 0.0 else "loading"))
		repetition = _positive_integer(raw.get("repetition", 1), f"step {index} repetition")
		samples = _positive_integer(raw.get("samples", default_samples), f"step {index} samples")
		identifier = str(raw.get("force_step_id") or _step_id(direction, repetition, index)).strip()
		if not identifier or identifier in identifiers:
			raise ValueError("Force step IDs must be non-empty and unique")
		identifiers.add(identifier)
		torque_raw = raw.get("motor_torque_Nm")
		torque = None if torque_raw is None else _finite_number(
			torque_raw, f"step {index} motor_torque_Nm"
		)
		steps.append(CollectionStep(
			index=index,
			force_N=force,
			direction=direction,
			repetition=repetition,
			samples=samples,
			force_step_id=identifier,
			motor_torque_Nm=torque,
		))
	return tuple(steps)


def _generated_steps(mapping: dict[str, Any], default_samples: int) -> tuple[CollectionStep, ...]:
	levels_raw = mapping.get("force_levels_N")
	if not isinstance(levels_raw, list) or not levels_raw:
		raise ValueError("force_levels_N must contain at least one force")
	levels = [_finite_number(value, "force level") for value in levels_raw]
	repetitions = _positive_integer(mapping.get("repetitions", 1), "repetitions")
	sequence = str(mapping.get("sequence", "loading_unloading")).strip().lower()
	if sequence not in {"loading", "unloading", "loading_unloading", "alternating"}:
		raise ValueError(
			"sequence must be loading, unloading, loading_unloading, or alternating"
		)
	steps: list[CollectionStep] = []
	index = 0
	for repetition in range(1, repetitions + 1):
		if sequence == "loading":
			passes = (("loading", levels),)
		elif sequence == "unloading":
			passes = (("unloading", list(reversed(levels))),)
		elif sequence == "alternating":
			passes = (
				("loading", levels),
			) if repetition % 2 else (("unloading", list(reversed(levels))),)
		else:
			passes = (("loading", levels), ("unloading", list(reversed(levels))))
		for pass_direction, pass_levels in passes:
			for force in pass_levels:
				index += 1
				direction = "baseline" if force == 0.0 else pass_direction
				steps.append(CollectionStep(
					index=index,
					force_N=force,
					direction=direction,
					repetition=repetition,
					samples=default_samples,
					force_step_id=_step_id(direction, repetition, index),
				))
	return tuple(steps)


def protocol_from_mapping(mapping: dict[str, Any]) -> CollectionProtocol:
	if not isinstance(mapping, dict):
		raise ValueError("Collection protocol must be a mapping")
	values = mapping.get("collection_protocol", mapping)
	if not isinstance(values, dict):
		raise ValueError("collection_protocol must be a mapping")
	experiment_id = str(values.get("experiment_id", "")).strip()
	if not experiment_id or Path(experiment_id).name != experiment_id:
		raise ValueError("experiment_id must be one non-empty path-safe name")
	default_samples = _positive_integer(values.get("samples_per_step", 1), "samples_per_step")
	steps_raw = values.get("steps")
	if steps_raw is not None:
		if not isinstance(steps_raw, list) or not steps_raw:
			raise ValueError("steps must contain at least one step")
		steps = _explicit_steps(steps_raw, default_samples)
	else:
		steps = _generated_steps(values, default_samples)
	require_zero = bool(values.get("require_zero_force_baseline", True))
	if require_zero and not any(step.force_N == 0.0 for step in steps):
		raise ValueError("Protocol requires a zero-force baseline but has no zero step")

	def quality(name: str, default: float) -> float:
		value = _finite_number(values.get(name, default), name)
		if not 0.0 <= value <= 1.0:
			raise ValueError(f"{name} must be between zero and one")
		return value

	return CollectionProtocol(
		experiment_id=experiment_id,
		steps=steps,
		settling_time_s=_positive_number(
			values.get("settling_time_s", 0.5), "settling_time_s", allow_zero=True
		),
		sample_interval_ms=int(_positive_number(
			values.get("sample_interval_ms", 100), "sample_interval_ms", allow_zero=True
		)),
		required_stability_duration_s=_positive_number(
			values.get("required_stability_duration_s", 0.5),
			"required_stability_duration_s",
			allow_zero=True,
		),
		require_operator_confirmation=bool(
			values.get("require_operator_confirmation", True)
		),
		use_motor=bool(values.get("use_motor", True)),
		maximum_motor_telemetry_age_s=_positive_number(
			values.get("maximum_motor_telemetry_age_s", 0.25),
			"maximum_motor_telemetry_age_s",
		),
		minimum_tracking_quality=quality("minimum_tracking_quality", 0.5),
		minimum_origin_quality=quality("minimum_origin_quality", 0.5),
		minimum_geometry_quality=quality("minimum_geometry_quality", 0.5),
		maximum_feature_movement=_positive_number(
			values.get("maximum_feature_movement", 0.25),
			"maximum_feature_movement",
			allow_zero=True,
		),
		maximum_motor_velocity_rad_s=_positive_number(
			values.get("maximum_motor_velocity_rad_s", 0.02),
			"maximum_motor_velocity_rad_s",
			allow_zero=True,
		),
		maximum_torque_variation_Nm=_positive_number(
			values.get("maximum_torque_variation_Nm", 0.1),
			"maximum_torque_variation_Nm",
			allow_zero=True,
		),
		require_zero_force_baseline=require_zero,
	)


def load_collection_protocol(path: str | Path) -> CollectionProtocol:
	protocol_path = Path(path).expanduser().resolve()
	with protocol_path.open("r", encoding="utf-8") as stream:
		mapping = yaml.safe_load(stream) or {}
	return protocol_from_mapping(mapping)
