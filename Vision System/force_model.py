"""Validated loading and inference for geometric force regressors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np


class ModelCompatibilityError(ValueError):
	"""The current geometry does not match the trained model contract."""


@dataclass
class RidgeRegressor:
	"""Small pickle-safe ridge regressor implemented using NumPy only."""

	alpha: float = 1.0
	coef_: np.ndarray | None = None
	intercept_: float = 0.0
	feature_mean_: np.ndarray | None = None
	feature_scale_: np.ndarray | None = None
	n_features_in_: int = 0

	def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeRegressor":
		x = np.asarray(features, dtype=np.float64)
		y = np.asarray(targets, dtype=np.float64)
		if x.ndim != 2:
			raise ValueError("Training features must be a two-dimensional matrix")
		if y.ndim != 1:
			raise ValueError("Training targets must be a one-dimensional vector")
		if x.shape[0] != y.size or not x.shape[0] or not x.shape[1]:
			raise ValueError("Training feature and target shapes are incompatible")
		if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
			raise ValueError("Training data must contain only finite values")
		alpha = float(self.alpha)
		if not math.isfinite(alpha) or alpha < 0:
			raise ValueError("Ridge alpha must be a finite non-negative number")

		self.feature_mean_ = x.mean(axis=0)
		scale = x.std(axis=0)
		self.feature_scale_ = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
		standardised = (x - self.feature_mean_) / self.feature_scale_
		self.intercept_ = float(y.mean())
		centred_targets = y - self.intercept_

		if alpha > 0:
			regulariser = math.sqrt(alpha) * np.eye(x.shape[1])
			fit_x = np.vstack((standardised, regulariser))
			fit_y = np.concatenate((centred_targets, np.zeros(x.shape[1])))
		else:
			fit_x, fit_y = standardised, centred_targets
		self.coef_ = np.linalg.lstsq(fit_x, fit_y, rcond=None)[0]
		self.n_features_in_ = int(x.shape[1])
		return self

	def predict(self, features: np.ndarray) -> np.ndarray:
		if (
			self.coef_ is None
			or self.feature_mean_ is None
			or self.feature_scale_ is None
			or self.n_features_in_ <= 0
		):
			raise RuntimeError("RidgeRegressor has not been fitted")
		x = np.asarray(features, dtype=np.float64)
		if x.ndim != 2 or x.shape[1] != self.n_features_in_:
			raise ValueError(
				f"Expected a two-dimensional matrix with {self.n_features_in_} features"
			)
		if not np.all(np.isfinite(x)):
			raise ValueError("Prediction features must contain only finite values")
		standardised = (x - self.feature_mean_) / self.feature_scale_
		return standardised @ self.coef_ + self.intercept_


@dataclass(frozen=True)
class ModelMetadata:
	model_type: str
	feature_names: list[str]
	calibrated_min_force_N: float = 5.0
	calibrated_max_force_N: float = 40.0
	training_date: str = ""
	model_version: str = "1"
	reference_profile_compatibility: str | list[str] = "*"
	geometry_configuration_version: str = "1"
	feature_schema_version: str = "1"
	training_metrics: dict[str, Any] = field(default_factory=dict)
	validation_metrics: dict[str, Any] = field(default_factory=dict)
	validation_metrics_by_force_N: dict[str, dict[str, Any]] = field(default_factory=dict)
	training_dataset_identifiers: list[str] = field(default_factory=list)
	training_configuration: dict[str, Any] = field(default_factory=dict)
	model_name: str = "force-regressor"

	@classmethod
	def from_dict(cls, raw: dict[str, Any]) -> "ModelMetadata":
		if not isinstance(raw, dict):
			raise ValueError("Model metadata must be a JSON object")
		values = dict(raw)

		# Recognised aliases make older and externally generated metadata usable.
		aliases = {
			"compatible_reference_profile": "reference_profile_compatibility",
			"compatible_reference_profiles": "reference_profile_compatibility",
			"geometry_configuration_identifier": "geometry_configuration_version",
			"training_dataset_ids": "training_dataset_identifiers",
			"per_force_validation_metrics": "validation_metrics_by_force_N",
		}
		for old_name, canonical_name in aliases.items():
			if old_name in values:
				if canonical_name in values:
					raise ValueError(
						f"Metadata contains both {old_name!r} and {canonical_name!r}"
					)
				values[canonical_name] = values.pop(old_name)

		allowed = set(cls.__dataclass_fields__)
		unknown = sorted(set(values) - allowed)
		if unknown:
			raise ValueError(f"Unrecognised model metadata fields: {', '.join(unknown)}")
		try:
			metadata = cls(**values)
		except TypeError as exc:
			raise ValueError(f"Invalid model metadata: {exc}") from exc
		metadata.validate()
		return metadata

	def validate(self) -> None:
		if not isinstance(self.model_type, str) or not self.model_type:
			raise ValueError("model_type must be a non-empty string")
		if (
			not isinstance(self.feature_names, list)
			or not self.feature_names
			or not all(isinstance(name, str) and name for name in self.feature_names)
		):
			raise ValueError("feature_names must be a non-empty list of strings")
		if len(set(self.feature_names)) != len(self.feature_names):
			raise ValueError("feature_names must be unique")
		try:
			minimum = float(self.calibrated_min_force_N)
			maximum = float(self.calibrated_max_force_N)
		except (TypeError, ValueError, OverflowError) as exc:
			raise ValueError("Calibrated force bounds must be finite numbers") from exc
		if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
			raise ValueError("Calibrated force range is invalid")
		for name, value in (
			("model_version", self.model_version),
			("geometry_configuration_version", self.geometry_configuration_version),
			("feature_schema_version", self.feature_schema_version),
			("model_name", self.model_name),
		):
			if not isinstance(value, str) or not value:
				raise ValueError(f"{name} must be a non-empty string")
		compatibility = self.reference_profile_compatibility
		if isinstance(compatibility, str):
			if not compatibility:
				raise ValueError("reference_profile_compatibility must not be empty")
		elif (
			not isinstance(compatibility, list)
			or not compatibility
			or not all(isinstance(item, str) and item for item in compatibility)
		):
			raise ValueError(
				"reference_profile_compatibility must be a string or non-empty list"
			)
		for name, value in (
			("training_metrics", self.training_metrics),
			("validation_metrics", self.validation_metrics),
			("validation_metrics_by_force_N", self.validation_metrics_by_force_N),
			("training_configuration", self.training_configuration),
		):
			if not isinstance(value, dict):
				raise ValueError(f"{name} must be an object")
		if (
			not isinstance(self.training_dataset_identifiers, list)
			or not all(isinstance(item, str) for item in self.training_dataset_identifiers)
		):
			raise ValueError("training_dataset_identifiers must be a list of strings")

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class ForcePrediction:
	raw_force_N: float | None
	displayed_force_N: float | None
	valid: bool
	model_name: str
	tracking_quality: float
	reference_quality: float
	geometry_quality: float
	warning: str = ""


class ForceModel:
	def __init__(self, estimator: Any, metadata: ModelMetadata):
		if not callable(getattr(estimator, "predict", None)):
			raise TypeError("Loaded model must implement predict()")
		metadata.validate()
		n_features = getattr(estimator, "n_features_in_", None)
		if n_features not in (None, 0) and int(n_features) != len(metadata.feature_names):
			raise ValueError(
				"Estimator feature count does not match model metadata"
			)
		self.estimator = estimator
		self.metadata = metadata

	@classmethod
	def load(
		cls,
		model_path: str | Path,
		metadata_path: str | Path | None = None,
	) -> "ForceModel":
		path = Path(model_path)
		meta_path = Path(metadata_path) if metadata_path else path.with_suffix(".json")
		raw_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
		metadata = ModelMetadata.from_dict(raw_metadata)
		try:
			with path.open("rb") as stream:
				estimator = pickle.load(stream)
			return cls(estimator, metadata)
		except (
			pickle.UnpicklingError,
			EOFError,
			AttributeError,
			ImportError,
			TypeError,
		) as exc:
			raise ValueError(f"Invalid force-model artifact: {exc}") from exc

	def validate_features(
		self,
		feature_names: list[str],
		features: np.ndarray | list[float] | None = None,
	) -> np.ndarray | None:
		if list(feature_names) != self.metadata.feature_names:
			raise ModelCompatibilityError(
				"Feature names/order do not match model metadata"
			)
		if features is None:
			return None
		try:
			vector = np.asarray(features, dtype=np.float64)
		except (TypeError, ValueError) as exc:
			raise ModelCompatibilityError(
				"Feature vector cannot be converted to finite numbers"
			) from exc
		if vector.ndim != 1 or vector.shape != (len(self.metadata.feature_names),):
			raise ModelCompatibilityError(
				f"Feature vector must have shape ({len(self.metadata.feature_names)},)"
			)
		if not np.all(np.isfinite(vector)):
			raise ModelCompatibilityError("Feature vector contains non-finite values")
		return vector

	def validate_context(
		self,
		*,
		reference_profile_id: str,
		geometry_version: str,
		feature_schema_version: str,
	) -> None:
		compatibility = self.metadata.reference_profile_compatibility
		compatible_profiles = (
			[compatibility] if isinstance(compatibility, str) else compatibility
		)
		if "*" not in compatible_profiles and reference_profile_id not in compatible_profiles:
			raise ModelCompatibilityError(
				"Reference profile is incompatible with model metadata"
			)
		if (
			self.metadata.geometry_configuration_version != "*"
			and self.metadata.geometry_configuration_version != str(geometry_version)
		):
			raise ModelCompatibilityError(
				"Geometry configuration is incompatible with model metadata"
			)
		if self.metadata.feature_schema_version != str(feature_schema_version):
			raise ModelCompatibilityError(
				"Feature schema is incompatible with model metadata"
			)

	def predict(
		self,
		features: np.ndarray,
		feature_names: list[str],
		*,
		tracking_quality: float,
		reference_quality: float,
		geometry_quality: float,
		reference_profile_id: str = "",
		geometry_version: str = "1",
		feature_schema_version: str = "1",
	) -> ForcePrediction:
		try:
			vector = self.validate_features(feature_names, features)
			self.validate_context(
				reference_profile_id=reference_profile_id,
				geometry_version=geometry_version,
				feature_schema_version=feature_schema_version,
			)
		except ModelCompatibilityError as exc:
			return ForcePrediction(
				None,
				None,
				False,
				self.metadata.model_name,
				float(tracking_quality),
				float(reference_quality),
				float(geometry_quality),
				str(exc),
			)

		# Estimator exceptions deliberately propagate: unexpected model bugs must not
		# be converted into ordinary compatibility failures.
		output = np.asarray(
			self.estimator.predict(vector.reshape(1, -1)), dtype=np.float64
		)
		if output.size != 1:
			raise ValueError("Estimator predict() must return exactly one value per sample")
		raw = float(output.reshape(-1)[0])
		if not math.isfinite(raw):
			raise ValueError("Estimator predict() returned a non-finite force")

		in_range = (
			float(self.metadata.calibrated_min_force_N)
			<= raw
			<= float(self.metadata.calibrated_max_force_N)
		)
		warning = "" if in_range else "Outside calibrated force range"
		return ForcePrediction(
			raw,
			raw,
			in_range,
			self.metadata.model_name,
			float(tracking_quality),
			float(reference_quality),
			float(geometry_quality),
			warning,
		)
