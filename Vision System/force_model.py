"""Common inference-only interface for geometric force regressors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
	model_type: str
	feature_names: list[str]
	calibrated_min_force_N: float = 5.0
	calibrated_max_force_N: float = 40.0
	training_date: str = ""
	model_version: str = "1"
	reference_profile_compatibility: str = "*"
	geometry_configuration_version: str = "1"
	model_name: str = "force-regressor"


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
		if not hasattr(estimator, "predict"):
			raise TypeError("Loaded model must implement predict()")
		self.estimator = estimator
		self.metadata = metadata

	@classmethod
	def load(cls, model_path: str | Path, metadata_path: str | Path | None = None) -> "ForceModel":
		path = Path(model_path)
		with path.open("rb") as stream:
			estimator = pickle.load(stream)
		meta_path = Path(metadata_path) if metadata_path else path.with_suffix(".json")
		metadata = ModelMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
		return cls(estimator, metadata)

	def validate_features(self, feature_names: list[str]) -> None:
		if feature_names != self.metadata.feature_names:
			raise ValueError("Feature names/order do not match model metadata")

	def predict(self, features: np.ndarray, feature_names: list[str], *,
				tracking_quality: float, reference_quality: float,
				geometry_quality: float, reference_profile_id: str = "",
				geometry_version: str = "1") -> ForcePrediction:
		try:
			self.validate_features(feature_names)
			if (self.metadata.reference_profile_compatibility not in ("*", reference_profile_id)
					or self.metadata.geometry_configuration_version != geometry_version):
				raise ValueError("Reference profile or geometry configuration is incompatible")
			raw = float(np.asarray(self.estimator.predict(np.asarray(features).reshape(1, -1))).reshape(-1)[0])
			in_range = self.metadata.calibrated_min_force_N <= raw <= self.metadata.calibrated_max_force_N
			warning = "" if in_range else "Outside calibrated force range"
			return ForcePrediction(raw, raw, in_range, self.metadata.model_name,
								   tracking_quality, reference_quality, geometry_quality, warning)
		except (ValueError, TypeError) as exc:
			return ForcePrediction(None, None, False, self.metadata.model_name,
								   tracking_quality, reference_quality, geometry_quality, str(exc))
