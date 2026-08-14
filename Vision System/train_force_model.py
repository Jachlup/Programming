"""Train a reproducible NumPy ridge baseline from visual-force sample CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from force_model import ModelMetadata, RidgeRegressor


def _configured_training_defaults() -> dict[str, Any]:
	path = Path(__file__).resolve().with_name("camera-config.yaml")
	with path.open(encoding="utf-8") as stream:
		raw = yaml.safe_load(stream) or {}
	defaults = raw.get("training", {})
	if not isinstance(defaults, dict):
		raise ValueError("camera-config.yaml training section must be a mapping")
	return defaults


@dataclass(frozen=True)
class SampleRecord:
	features: np.ndarray
	known_force_N: float
	feature_names: tuple[str, ...]
	dataset_schema_version: str
	feature_schema_version: str
	reference_profile_id: str
	reference_profile_configuration_version: str
	structural_connection_version: str
	geometry_configuration_version: str
	experiment_id: str
	acquisition_batch_id: str
	force_step_id: str
	source_file: str
	source_row: int

	@property
	def dataset_identifier(self) -> str:
		return f"{self.experiment_id}/{self.acquisition_batch_id}"


@dataclass
class LoadedSamples:
	records: list[SampleRecord]
	source_files: list[Path]
	rejected_reasons: dict[str, int]

	@property
	def rejected_count(self) -> int:
		return sum(self.rejected_reasons.values())


def discover_sample_files(inputs: Iterable[str | Path]) -> list[Path]:
	"""Expand explicit CSV paths and recursively discover samples.csv in directories."""
	discovered: dict[Path, Path] = {}
	for value in inputs:
		path = Path(value).expanduser()
		if not path.exists():
			raise FileNotFoundError(f"Dataset input does not exist: {path}")
		candidates = [path] if path.is_file() else sorted(path.rglob("samples.csv"))
		for candidate in candidates:
			if not candidate.is_file():
				continue
			resolved = candidate.resolve()
			discovered[resolved] = candidate
	if not discovered:
		raise ValueError("No samples.csv files were found")
	return [discovered[key] for key in sorted(discovered, key=str)]


def _parse_bool(value: str, field_name: str) -> bool:
	normalised = str(value).strip().lower()
	if normalised in {"1", "true", "yes", "on"}:
		return True
	if normalised in {"0", "false", "no", "off"}:
		return False
	raise ValueError(f"{field_name} is not a boolean")


def _decode_feature_names(value: str) -> tuple[str, ...]:
	decoded = json.loads(value)
	if (
		not isinstance(decoded, list)
		or not decoded
		or not all(isinstance(name, str) and name for name in decoded)
	):
		raise ValueError("feature_names must be a non-empty JSON string list")
	if len(set(decoded)) != len(decoded):
		raise ValueError("feature_names contains duplicates")
	return tuple(decoded)


def _decode_feature_values(value: str, expected_count: int) -> np.ndarray:
	decoded = json.loads(value)
	vector = np.asarray(decoded, dtype=np.float64)
	if vector.ndim != 1 or vector.shape != (expected_count,):
		raise ValueError(f"feature_values must contain exactly {expected_count} values")
	if not np.all(np.isfinite(vector)):
		raise ValueError("feature_values contains non-finite values")
	return vector


def _increment(counter: dict[str, int], reason: str) -> None:
	counter[reason] = counter.get(reason, 0) + 1


def load_samples(inputs: Iterable[str | Path]) -> LoadedSamples:
	"""Load valid rows while recording frame-level rejection reasons."""
	files = discover_sample_files(inputs)
	records: list[SampleRecord] = []
	rejected: dict[str, int] = {}

	for path in files:
		last_force: float | None = None
		derived_force_step = 0
		with path.open(newline="", encoding="utf-8") as stream:
			reader = csv.DictReader(stream)
			if reader.fieldnames is None:
				raise ValueError(f"{path} has no CSV header")
			required = {"known_force_N", "feature_names", "feature_values"}
			missing = sorted(required - set(reader.fieldnames))
			if missing:
				raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

			for row_number, row in enumerate(reader, start=2):
				try:
					for validity_field in ("tracking_valid", "origin_valid", "geometry_valid"):
						if validity_field in row and not _parse_bool(
							row[validity_field], validity_field
						):
							raise ValueError(f"{validity_field}=false")
					origin_status = row.get("origin_status", "")
					if origin_status in {"MISSING", "INVALID"}:
						raise ValueError(f"origin_status={origin_status}")
					if "reference_origin_point_id" in row and not row[
						"reference_origin_point_id"
					].strip():
						raise ValueError("reference_origin_point_id is empty")

					target = float(row["known_force_N"])
					if not math.isfinite(target):
						raise ValueError("known_force_N is not finite")
					names = _decode_feature_names(row["feature_names"])
					values = _decode_feature_values(row["feature_values"], len(names))

					profile_id = row.get("reference_profile_identifier", "").strip()
					if not profile_id:
						raise ValueError("reference_profile_identifier is empty")
					geometry_version = row.get(
						"geometry_configuration_identifier", "1"
					).strip()
					profile_version = row.get(
						"reference_profile_configuration_version", "1"
					).strip()
					structural_version = row.get(
						"structural_connection_version", "1"
					).strip()
					dataset_schema_version = row.get(
						"dataset_schema_version", "1"
					).strip()
					schema_version = row.get("feature_schema_version", "1").strip()
					if (
						not dataset_schema_version
						or not profile_version
						or not structural_version
						or not geometry_version
						or not schema_version
					):
						raise ValueError("dataset, geometry, or feature schema version is empty")

					experiment_id = row.get("experiment_id", "").strip() or path.parent.parent.name
					batch_id = (
						row.get("acquisition_batch_id", "").strip() or path.parent.name
					)
					if last_force is None or target != last_force:
						derived_force_step += 1
						last_force = target
					force_step_id = row.get("force_step_id", "").strip()
					if not force_step_id:
						force_step_id = f"derived-{derived_force_step:04d}"

					records.append(SampleRecord(
						features=values,
						known_force_N=target,
						feature_names=names,
						dataset_schema_version=dataset_schema_version,
						feature_schema_version=schema_version,
						reference_profile_id=profile_id,
						reference_profile_configuration_version=profile_version,
						structural_connection_version=structural_version,
						geometry_configuration_version=geometry_version,
						experiment_id=experiment_id,
						acquisition_batch_id=batch_id,
						force_step_id=force_step_id,
						source_file=str(path),
						source_row=row_number,
					))
				except (json.JSONDecodeError, TypeError, ValueError) as exc:
					_increment(rejected, str(exc))

	if not records:
		detail = ", ".join(f"{reason}: {count}" for reason, count in sorted(rejected.items()))
		raise ValueError(f"No valid training samples were found ({detail})")
	return LoadedSamples(records, files, rejected)


def validate_sample_compatibility(
	records: Sequence[SampleRecord],
	*,
	allow_multiple_profiles: bool = False,
) -> tuple[list[str], str, str, str, list[str]]:
	"""Validate the ordered feature and version contract across every sample."""
	if not records:
		raise ValueError("No samples were supplied")
	feature_names = records[0].feature_names
	dataset_schema_version = records[0].dataset_schema_version
	schema_version = records[0].feature_schema_version
	geometry_version = records[0].geometry_configuration_version
	profile_version = records[0].reference_profile_configuration_version
	structural_version = records[0].structural_connection_version
	profiles: set[str] = set()
	for record in records:
		location = f"{record.source_file}:{record.source_row}"
		if record.feature_names != feature_names:
			raise ValueError(f"Feature names/order differ at {location}")
		if record.dataset_schema_version != dataset_schema_version:
			raise ValueError(f"Dataset schema version differs at {location}")
		if record.feature_schema_version != schema_version:
			raise ValueError(f"Feature schema version differs at {location}")
		if record.geometry_configuration_version != geometry_version:
			raise ValueError(f"Geometry configuration version differs at {location}")
		if record.reference_profile_configuration_version != profile_version:
			raise ValueError(f"Reference profile configuration version differs at {location}")
		if record.structural_connection_version != structural_version:
			raise ValueError(f"Structural connection version differs at {location}")
		profiles.add(record.reference_profile_id)
	if len(profiles) > 1 and not allow_multiple_profiles:
		raise ValueError(
			"Multiple reference profiles were found; use --allow-multiple-profiles "
			"only when they are known to be physically compatible"
		)
	return (
		list(feature_names),
		dataset_schema_version,
		schema_version,
		geometry_version,
		sorted(profiles),
	)


def _candidate_groups(
	records: Sequence[SampleRecord], grouping: str
) -> tuple[str, list[str]]:
	def values(mode: str) -> list[str]:
		if mode == "experiment":
			return [record.experiment_id for record in records]
		if mode == "acquisition_batch":
			return [
				f"{record.experiment_id}/{record.acquisition_batch_id}"
				for record in records
			]
		if mode == "force_step":
			return [
				f"{record.experiment_id}/{record.acquisition_batch_id}/{record.force_step_id}"
				for record in records
			]
		raise ValueError(f"Unsupported split grouping: {mode}")

	if grouping != "auto":
		return grouping, values(grouping)
	for mode in ("experiment", "acquisition_batch", "force_step"):
		group_values = values(mode)
		if len(set(group_values)) >= 2:
			return mode, group_values
	raise ValueError(
		"A leakage-safe split needs at least two experiments, acquisition batches, "
		"or force steps"
	)


def grouped_train_validation_split(
	records: Sequence[SampleRecord],
	*,
	validation_fraction: float,
	random_seed: int,
	grouping: str = "auto",
) -> tuple[list[SampleRecord], list[SampleRecord], str, list[str]]:
	"""Split whole acquisition groups so adjacent frames never cross the boundary."""
	if not 0.0 < validation_fraction < 1.0:
		raise ValueError("validation_fraction must be between 0 and 1")
	mode, row_groups = _candidate_groups(records, grouping)
	unique_groups = sorted(set(row_groups))
	if len(unique_groups) < 2:
		raise ValueError(f"Grouping by {mode} produced fewer than two groups")

	rng = np.random.default_rng(int(random_seed))
	shuffled = list(unique_groups)
	rng.shuffle(shuffled)
	target_validation_rows = max(1, int(round(len(records) * validation_fraction)))
	group_sizes = {
		group: sum(value == group for value in row_groups) for group in unique_groups
	}
	validation_groups: list[str] = []
	validation_rows = 0
	for group in shuffled:
		if len(validation_groups) >= len(unique_groups) - 1:
			break
		validation_groups.append(group)
		validation_rows += group_sizes[group]
		if validation_rows >= target_validation_rows:
			break

	validation_set = set(validation_groups)
	training = [
		record for record, group in zip(records, row_groups)
		if group not in validation_set
	]
	validation = [
		record for record, group in zip(records, row_groups)
		if group in validation_set
	]
	if not training or not validation:
		raise ValueError("Grouped split produced an empty training or validation set")
	return training, validation, mode, sorted(validation_groups)


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
	y = np.asarray(targets, dtype=np.float64).reshape(-1)
	predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
	if y.shape != predicted.shape or not y.size:
		raise ValueError("Metric target and prediction arrays are incompatible")
	if not np.all(np.isfinite(y)) or not np.all(np.isfinite(predicted)):
		raise ValueError("Metrics require finite targets and predictions")
	errors = predicted - y
	absolute = np.abs(errors)
	residual_sum = float(np.sum(errors ** 2))
	total_sum = float(np.sum((y - y.mean()) ** 2))
	if total_sum <= np.finfo(np.float64).eps:
		r2 = 1.0 if residual_sum <= np.finfo(np.float64).eps else 0.0
	else:
		r2 = 1.0 - residual_sum / total_sum
	mae = float(np.mean(absolute))
	rmse = float(np.sqrt(np.mean(errors ** 2)))
	maximum_absolute_error = float(np.max(absolute))
	return {
		"sample_count": int(y.size),
		"mae": mae,
		"rmse": rmse,
		"r2": float(r2),
		"max_absolute_error": maximum_absolute_error,
		# Unit-labelled aliases make the JSON self-describing for human readers.
		"mae_N": mae,
		"rmse_N": rmse,
		"maximum_absolute_error_N": maximum_absolute_error,
	}


def per_force_metrics(
	targets: np.ndarray, predictions: np.ndarray
) -> dict[str, dict[str, Any]]:
	y = np.asarray(targets, dtype=np.float64).reshape(-1)
	predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
	return {
		f"{force:.12g}": regression_metrics(y[y == force], predicted[y == force])
		for force in sorted(set(float(value) for value in y))
	}


def _matrix(records: Sequence[SampleRecord]) -> tuple[np.ndarray, np.ndarray]:
	return (
		np.vstack([record.features for record in records]),
		np.asarray([record.known_force_N for record in records], dtype=np.float64),
	)


def save_model_artifacts(
	estimator: RidgeRegressor,
	metadata: ModelMetadata,
	*,
	model_path: str | Path,
	metadata_path: str | Path | None = None,
	overwrite: bool = False,
) -> tuple[Path, Path]:
	model_output = Path(model_path)
	metadata_output = (
		Path(metadata_path) if metadata_path is not None else model_output.with_suffix(".json")
	)
	if model_output.resolve() == metadata_output.resolve():
		raise ValueError("Model and metadata outputs must be different paths")
	existing = [path for path in (model_output, metadata_output) if path.exists()]
	if existing and not overwrite:
		raise FileExistsError(
			"Refusing to overwrite existing output: "
			+ ", ".join(str(path) for path in existing)
		)
	model_output.parent.mkdir(parents=True, exist_ok=True)
	metadata_output.parent.mkdir(parents=True, exist_ok=True)
	with model_output.open("wb") as stream:
		pickle.dump(estimator, stream, protocol=pickle.HIGHEST_PROTOCOL)
	metadata_output.write_text(
		json.dumps(metadata.to_dict(), indent=2, allow_nan=False) + "\n",
		encoding="utf-8",
	)
	return model_output, metadata_output


def train(
	inputs: Iterable[str | Path],
	*,
	model_path: str | Path,
	metadata_path: str | Path | None = None,
	validation_fraction: float = 0.2,
	random_seed: int = 42,
	ridge_alpha: float = 1.0,
	split_group: str = "auto",
	model_name: str = "force-ridge",
	model_version: str = "1",
	allow_multiple_profiles: bool = False,
	overwrite: bool = False,
) -> tuple[RidgeRegressor, ModelMetadata, dict[str, Any]]:
	loaded = load_samples(inputs)
	feature_names, dataset_schema_version, schema_version, geometry_version, profiles = (
		validate_sample_compatibility(
			loaded.records, allow_multiple_profiles=allow_multiple_profiles
		)
	)
	training, validation, actual_grouping, validation_groups = (
		grouped_train_validation_split(
			loaded.records,
			validation_fraction=validation_fraction,
			random_seed=random_seed,
			grouping=split_group,
		)
	)
	_grouping, all_groups = _candidate_groups(loaded.records, actual_grouping)
	training_groups = sorted(set(all_groups) - set(validation_groups))
	train_x, train_y = _matrix(training)
	validation_x, validation_y = _matrix(validation)
	estimator = RidgeRegressor(alpha=float(ridge_alpha)).fit(train_x, train_y)
	train_prediction = estimator.predict(train_x)
	validation_prediction = estimator.predict(validation_x)
	training_metrics = regression_metrics(train_y, train_prediction)
	validation_metrics = regression_metrics(validation_y, validation_prediction)
	by_force = per_force_metrics(validation_y, validation_prediction)

	dataset_ids = sorted({record.dataset_identifier for record in loaded.records})
	profile_compatibility: str | list[str] = (
		profiles[0] if len(profiles) == 1 else profiles
	)
	metadata = ModelMetadata(
		model_type="numpy_ridge",
		feature_names=feature_names,
		calibrated_min_force_N=float(min(record.known_force_N for record in loaded.records)),
		calibrated_max_force_N=float(max(record.known_force_N for record in loaded.records)),
		training_date=datetime.now(timezone.utc).isoformat(),
		model_version=str(model_version),
		reference_profile_compatibility=profile_compatibility,
		geometry_configuration_version=geometry_version,
		feature_schema_version=schema_version,
		training_metrics=training_metrics,
		validation_metrics=validation_metrics,
		validation_metrics_by_force_N=by_force,
		training_dataset_identifiers=dataset_ids,
		training_configuration={
			"ridge_alpha": float(ridge_alpha),
			"random_seed": int(random_seed),
			"validation_fraction": float(validation_fraction),
			"split_group": actual_grouping,
			"training_groups": training_groups,
			"validation_groups": validation_groups,
			"training_sample_count": len(training),
			"validation_sample_count": len(validation),
			"rejected_sample_count": loaded.rejected_count,
			"source_files": [str(path) for path in loaded.source_files],
			"dataset_schema_version": dataset_schema_version,
			"reference_profile_configuration_version":
				loaded.records[0].reference_profile_configuration_version,
			"structural_connection_version":
				loaded.records[0].structural_connection_version,
		},
		model_name=str(model_name),
	)
	metadata.validate()
	outputs = save_model_artifacts(
		estimator,
		metadata,
		model_path=model_path,
		metadata_path=metadata_path,
		overwrite=overwrite,
	)
	report = {
		"model_path": str(outputs[0]),
		"metadata_path": str(outputs[1]),
		"training_metrics": training_metrics,
		"validation_metrics": validation_metrics,
		"validation_metrics_by_force_N": by_force,
		"training_groups": training_groups,
		"validation_groups": validation_groups,
		"rejected_reasons": loaded.rejected_reasons,
	}
	return estimator, metadata, report


def build_argument_parser() -> argparse.ArgumentParser:
	defaults = _configured_training_defaults()
	parser = argparse.ArgumentParser(
		description=(
			"Train a leakage-safe NumPy ridge force regressor from one or more "
			"samples.csv files or dataset directories."
		)
	)
	parser.add_argument("datasets", nargs="+", help="samples.csv file(s) or directory roots")
	parser.add_argument("--model-out", required=True, help="Output estimator pickle")
	parser.add_argument(
		"--metadata-out",
		help="Output metadata JSON (default: model path with .json suffix)",
	)
	parser.add_argument(
		"--validation-fraction",
		type=float,
		default=float(defaults.get("validation_fraction", 0.2)),
	)
	parser.add_argument(
		"--random-seed",
		type=int,
		default=int(defaults.get("random_seed", 42)),
	)
	parser.add_argument(
		"--ridge-alpha",
		type=float,
		default=float(defaults.get("ridge_alpha", 1.0)),
	)
	parser.add_argument(
		"--split-group",
		choices=("auto", "experiment", "acquisition_batch", "force_step"),
		default="auto",
	)
	parser.add_argument("--model-name", default=str(defaults.get("model_name", "force-ridge")))
	parser.add_argument("--model-version", default=str(defaults.get("model_version", "1")))
	parser.add_argument(
		"--allow-multiple-profiles",
		action="store_true",
		help="Accept multiple profiles only when they are known to be compatible",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Explicitly permit replacing existing model/metadata outputs",
	)
	return parser


def _format_metric(value: Any) -> str:
	return str(value) if isinstance(value, int) else f"{float(value):.6g}"


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
	print(label)
	for name in ("sample_count", "mae_N", "rmse_N", "r2", "maximum_absolute_error_N"):
		print(f"  {name}: {_format_metric(metrics[name])}")


def main(argv: Sequence[str] | None = None) -> int:
	args = build_argument_parser().parse_args(argv)
	try:
		_, _, report = train(
			args.datasets,
			model_path=args.model_out,
			metadata_path=args.metadata_out,
			validation_fraction=args.validation_fraction,
			random_seed=args.random_seed,
			ridge_alpha=args.ridge_alpha,
			split_group=args.split_group,
			model_name=args.model_name,
			model_version=args.model_version,
			allow_multiple_profiles=args.allow_multiple_profiles,
			overwrite=args.overwrite,
		)
	except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
		print(f"Training failed: {exc}", file=sys.stderr)
		return 2

	_print_metrics("Training metrics:", report["training_metrics"])
	_print_metrics("Validation metrics:", report["validation_metrics"])
	if report["validation_metrics_by_force_N"]:
		print("Validation errors by force (N):")
		for force, metrics in report["validation_metrics_by_force_N"].items():
			print(
				f"  {force}: MAE={metrics['mae_N']:.6g}, "
				f"RMSE={metrics['rmse_N']:.6g}, "
				f"max={metrics['maximum_absolute_error_N']:.6g}, "
				f"n={metrics['sample_count']}"
			)
	if report["rejected_reasons"]:
		print("Rejected input rows:")
		for reason, count in sorted(report["rejected_reasons"].items()):
			print(f"  {reason}: {count}")
	print(f"Saved model: {report['model_path']}")
	print(f"Saved metadata: {report['metadata_path']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
