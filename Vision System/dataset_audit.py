"""Read-only dataset audit used by the collection and training interfaces."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from train_force_model import (
	discover_sample_files,
	load_samples,
	validate_sample_compatibility,
)


@dataclass(frozen=True)
class DatasetAudit:
	report: dict[str, Any]

	@property
	def training_allowed(self) -> bool:
		return bool(self.report.get("training_allowed"))

	@property
	def errors(self) -> list[str]:
		return list(self.report.get("errors", []))

	@property
	def warnings(self) -> list[str]:
		return list(self.report.get("warnings", []))


def _summary(values: Sequence[float]) -> dict[str, float | int]:
	array = np.asarray(values, dtype=np.float64)
	array = array[np.isfinite(array)]
	if not array.size:
		return {"count": 0}
	return {
		"count": int(array.size),
		"minimum": float(np.min(array)),
		"p05": float(np.percentile(array, 5)),
		"median": float(np.median(array)),
		"mean": float(np.mean(array)),
		"p95": float(np.percentile(array, 95)),
		"maximum": float(np.max(array)),
	}


def _read_progress(files: Sequence[Path]) -> dict[str, Any]:
	progress_files = sorted({
		path.parent / "collection-progress.yaml" for path in files
		if (path.parent / "collection-progress.yaml").is_file()
	})
	if not progress_files:
		return {}
	planned: list[str] = []
	completed: list[str] = []
	skipped: list[str] = []
	states: list[str] = []
	for path in progress_files:
		with path.open(encoding="utf-8") as stream:
			mapping = yaml.safe_load(stream) or {}
		if not isinstance(mapping, dict):
			continue
		planned.extend(str(value) for value in mapping.get("planned_step_ids", []))
		completed.extend(str(value) for value in mapping.get("completed_step_ids", []))
		skipped.extend(str(value) for value in mapping.get("skipped_step_ids", []))
		states.append(str(mapping.get("state", "")))
	return {
		"files": [str(path) for path in progress_files],
		"planned_step_ids": sorted(set(planned)),
		"completed_step_ids": sorted(set(completed)),
		"skipped_step_ids": sorted(set(skipped)),
		"states": states,
	}


def audit_dataset(inputs: Iterable[str | Path]) -> DatasetAudit:
	files = discover_sample_files(inputs)
	errors: list[str] = []
	warnings: list[str] = []
	accepted_rows = 0
	empty_step_rows = 0
	force_counts: dict[str, int] = {}
	batch_counts: dict[str, int] = {}
	repetition_counts: dict[str, int] = {}
	direction_counts: dict[str, int] = {}
	quality_values: dict[str, list[float]] = {
		"tracking_quality": [],
		"origin_tracking_quality": [],
		"geometry_quality": [],
	}
	step_contracts: dict[str, set[tuple[str, str, str, str]]] = {}
	step_batches: dict[str, set[str]] = {}
	dataset_schemas: set[str] = set()
	feature_schemas: set[str] = set()
	geometry_schemas: set[str] = set()
	rejection_reasons: dict[str, int] = {}

	for path in files:
		with path.open(newline="", encoding="utf-8") as stream:
			reader = csv.DictReader(stream)
			for row in reader:
				accepted_rows += 1
				force = str(row.get("known_force_N", "")).strip()
				force_counts[force] = force_counts.get(force, 0) + 1
				experiment = str(row.get("experiment_id", path.parent.parent.name)).strip()
				batch = str(row.get("acquisition_batch_id", path.parent.name)).strip()
				batch_key = f"{experiment}/{batch}"
				batch_counts[batch_key] = batch_counts.get(batch_key, 0) + 1
				repetition = str(row.get("repetition", "1")).strip() or "1"
				repetition_key = f"{batch_key}/r{repetition}"
				repetition_counts[repetition_key] = repetition_counts.get(repetition_key, 0) + 1
				direction = str(row.get("loading_direction", "unspecified")).strip().lower()
				direction_counts[direction] = direction_counts.get(direction, 0) + 1
				step_id = str(row.get("force_step_id", "")).strip()
				if not step_id:
					empty_step_rows += 1
				else:
					step_contracts.setdefault(step_id, set()).add(
						(force, direction, repetition, batch_key)
					)
					step_batches.setdefault(step_id, set()).add(batch_key)
				for field in quality_values:
					value = row.get(field, "")
					try:
						number = float(value)
					except (TypeError, ValueError, OverflowError):
						continue
					if math.isfinite(number):
						quality_values[field].append(number)
				dataset_schemas.add(str(row.get("dataset_schema_version", "")).strip())
				feature_schemas.add(str(row.get("feature_schema_version", "")).strip())
				geometry_schemas.add(
					str(row.get("geometry_configuration_identifier", "")).strip()
				)

		rejection_file = path.parent / "rejections.csv"
		if rejection_file.is_file():
			with rejection_file.open(newline="", encoding="utf-8") as stream:
				for row in csv.DictReader(stream):
					reason = str(row.get("reason", "unspecified")).strip() or "unspecified"
					rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

	try:
		loaded = load_samples(files)
	except ValueError as exc:
		errors.append(str(exc))
		loaded = None
	if loaded is not None:
		for reason, count in loaded.rejected_reasons.items():
			rejection_reasons[f"training row: {reason}"] = count
		try:
			feature_names, dataset_schema, feature_schema, geometry_schema, profiles = (
				validate_sample_compatibility(loaded.records)
			)
		except ValueError as exc:
			errors.append(str(exc))
			feature_names, dataset_schema, feature_schema, geometry_schema, profiles = (
				[], "", "", "", []
			)
	else:
		feature_names, dataset_schema, feature_schema, geometry_schema, profiles = (
			[], "", "", "", []
		)

	if empty_step_rows:
		errors.append(f"{empty_step_rows} accepted row(s) have an empty force_step_id")
	inconsistent_steps = sorted(
		step for step, contracts in step_contracts.items()
		if len({contract[:3] for contract in contracts}) > 1
	)
	if inconsistent_steps:
		errors.append(
			"Step IDs have inconsistent force/direction/repetition labels: "
			+ ", ".join(inconsistent_steps)
		)
	duplicate_batch_steps = sorted(
		step for step, batches in step_batches.items() if len(batches) > 1
	)
	if duplicate_batch_steps:
		warnings.append(
			"Step IDs occur in multiple acquisition batches: "
			+ ", ".join(duplicate_batch_steps)
		)
	for label, versions in (
		("dataset", dataset_schemas),
		("feature", feature_schemas),
		("geometry", geometry_schemas),
	):
		clean = {value for value in versions if value}
		if len(clean) > 1:
			errors.append(f"Multiple {label} schema versions were found: {sorted(clean)}")

	force_levels = (
		[] if loaded is None else sorted({record.known_force_N for record in loaded.records})
	)
	if len(force_levels) < 2:
		errors.append("At least two distinct force levels are required for training")
	groups = {
		"experiments": 0,
		"acquisition_batches": len(batch_counts),
		"force_steps": len(step_contracts),
	}
	if loaded is not None:
		groups["experiments"] = len({record.experiment_id for record in loaded.records})
	if max(groups.values(), default=0) < 2:
		errors.append("At least two independent split groups are required for training")
	if not direction_counts.get("loading"):
		warnings.append("No loading samples were found")
	if not direction_counts.get("unloading"):
		warnings.append("No unloading samples were found")

	progress = _read_progress(files)
	planned_ids = set(progress.get("planned_step_ids", []))
	recorded_ids = set(step_contracts)
	missing_planned = sorted(planned_ids - recorded_ids)
	unexpected_steps = sorted(recorded_ids - planned_ids) if planned_ids else []
	if missing_planned:
		warnings.append(
			"Planned steps without accepted rows: " + ", ".join(missing_planned)
		)

	report = {
		"training_allowed": not errors,
		"errors": errors,
		"warnings": warnings,
		"source_files": [str(path) for path in files],
		"accepted_rows": accepted_rows,
		"rejected_rows": sum(rejection_reasons.values()),
		"rejection_reasons": dict(sorted(rejection_reasons.items())),
		"samples_per_force_N": dict(sorted(force_counts.items(), key=lambda item: item[0])),
		"samples_per_acquisition_batch": dict(sorted(batch_counts.items())),
		"samples_per_repetition": dict(sorted(repetition_counts.items())),
		"direction_coverage": dict(sorted(direction_counts.items())),
		"quality_distributions": {
			name: _summary(values) for name, values in quality_values.items()
		},
		"force_step_ids": sorted(recorded_ids),
		"missing_force_step_rows": empty_step_rows,
		"inconsistent_force_step_ids": inconsistent_steps,
		"step_ids_in_multiple_batches": duplicate_batch_steps,
		"planned_step_ids": sorted(planned_ids),
		"completed_step_ids": progress.get("completed_step_ids", []),
		"skipped_step_ids": progress.get("skipped_step_ids", []),
		"missing_planned_step_ids": missing_planned,
		"unexpected_step_ids": unexpected_steps,
		"independent_group_counts": groups,
		"feature_count": len(feature_names),
		"feature_names": feature_names,
		"dataset_schema_version": dataset_schema,
		"feature_schema_version": feature_schema,
		"geometry_configuration_version": geometry_schema,
		"reference_profiles": profiles,
		"force_range_N": (
			None if not force_levels else [min(force_levels), max(force_levels)]
		),
	}
	return DatasetAudit(report)


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Audit force-training datasets")
	parser.add_argument("datasets", nargs="+")
	parser.add_argument("--json-out")
	args = parser.parse_args(argv)
	try:
		audit = audit_dataset(args.datasets)
	except (FileNotFoundError, OSError, ValueError) as exc:
		print(f"Dataset audit failed: {exc}")
		return 2
	text = json.dumps(audit.report, indent=2, allow_nan=False) + "\n"
	if args.json_out:
		Path(args.json_out).write_text(text, encoding="utf-8")
	else:
		print(text, end="")
	return 0 if audit.training_allowed else 3


if __name__ == "__main__":
	raise SystemExit(main())
