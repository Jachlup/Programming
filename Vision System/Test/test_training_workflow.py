"""Hardware-free dataset audit, process training, and model activation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QObject, QProcess, Signal
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application import ApplicationState
from commands import execute_command
from dataset_audit import audit_dataset
from force_model import ForceModel, ModelCompatibilityError, ModelMetadata, RidgeRegressor
from model_registry import ModelRegistry
from processing import config_from_mapping, config_to_mapping, cfg
from train_force_model import save_model_artifacts
from training_control import TrainingController, TrainingState


def _dataset(tmp_path: Path) -> Path:
	root = tmp_path / "experiment-a" / "batch-a"
	root.mkdir(parents=True)
	header = [
		"dataset_schema_version", "feature_schema_version", "experiment_id",
		"acquisition_batch_id", "force_step_id", "loading_direction", "repetition",
		"known_force_N", "tracking_valid", "tracking_quality", "origin_valid",
		"origin_status", "origin_tracking_quality", "reference_origin_point_id",
		"geometry_valid", "geometry_quality", "feature_names", "feature_values",
		"reference_profile_identifier", "reference_profile_configuration_version",
		"structural_connection_version", "geometry_configuration_identifier",
	]
	with (root / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=header)
		writer.writeheader()
		for index, (force, step, direction, features) in enumerate((
			(0.0, "baseline-r01-s001", "baseline", [0.0, 0.0]),
			(0.0, "baseline-r01-s001", "baseline", [0.1, 0.0]),
			(10.0, "loading-r01-s002", "loading", [1.0, 1.0]),
			(10.0, "loading-r01-s002", "loading", [1.1, 1.0]),
		), start=1):
			writer.writerow({
				"dataset_schema_version": "3",
				"feature_schema_version": "test-feature",
				"experiment_id": "experiment-a",
				"acquisition_batch_id": "batch-a",
				"force_step_id": step,
				"loading_direction": direction,
				"repetition": 1,
				"known_force_N": force,
				"tracking_valid": True,
				"tracking_quality": 0.9 - index * 0.01,
				"origin_valid": True,
				"origin_status": "DETECTED",
				"origin_tracking_quality": 0.95,
				"reference_origin_point_id": "REFERENCE_ORIGIN",
				"geometry_valid": True,
				"geometry_quality": 0.92,
				"feature_names": json.dumps(["x", "y"]),
				"feature_values": json.dumps(features),
				"reference_profile_identifier": "profile-a",
				"reference_profile_configuration_version": "test-profile",
				"structural_connection_version": "test-structure",
				"geometry_configuration_identifier": "test-geometry",
			})
	with (root / "rejections.csv").open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=("timestamp", "frame_number", "reason"))
		writer.writeheader()
		writer.writerow({"timestamp": 1, "frame_number": 1, "reason": "tracking invalid"})
	return tmp_path


def test_dataset_audit_reports_coverage_quality_groups_and_rejections(tmp_path: Path) -> None:
	audit = audit_dataset([_dataset(tmp_path)])
	report = audit.report
	assert audit.training_allowed
	assert report["accepted_rows"] == 4
	assert report["rejected_rows"] == 1
	assert report["samples_per_force_N"] == {"0.0": 2, "10.0": 2}
	assert report["direction_coverage"] == {"baseline": 2, "loading": 2}
	assert report["independent_group_counts"]["force_steps"] == 2
	assert report["quality_distributions"]["tracking_quality"]["count"] == 4
	assert report["force_range_N"] == [0.0, 10.0]
	assert "No unloading samples were found" in report["warnings"]


class _ApplicationController(QObject):
	model_activation_completed = Signal(str, bool, str)

	def __init__(self):
		super().__init__()
		self.activations = []

	def activate_force_model(self, model_path, metadata_path=None):
		self.activations.append((model_path, metadata_path))
		self.model_activation_completed.emit(model_path, True, "activated")


def test_training_runs_in_qprocess_and_requires_explicit_overwrite(
	tmp_path: Path,
	qtbot,
) -> None:
	dataset = _dataset(tmp_path / "data")
	application = _ApplicationController()
	controller = TrainingController(application)
	controller.registry = ModelRegistry(tmp_path / "registry.json")
	model_path = tmp_path / "model.pkl"
	try:
		controller.audit(str(dataset))
		qtbot.waitUntil(
			lambda: controller.state == TrainingState.AUDIT_READY,
			timeout=10_000,
		)
		assert controller.audit_report["training_allowed"]

		controller.train({
			"model_path": str(model_path),
			"model_name": "gui-ridge",
			"model_version": "7",
			"validation_fraction": 0.5,
			"ridge_alpha": 1e-6,
			"random_seed": 7,
			"split_group": "force_step",
			"overwrite": False,
		})
		assert controller.process.program() == sys.executable
		qtbot.waitUntil(
			lambda: controller.state == TrainingState.MODEL_REVIEW,
			timeout=10_000,
		)
		assert model_path.exists()
		metadata = controller.training_report["metadata"]
		assert metadata["model_name"] == "gui-ridge"
		assert "mae_N" in metadata["validation_metrics"]
		assert metadata["validation_metrics_by_force_N"]
		assert metadata["training_configuration"]["training_groups"]
		assert metadata["training_configuration"]["validation_groups"]

		controller.accept_model()
		assert controller.state == TrainingState.ACCEPTED
		assert application.activations[-1][0] == str(model_path.resolve())
		assert controller.registry.load()["active_model"] is not None

		controller.state = TrainingState.AUDIT_READY
		controller.train({"model_path": str(model_path), "overwrite": False})
		assert controller.state == TrainingState.FAULT
		assert "overwrite approval" in controller._message.lower()
	finally:
		assert controller.shutdown()


def test_training_process_can_be_cancelled_without_blocking_gui(qtbot, tmp_path: Path) -> None:
	controller = TrainingController(_ApplicationController())
	controller.registry = ModelRegistry(tmp_path / "registry.json")
	try:
		controller._set_state(TrainingState.TRAINING, "test process")
		controller._start_process(
			"training", sys.executable, ["-c", "import time; time.sleep(30)"]
		)
		assert controller.process.state() != QProcess.ProcessState.NotRunning
		controller.cancel()
		qtbot.waitUntil(
			lambda: controller.state == TrainingState.CANCELLED,
			timeout=5_000,
		)
	finally:
		assert controller.shutdown()


def _model_artifacts(
	root: Path,
	name: str,
	profile: str,
	geometry: str,
	feature_schema: str,
) -> tuple[Path, Path]:
	estimator = RidgeRegressor(alpha=1.0).fit(
		np.asarray([[0.0, 0.0], [1.0, 1.0]]),
		np.asarray([0.0, 1.0]),
	)
	metadata = ModelMetadata(
		model_type="numpy_ridge",
		feature_names=["x", "y"],
		calibrated_min_force_N=0.0,
		calibrated_max_force_N=1.0,
		model_name=name,
		model_version="1",
		reference_profile_compatibility=profile,
		geometry_configuration_version=geometry,
		feature_schema_version=feature_schema,
	)
	return save_model_artifacts(estimator, metadata, model_path=root / f"{name}.pkl")


def test_incompatible_activation_keeps_the_previous_model(tmp_path: Path) -> None:
	config = config_from_mapping(config_to_mapping(cfg))
	geometry = str(config.geometry.get("geometry_configuration_version", "2"))
	feature_schema = str(config.geometry.get("feature_schema_version", "2"))
	valid_model, valid_metadata = _model_artifacts(
		tmp_path, "valid", "profile-a", geometry, feature_schema
	)
	bad_model, bad_metadata = _model_artifacts(
		tmp_path, "bad", "different-profile", geometry, feature_schema
	)
	state = ApplicationState(config=config)
	state.reference_profile = type("Profile", (), {"profile_id": "profile-a"})()
	state.current_feature_names = ["x", "y"]
	previous = ForceModel.load(valid_model, valid_metadata)
	state.loaded_force_model = previous
	with pytest.raises(ModelCompatibilityError):
		state.activate_force_model(bad_model, bad_metadata)
	assert state.loaded_force_model is previous


def test_configured_model_paths_are_used_without_overriding_artifact_range(
	tmp_path: Path,
) -> None:
	config = config_from_mapping(config_to_mapping(cfg))
	model_path, metadata_path = _model_artifacts(
		tmp_path,
		"configured",
		"*",
		str(config.geometry.get("geometry_configuration_version", "2")),
		str(config.geometry.get("feature_schema_version", "2")),
	)
	config.force_model.update({
		"model_path": str(model_path),
		"metadata_path": str(metadata_path),
		"minimum_calibrated_force_N": -999.0,
		"maximum_calibrated_force_N": 999.0,
	})
	state = ApplicationState(config=config)
	assert execute_command("model_load", state)
	assert state.loaded_force_model.metadata.calibrated_min_force_N == 0.0
	assert state.loaded_force_model.metadata.calibrated_max_force_N == 1.0
	notes = state.status_snapshot()["configuration_compatibility_notes"]
	assert any("minimum_valid_points" in note for note in notes)
	assert any("model metadata is authoritative" in note for note in notes)


def test_model_registry_can_roll_back_to_the_prior_active_model(tmp_path: Path) -> None:
	registry = ModelRegistry(tmp_path / "registry.json")
	metadata_a = ModelMetadata(model_type="test", feature_names=["x"], model_name="a")
	metadata_b = ModelMetadata(model_type="test", feature_names=["x"], model_name="b")
	id_a = registry.register(tmp_path / "a.pkl", tmp_path / "a.json", metadata_a)
	id_b = registry.register(tmp_path / "b.pkl", tmp_path / "b.json", metadata_b)
	registry.mark_active(id_a)
	registry.mark_active(id_b)
	assert registry.previous_model()["id"] == id_a
	entry = registry.mark_rollback()
	assert entry["id"] == id_a
	assert registry.load()["active_model"] == id_a
