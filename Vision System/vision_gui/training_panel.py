"""Dataset audit and cancellable force-model training controls."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QPlainTextEdit,
	QProgressBar,
	QPushButton,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from processing import cfg
from training_control import TrainingState

from .common import StatusBadge


class TrainingPanel(QWidget):
	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		base = Path(__file__).resolve().parents[1]
		defaults = cfg.training

		self.dataset = QLineEdit(str(base / cfg.dataset.get("directory", "datasets")))
		self.dataset_browse = QPushButton("Browse…")
		self.audit = QPushButton("Audit dataset")
		self.dataset_browse.clicked.connect(self._browse_dataset)
		self.audit.clicked.connect(lambda: controller.audit(self.dataset.text()))
		dataset_row = QHBoxLayout()
		dataset_row.addWidget(self.dataset, 1)
		dataset_row.addWidget(self.dataset_browse)
		dataset_row.addWidget(self.audit)

		self.output = QLineEdit(str(base / "models" / "force-ridge-v1.pkl"))
		self.output_browse = QPushButton("Browse…")
		self.output_browse.clicked.connect(self._browse_output)
		output_row = QHBoxLayout()
		output_row.addWidget(self.output, 1)
		output_row.addWidget(self.output_browse)

		self.model_name = QLineEdit(str(defaults.get("model_name", "force-ridge")))
		self.model_version = QLineEdit(str(defaults.get("model_version", "1")))
		self.validation_fraction = QDoubleSpinBox()
		self.validation_fraction.setRange(0.01, 0.99)
		self.validation_fraction.setDecimals(3)
		self.validation_fraction.setValue(float(defaults.get("validation_fraction", 0.2)))
		self.ridge_alpha = QDoubleSpinBox()
		self.ridge_alpha.setRange(0.0, 1_000_000.0)
		self.ridge_alpha.setDecimals(9)
		self.ridge_alpha.setValue(float(defaults.get("ridge_alpha", 1.0)))
		self.random_seed = QSpinBox()
		self.random_seed.setRange(0, 2_147_483_647)
		self.random_seed.setValue(int(defaults.get("random_seed", 42)))
		self.split_group = QComboBox()
		self.split_group.addItems(("auto", "experiment", "acquisition_batch", "force_step"))
		self.overwrite = QCheckBox("Explicitly approve replacing existing artifacts")
		form = QFormLayout()
		form.addRow("Model output", output_row)
		form.addRow("Model name", self.model_name)
		form.addRow("Model version", self.model_version)
		form.addRow("Validation fraction", self.validation_fraction)
		form.addRow("Ridge alpha", self.ridge_alpha)
		form.addRow("Random seed", self.random_seed)
		form.addRow("Leakage-safe split", self.split_group)
		form.addRow("Overwrite", self.overwrite)
		settings = QGroupBox("Training settings")
		settings.setLayout(form)

		self.train = QPushButton("Train model")
		self.cancel = QPushButton("Cancel")
		self.accept = QPushButton("Accept and load model")
		self.rollback = QPushButton("Load previous active model")
		self.train.clicked.connect(self._train)
		self.cancel.clicked.connect(controller.cancel)
		self.accept.clicked.connect(controller.accept_model)
		self.rollback.clicked.connect(controller.rollback_model)
		buttons = QHBoxLayout()
		for button in (self.train, self.cancel, self.accept, self.rollback):
			buttons.addWidget(button)

		self.status = StatusBadge("IDLE")
		self.message = QLabel("Select and audit a dataset.")
		self.message.setWordWrap(True)
		self.progress = QProgressBar()
		self.progress.setRange(0, 1)
		self.progress.setValue(0)
		status_row = QFormLayout()
		status_row.addRow("State", self.status)
		status_row.addRow("Result", self.message)
		status_row.addRow("Progress", self.progress)
		status_box = QGroupBox("Worker status")
		status_box.setLayout(status_row)

		self.report = QPlainTextEdit()
		self.report.setReadOnly(True)
		self.report.setPlaceholderText(
			"Audit coverage, quality distributions, split groups, and model metrics appear here."
		)
		report_box = QGroupBox("Dataset audit and model review")
		report_layout = QVBoxLayout(report_box)
		report_layout.addWidget(self.report)

		note = QLabel(
			"Training runs in a separate cancellable process. Existing artifacts are "
			"protected unless overwrite is explicitly checked. A trained candidate is "
			"not active until Accept and load model succeeds against the current "
			"reference, geometry, and feature schema. Force inference must still be "
			"started explicitly."
		)
		note.setWordWrap(True)

		layout = QVBoxLayout(self)
		layout.addLayout(dataset_row)
		layout.addWidget(settings)
		layout.addLayout(buttons)
		layout.addWidget(status_box)
		layout.addWidget(report_box, 1)
		layout.addWidget(note)

		controller.state_updated.connect(self.update_state)

	def _browse_dataset(self) -> None:
		path = QFileDialog.getExistingDirectory(
			self, "Select dataset directory", self.dataset.text()
		)
		if path:
			self.dataset.setText(path)

	def _browse_output(self) -> None:
		path, _filter = QFileDialog.getSaveFileName(
			self, "Select model output", self.output.text(), "Pickle model (*.pkl)"
		)
		if path:
			self.output.setText(path)

	def _train(self) -> None:
		self.controller.train({
			"model_path": self.output.text(),
			"model_name": self.model_name.text(),
			"model_version": self.model_version.text(),
			"validation_fraction": self.validation_fraction.value(),
			"ridge_alpha": self.ridge_alpha.value(),
			"random_seed": self.random_seed.value(),
			"split_group": self.split_group.currentText(),
			"overwrite": self.overwrite.isChecked(),
		})

	@staticmethod
	def _compact_report(state: dict) -> str:
		audit = state.get("audit")
		training = state.get("training")
		sections = []
		if audit:
			sections.append("DATASET AUDIT\n" + json.dumps(audit, indent=2, sort_keys=True))
		if training:
			metadata = training.get("metadata", {})
			review = {
				"model_name": metadata.get("model_name"),
				"model_version": metadata.get("model_version"),
				"calibrated_force_range_N": [
					metadata.get("calibrated_min_force_N"),
					metadata.get("calibrated_max_force_N"),
				],
				"training_metrics": metadata.get("training_metrics"),
				"validation_metrics": metadata.get("validation_metrics"),
				"validation_metrics_by_force_N": metadata.get(
					"validation_metrics_by_force_N"
				),
				"training_configuration": metadata.get("training_configuration"),
				"reference_profile_compatibility": metadata.get(
					"reference_profile_compatibility"
				),
				"geometry_configuration_version": metadata.get(
					"geometry_configuration_version"
				),
				"feature_schema_version": metadata.get("feature_schema_version"),
				"model_path": training.get("model_path"),
			}
			sections.append("MODEL REVIEW\n" + json.dumps(review, indent=2, sort_keys=True))
		return "\n\n".join(sections)

	def update_state(self, state: dict) -> None:
		name = str(state.get("state", TrainingState.IDLE.value))
		kind = "good" if name in {
			TrainingState.AUDIT_READY.value,
			TrainingState.MODEL_REVIEW.value,
			TrainingState.ACCEPTED.value,
		} else "error" if name == TrainingState.FAULT.value else (
			"warning" if name in {
				TrainingState.AUDITING.value, TrainingState.TRAINING.value,
				TrainingState.ACTIVATING.value, TrainingState.CANCELLED.value,
			} else "neutral"
		)
		self.status.set_status(name, kind)
		self.message.setText(state.get("message") or "—")
		busy = bool(state.get("busy"))
		self.progress.setRange(0, 0 if busy else 1)
		if not busy:
			self.progress.setValue(1 if state.get("audit") else 0)
		self.audit.setEnabled(not busy)
		self.train.setEnabled(not busy and bool(state.get("can_train")))
		self.cancel.setEnabled(busy)
		self.accept.setEnabled(bool(state.get("can_accept")))
		self.rollback.setEnabled(not busy and bool(state.get("can_rollback")))
		self.report.setPlainText(self._compact_report(state))
