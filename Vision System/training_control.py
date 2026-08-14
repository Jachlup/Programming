"""Cancellable QProcess-based dataset audit, training, and model activation."""

from __future__ import annotations

from enum import Enum
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

from PySide6.QtCore import QObject, QProcess, QTimer, Signal, Slot

from force_model import ModelMetadata
from model_registry import ModelRegistry


class TrainingState(str, Enum):
	IDLE = "IDLE"
	AUDITING = "AUDITING"
	AUDIT_READY = "AUDIT_READY"
	TRAINING = "TRAINING"
	MODEL_REVIEW = "MODEL_REVIEW"
	ACTIVATING = "ACTIVATING"
	ACCEPTED = "ACCEPTED"
	CANCELLED = "CANCELLED"
	FAULT = "FAULT"


class TrainingController(QObject):
	state_updated = Signal(object)
	log_emitted = Signal(str, str)

	def __init__(self, application_controller, parent=None) -> None:
		super().__init__(parent)
		self.application_controller = application_controller
		self.state = TrainingState.IDLE
		self.process = QProcess(self)
		self.process.setWorkingDirectory(str(Path(__file__).resolve().parent))
		self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
		self.process.readyReadStandardOutput.connect(self._read_output)
		self.process.finished.connect(self._process_finished)
		self.process.errorOccurred.connect(self._process_error)
		application_controller.model_activation_completed.connect(
			self._activation_completed
		)

		self.dataset_path = ""
		self.audit_report: dict[str, Any] | None = None
		self.training_report: dict[str, Any] | None = None
		self.candidate_model_path = ""
		self.candidate_metadata_path = ""
		self._process_kind = ""
		self._temporary_report_path = ""
		self._message = "Select and audit a dataset."
		self._pending_activation = ""
		self._cancel_requested = False
		registry_path = Path(__file__).resolve().parent / "models" / "model-registry.json"
		self.registry = ModelRegistry(registry_path)

	def _emit_state(self) -> None:
		metadata = None
		if self.training_report is not None:
			metadata = self.training_report.get("metadata")
		try:
			can_rollback = self.registry.previous_model() is not None
		except (OSError, ValueError, json.JSONDecodeError):
			can_rollback = False
		self.state_updated.emit({
			"state": self.state.value,
			"message": self._message,
			"dataset_path": self.dataset_path,
			"busy": self.process.state() != QProcess.ProcessState.NotRunning,
			"audit": self.audit_report,
			"training": self.training_report,
			"metadata": metadata,
			"candidate_model_path": self.candidate_model_path,
			"candidate_metadata_path": self.candidate_metadata_path,
			"can_train": bool(
				self.audit_report and self.audit_report.get("training_allowed")
			),
			"can_accept": self.state == TrainingState.MODEL_REVIEW,
			"can_rollback": can_rollback,
		})

	def _set_state(self, state: TrainingState, message: str) -> None:
		self.state = state
		self._message = str(message)
		self._emit_state()

	def _start_process(self, kind: str, program: str, arguments: list[str]) -> None:
		if self.process.state() != QProcess.ProcessState.NotRunning:
			raise RuntimeError("A training operation is already running")
		self._process_kind = kind
		self._cancel_requested = False
		self.process.setProgram(program)
		self.process.setArguments(arguments)
		self.process.start()
		self._emit_state()

	@Slot(str)
	def audit(self, dataset_path: str) -> None:
		path = Path(dataset_path).expanduser()
		if not path.exists():
			self._set_state(TrainingState.FAULT, f"Dataset does not exist: {path}")
			return
		try:
			with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as stream:
				self._temporary_report_path = stream.name
			self.dataset_path = str(path.resolve())
			self.audit_report = None
			self._set_state(TrainingState.AUDITING, "Auditing dataset in a separate process…")
			self._start_process(
				"audit",
				sys.executable,
				[
					str(Path(__file__).resolve().with_name("dataset_audit.py")),
					self.dataset_path,
					"--json-out",
					self._temporary_report_path,
				],
			)
		except Exception as exc:
			self._set_state(TrainingState.FAULT, f"Audit could not start: {exc}")

	def train(self, specification: dict[str, Any]) -> None:
		if not self.audit_report or not self.audit_report.get("training_allowed"):
			self._set_state(TrainingState.FAULT, "Run a passing dataset audit first")
			return
		try:
			output = Path(str(specification["model_path"])).expanduser()
			validation_fraction = float(specification.get("validation_fraction", 0.2))
			random_seed = int(specification.get("random_seed", 42))
			ridge_alpha = float(specification.get("ridge_alpha", 1.0))
			model_name = str(specification.get("model_name", "force-ridge")).strip()
			model_version = str(specification.get("model_version", "1")).strip()
			split_group = str(specification.get("split_group", "auto"))
			overwrite = bool(specification.get("overwrite", False))
			if not 0.0 < validation_fraction < 1.0:
				raise ValueError("Validation fraction must be between zero and one")
			if not math.isfinite(ridge_alpha) or ridge_alpha < 0.0:
				raise ValueError("Ridge alpha must be finite and non-negative")
			if not model_name or not model_version:
				raise ValueError("Model name and version must not be empty")
			if split_group not in {"auto", "experiment", "acquisition_batch", "force_step"}:
				raise ValueError("Split group is invalid")
			metadata = output.with_suffix(".json")
			existing = [path for path in (output, metadata) if path.exists()]
			if existing and not overwrite:
				raise FileExistsError(
					"Explicit overwrite approval is required for: "
					+ ", ".join(str(path) for path in existing)
				)
			arguments = [
				str(Path(__file__).resolve().with_name("train_force_model.py")),
				self.dataset_path,
				"--model-out", str(output.resolve()),
				"--validation-fraction", f"{validation_fraction:.12g}",
				"--random-seed", str(random_seed),
				"--ridge-alpha", f"{ridge_alpha:.12g}",
				"--split-group", split_group,
				"--model-name", model_name,
				"--model-version", model_version,
			]
			if overwrite:
				arguments.append("--overwrite")
			self.candidate_model_path = str(output.resolve())
			self.candidate_metadata_path = str(metadata.resolve())
			self.training_report = None
			self._set_state(TrainingState.TRAINING, "Training in a cancellable process…")
			self._start_process("training", sys.executable, arguments)
		except Exception as exc:
			self._set_state(TrainingState.FAULT, f"Training could not start: {exc}")

	@Slot()
	def cancel(self) -> None:
		if self.process.state() == QProcess.ProcessState.NotRunning:
			return
		self._message = "Cancelling worker process…"
		self._cancel_requested = True
		self.process.terminate()
		QTimer.singleShot(
			1000,
			lambda: self.process.kill()
			if self.process.state() != QProcess.ProcessState.NotRunning else None,
		)

	def _read_output(self) -> None:
		text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
		for line in text.splitlines():
			if line.strip():
				self.log_emitted.emit("info", line.rstrip())

	def _process_finished(self, exit_code: int, exit_status) -> None:
		kind, self._process_kind = self._process_kind, ""
		if self._cancel_requested:
			self._cancel_requested = False
			if self._temporary_report_path:
				Path(self._temporary_report_path).unlink(missing_ok=True)
				self._temporary_report_path = ""
			self._set_state(TrainingState.CANCELLED, f"{kind.capitalize()} cancelled.")
			return
		if kind == "audit":
			try:
				path = Path(self._temporary_report_path)
				if path.stat().st_size:
					self.audit_report = json.loads(path.read_text(encoding="utf-8"))
				path.unlink(missing_ok=True)
			except (OSError, ValueError, json.JSONDecodeError) as exc:
				self._set_state(TrainingState.FAULT, f"Audit report could not be read: {exc}")
				return
			finally:
				self._temporary_report_path = ""
			if self.audit_report is None:
				self._set_state(TrainingState.FAULT, "Audit process produced no report")
			elif exit_code in {0, 3}:
				allowed = bool(self.audit_report.get("training_allowed"))
				self._set_state(
					TrainingState.AUDIT_READY,
					"Dataset audit passed." if allowed else "Dataset audit found blocking errors.",
				)
			else:
				self._set_state(TrainingState.FAULT, f"Audit process exited with code {exit_code}")
			return

		if kind == "training":
			if exit_code != 0:
				state = (
					TrainingState.CANCELLED
					if self.process.error() == QProcess.ProcessError.Crashed
					else TrainingState.FAULT
				)
				self._set_state(state, f"Training process exited with code {exit_code}")
				return
			try:
				raw = json.loads(Path(self.candidate_metadata_path).read_text(encoding="utf-8"))
				metadata = ModelMetadata.from_dict(raw)
				identifier = self.registry.register(
					self.candidate_model_path,
					self.candidate_metadata_path,
					metadata,
				)
				self.training_report = {
					"metadata": metadata.to_dict(),
					"registry_id": identifier,
					"model_path": self.candidate_model_path,
					"metadata_path": self.candidate_metadata_path,
					"audit_rejected_rows": self.audit_report.get("rejected_rows", 0),
				}
			except Exception as exc:
				self._set_state(TrainingState.FAULT, f"Trained artifacts failed review: {exc}")
				return
			self._set_state(
				TrainingState.MODEL_REVIEW,
				"Training finished; review metrics before explicit activation.",
			)

	def _process_error(self, error) -> None:
		if error == QProcess.ProcessError.Crashed and self._message.startswith("Cancelling"):
			return
		self.log_emitted.emit("error", f"Worker process error: {self.process.errorString()}")
		if error == QProcess.ProcessError.FailedToStart:
			self._process_kind = ""
			self._set_state(
				TrainingState.FAULT,
				f"Worker process could not start: {self.process.errorString()}",
			)

	@Slot()
	def accept_model(self) -> None:
		if self.state != TrainingState.MODEL_REVIEW or self.training_report is None:
			return
		self._pending_activation = "candidate"
		self._set_state(
			TrainingState.ACTIVATING,
			"Validating model compatibility in the camera worker…",
		)
		self.application_controller.activate_force_model(
			self.candidate_model_path, self.candidate_metadata_path
		)

	@Slot()
	def rollback_model(self) -> None:
		entry = self.registry.previous_model()
		if entry is None:
			self.log_emitted.emit("warning", "No prior active model is available.")
			return
		self._pending_activation = "rollback"
		self._set_state(TrainingState.ACTIVATING, "Validating the previous model…")
		self.application_controller.activate_force_model(
			entry["model_path"], entry["metadata_path"]
		)

	@Slot(str, bool, str)
	def _activation_completed(self, model_path: str, ok: bool, message: str) -> None:
		if self.state != TrainingState.ACTIVATING:
			return
		if not ok:
			self._pending_activation = ""
			self._set_state(
				TrainingState.MODEL_REVIEW,
				f"Activation rejected; prior model retained: {message}",
			)
			return
		try:
			if self._pending_activation == "candidate":
				assert self.training_report is not None
				self.registry.mark_active(self.training_report["registry_id"])
				result = "Candidate model activated; inference remains operator-controlled."
			else:
				entry = self.registry.mark_rollback()
				result = f"Rolled back to {entry['model_name']} {entry['model_version']}."
		except Exception as exc:
			self._set_state(
				TrainingState.FAULT,
				f"Model loaded but registry update failed: {exc}",
			)
			return
		finally:
			self._pending_activation = ""
		self._set_state(TrainingState.ACCEPTED, result)

	def shutdown(self, timeout_ms: int = 2000) -> bool:
		if self.process.state() == QProcess.ProcessState.NotRunning:
			return True
		self.process.terminate()
		if self.process.waitForFinished(timeout_ms):
			return True
		self.process.kill()
		return self.process.waitForFinished(timeout_ms)
