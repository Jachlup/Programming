"""Operator controls for the semi-automatic collection coordinator."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
	QCheckBox,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QProgressBar,
	QPushButton,
	QVBoxLayout,
	QWidget,
)

from experiment_control import ExperimentState

from .common import StatusBadge


class CollectionPanel(QWidget):
	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self._state: dict = {}

		self.protocol_path = QLineEdit(
			str(Path(__file__).resolve().parents[1] / "collection-protocol.yaml")
		)
		self.browse = QPushButton("Browse…")
		self.load = QPushButton("Load protocol")
		self.browse.clicked.connect(self._browse)
		self.load.clicked.connect(self._load)
		protocol_row = QHBoxLayout()
		protocol_row.addWidget(self.protocol_path, 1)
		protocol_row.addWidget(self.browse)
		protocol_row.addWidget(self.load)

		self.confirm_every_step = QCheckBox("Require confirmation before each transition")
		self.confirm_every_step.setChecked(True)
		self.confirm_every_step.toggled.connect(
			controller.set_confirmation_every_step
		)
		self.operator_force = QDoubleSpinBox()
		self.operator_force.setRange(-100000.0, 100000.0)
		self.operator_force.setDecimals(4)
		self.operator_force.setSuffix(" N")
		definition = QFormLayout()
		definition.addRow("Confirmed known force", self.operator_force)
		definition.addRow("Policy", self.confirm_every_step)
		definition_box = QGroupBox("Operator confirmation")
		definition_box.setLayout(definition)

		self.start = QPushButton("Run preflight and start")
		self.confirm = QPushButton("Confirm force and save step")
		self.accept = QPushButton("Accept recorded step")
		self.retry = QPushButton("Retry step")
		self.skip = QPushButton("Skip step")
		self.pause = QPushButton("Pause")
		self.abort = QPushButton("Abort experiment")
		self.start.clicked.connect(controller.start)
		self.confirm.clicked.connect(
			lambda: controller.confirm_step(self.operator_force.value())
		)
		self.accept.clicked.connect(controller.accept_review)
		self.retry.clicked.connect(controller.retry_step)
		self.skip.clicked.connect(controller.skip_step)
		self.pause.clicked.connect(controller.pause)
		self.abort.clicked.connect(controller.abort)
		controls = QHBoxLayout()
		for button in (
			self.start, self.confirm, self.accept, self.retry, self.skip,
			self.pause, self.abort,
		):
			controls.addWidget(button)

		self.status = StatusBadge("PREFLIGHT")
		self.message = QLabel("Load a protocol.")
		self.message.setWordWrap(True)
		self.experiment = QLabel("—")
		self.step = QLabel("0 / 0")
		self.step_id = QLabel("—")
		self.direction = QLabel("—")
		self.repetition = QLabel("—")
		self.planned_force = QLabel("—")
		self.red_point = QLabel("—")
		self.dataset_root = QLabel("—")
		self.dataset_root.setTextInteractionFlags(
			self.dataset_root.textInteractionFlags()
		)
		self.stability = QLabel("—")
		self.stability.setWordWrap(True)
		self.preflight = QLabel("—")
		self.preflight.setWordWrap(True)
		self.progress = QProgressBar()
		self.progress.setRange(0, 1)
		status_form = QFormLayout()
		status_form.addRow("State", self.status)
		status_form.addRow("Instruction", self.message)
		status_form.addRow("Experiment", self.experiment)
		status_form.addRow("Step", self.step)
		status_form.addRow("Force step ID", self.step_id)
		status_form.addRow("Direction", self.direction)
		status_form.addRow("Repetition", self.repetition)
		status_form.addRow("Planned force", self.planned_force)
		status_form.addRow("Closest red point to blue", self.red_point)
		status_form.addRow("Stability", self.stability)
		status_form.addRow("Dataset target", self.dataset_root)
		status_form.addRow("Preflight", self.preflight)
		status_form.addRow("Progress", self.progress)
		status_box = QGroupBox("Collection state")
		status_box.setLayout(status_form)

		note = QLabel(
			"Ground-truth force is the operator-confirmed value in newtons. Motor "
			"torque is auxiliary metadata only. Set or adjust torque in the Motor tab; "
			"a lower torque target automatically backs off 0.3 rad before reapproach. "
			"The selected measurement point is the closest valid permanent red point "
			"to the calibrated blue target. A physical emergency stop is required."
		)
		note.setWordWrap(True)

		layout = QVBoxLayout(self)
		layout.addLayout(protocol_row)
		layout.addWidget(definition_box)
		layout.addLayout(controls)
		layout.addWidget(status_box)
		layout.addWidget(note)
		layout.addStretch()

		controller.state_updated.connect(self.update_state)
		controller.protocol_loaded.connect(self._protocol_loaded)

	def _browse(self) -> None:
		path, _filter = QFileDialog.getOpenFileName(
			self, "Open collection protocol", self.protocol_path.text(), "YAML (*.yaml *.yml)"
		)
		if path:
			self.protocol_path.setText(path)

	def _load(self) -> None:
		try:
			self.controller.load_protocol(self.protocol_path.text())
		except Exception as exc:
			self.controller.log_emitted.emit("error", f"Protocol load failed: {exc}")

	def _protocol_loaded(self, protocol) -> None:
		self.confirm_every_step.setChecked(protocol.require_operator_confirmation)
		if protocol.steps:
			self.operator_force.setValue(protocol.steps[0].force_N)

	def update_state(self, state: dict) -> None:
		self._state = dict(state)
		name = str(state.get("state", ExperimentState.PREFLIGHT.value))
		kind = "good" if name == ExperimentState.COMPLETE.value else (
			"error" if name in {ExperimentState.FAULT.value, ExperimentState.ABORTED.value}
			else "warning" if name not in {ExperimentState.PREFLIGHT.value} else "neutral"
		)
		self.status.set_status(name, kind)
		self.message.setText(state.get("message") or "—")
		self.experiment.setText(state.get("experiment_id") or "—")
		current = int(state.get("current_step", 0))
		planned = int(state.get("planned_steps", 0))
		self.step.setText(f"{current} / {planned}")
		self.step_id.setText(state.get("force_step_id") or "—")
		self.direction.setText(state.get("direction") or "—")
		repetition = int(state.get("repetition", 0))
		self.repetition.setText("—" if repetition == 0 else str(repetition))
		force = state.get("planned_force_N")
		self.planned_force.setText("—" if force is None else f"{float(force):g} N")
		if force is not None and not self.operator_force.hasFocus():
			self.operator_force.setValue(float(force))
		self.red_point.setText(state.get("nearest_red_point_id") or "Unavailable")
		self.dataset_root.setText(state.get("dataset_root") or "—")
		self.stability.setText(
			f"{float(state.get('stability_elapsed_s', 0.0)):.2f}s — "
			f"{state.get('stability_reason') or 'waiting'}"
		)
		errors = list(state.get("preflight_errors", []))
		self.preflight.setText("Ready" if not errors else "\n".join(f"• {item}" for item in errors))
		self.progress.setRange(0, max(planned, 1))
		self.progress.setValue(int(state.get("completed_steps", 0)) + int(state.get("skipped_steps", 0)))

		loaded = bool(state.get("protocol_loaded"))
		waiting = name == ExperimentState.WAITING_FOR_OPERATOR.value
		reviewing = name == ExperimentState.REVIEWING_STEP.value
		terminal = name in {
			ExperimentState.PREFLIGHT.value,
			ExperimentState.COMPLETE.value,
			ExperimentState.ABORTED.value,
			ExperimentState.FAULT.value,
		}
		self.start.setEnabled(loaded and name == ExperimentState.PREFLIGHT.value and not errors)
		self.confirm.setEnabled(waiting)
		self.accept.setEnabled(reviewing)
		self.retry.setEnabled(reviewing)
		self.skip.setEnabled(waiting or reviewing)
		self.pause.setEnabled(
			loaded and not terminal and name not in {
				ExperimentState.MOVING.value, ExperimentState.RECORDING.value,
			}
		)
		self.pause.setText("Resume" if state.get("paused") else "Pause")
		self.abort.setEnabled(loaded and name not in {
			ExperimentState.COMPLETE.value, ExperimentState.ABORTED.value,
		})
		if state.get("pending_action") == "repeat_reset":
			self.confirm.setText("Confirm open and reapply")
		else:
			self.confirm.setText("Confirm force and save step")
