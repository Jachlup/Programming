"""Validated dataset-acquisition controls."""

from __future__ import annotations

import shlex

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
	QDoubleSpinBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QPushButton,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from .common import StatusBadge


class DatasetPanel(QWidget):
	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self._default_frames = 1
		self._camera_running = False
		self._state: dict = {}

		self.experiment = QLineEdit()
		self.experiment.setPlaceholderText("e.g. finger_calibration")
		self.force = QDoubleSpinBox()
		self.force.setRange(-100000.0, 100000.0)
		self.force.setDecimals(4)
		self.force.setSuffix(" N")
		self.frames = QSpinBox()
		self.frames.setRange(1, 100000)
		self.frames.setValue(1)

		entry = QFormLayout()
		entry.addRow("Experiment ID", self.experiment)
		entry.addRow("Known force", self.force)
		entry.addRow("Frames per sample batch", self.frames)
		entry_box = QGroupBox("Sample definition")
		entry_box.setLayout(entry)

		self.start = QPushButton("Start session")
		self.change_force = QPushButton("Change force label")
		self.sample = QPushButton("Queue sample batch")
		self.stop = QPushButton("Stop session")
		self.status = QPushButton("Print dataset status")
		self.start.clicked.connect(self._start)
		self.change_force.clicked.connect(
			lambda: controller.execute(f"dataset_force {self.force.value():.12g}")
		)
		self.sample.clicked.connect(
			lambda: controller.execute(f"dataset_sample {self.frames.value()}")
		)
		self.stop.clicked.connect(lambda: controller.execute("dataset_stop"))
		self.status.clicked.connect(lambda: controller.execute("dataset_status"))
		buttons = QHBoxLayout()
		for button in (self.start, self.change_force, self.sample, self.stop, self.status):
			buttons.addWidget(button)

		self.session = StatusBadge("Inactive")
		self.recording = StatusBadge("Not ready")
		self.active_experiment = QLabel("—")
		self.current_force = QLabel("—")
		self.accepted = QLabel("0")
		self.rejected = QLabel("0")
		self.pending = QLabel("0")
		self.last_result = QLabel("—")
		self.last_result.setWordWrap(True)
		status_form = QFormLayout()
		status_form.addRow("Session", self.session)
		status_form.addRow("Current frame recordable", self.recording)
		status_form.addRow("Active experiment", self.active_experiment)
		status_form.addRow("Current force label", self.current_force)
		status_form.addRow("Accepted samples", self.accepted)
		status_form.addRow("Rejected attempts", self.rejected)
		status_form.addRow("Pending frames", self.pending)
		status_form.addRow("Last result", self.last_result)
		status_box = QGroupBox("Dataset status")
		status_box.setLayout(status_form)

		note = QLabel(
			"Queued frames pass through the existing tracker, origin, geometry, feature, "
			"and image validation. Invalid frames are recorded as rejections and follow "
			"the configured retry policy."
		)
		note.setWordWrap(True)

		layout = QVBoxLayout(self)
		layout.addWidget(entry_box)
		layout.addLayout(buttons)
		layout.addWidget(status_box)
		layout.addWidget(note)
		layout.addStretch()

		controller.state_updated.connect(self.update_state)
		controller.configuration_applied.connect(self.update_configuration)
		controller.camera_state_changed.connect(self.update_camera_state)

	def update_camera_state(self, running: bool, _paused: bool) -> None:
		self._camera_running = running
		if self._state:
			self.update_state(self._state)

	def _start(self) -> None:
		experiment = self.experiment.text().strip()
		if not experiment:
			self.controller.log_emitted.emit("error", "Experiment ID must not be empty.")
			return
		self.controller.execute(
			f"dataset_start {shlex.quote(experiment)} {self.force.value():.12g}"
		)

	def update_configuration(self, mapping: dict) -> None:
		self._default_frames = int(mapping.get("dataset", {}).get("frames_per_sample", 1))
		if not self.frames.hasFocus():
			self.frames.setValue(self._default_frames)

	def update_state(self, state: dict) -> None:
		self._state = state
		active = bool(state.get("dataset_active"))
		self.session.set_status("Active" if active else "Inactive", "good" if active else "neutral")
		valid = bool(state.get("recording_valid")) and state.get("tracking_active")
		if valid:
			self.recording.set_status("Valid", "good")
		else:
			reason = state.get("recording_invalid_reason") or "Tracking is not active"
			self.recording.set_status("Invalid", "error")
			self.recording.setToolTip(reason)
		self.active_experiment.setText(state.get("dataset_experiment_id") or "—")
		force = state.get("dataset_force_N")
		self.current_force.setText("—" if force is None else f"{float(force):g} N")
		if active and force is not None and not self.force.hasFocus():
			with QSignalBlocker(self.force):
				self.force.setValue(float(force))
		self.accepted.setText(str(state.get("dataset_accepted", 0)))
		self.rejected.setText(str(state.get("dataset_rejected", 0)))
		self.pending.setText(str(state.get("dataset_pending", 0)))
		self.last_result.setText(state.get("dataset_last_result") or "—")

		tracking_ready = bool(state.get("reference_configured")) and bool(
			state.get("tracking_active")
		)
		self.start.setEnabled(not active and tracking_ready)
		self.experiment.setEnabled(not active)
		self.change_force.setEnabled(active)
		self.sample.setEnabled(active and self._camera_running)
		self.stop.setEnabled(active)
