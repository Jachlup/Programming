"""Reference calibration and permanent-origin workflow."""

from __future__ import annotations

import shlex

from PySide6.QtWidgets import (
	QFileDialog,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QPlainTextEdit,
	QPushButton,
	QScrollArea,
	QVBoxLayout,
	QWidget,
)

from .common import StatusBadge


class ReferencePanel(QWidget):
	def __init__(self, controller, parent=None, *, compact: bool = False) -> None:
		super().__init__(parent)
		self.controller = controller
		self._state: dict = {}
		self._camera_running = False

		self.workflow = StatusBadge("Not configured")
		self.tracking = StatusBadge("Tracking stopped")
		self.origin = StatusBadge("Origin not selected")
		self.quality = QLabel("—")
		status_form = QFormLayout()
		status_form.addRow("Reference", self.workflow)
		status_form.addRow("Tracking", self.tracking)
		status_form.addRow("Origin", self.origin)
		status_form.addRow("Proposal quality", self.quality)
		status_box = QGroupBox("Reference state")
		status_box.setLayout(status_form)

		self.rigid_indices = QLineEdit()
		self.rigid_indices.setPlaceholderText("Optional, e.g. 2,5")
		self.start = QPushButton("1. Start reference setup")
		self.origin_select = QPushButton("2. Enable origin selection")
		self.origin_clear = QPushButton("Clear origin candidate")
		self.origin_status = QPushButton("Origin status")
		self.preview = QPushButton("3. Preview RANSAC")
		self.accept = QPushButton("4. Accept proposal")
		self.reject = QPushButton("Reject proposal")
		self.clear = QPushButton("Clear reference")
		self.reacquire = QPushButton("Reacquire lost origin")
		self.reference_status = QPushButton("Reference status")
		self.tracking_start = QPushButton("Start tracking")
		self.tracking_stop = QPushButton("Stop tracking")

		self.start.clicked.connect(self._start_reference)
		self.origin_select.clicked.connect(
			lambda: controller.execute("reference_origin_select")
		)
		self.origin_clear.clicked.connect(
			lambda: controller.execute("reference_origin_clear")
		)
		self.origin_status.clicked.connect(
			lambda: controller.execute("reference_origin_status")
		)
		self.preview.clicked.connect(lambda: controller.execute("reference_preview"))
		self.accept.clicked.connect(lambda: controller.execute("reference_accept"))
		self.reject.clicked.connect(lambda: controller.execute("reference_reject"))
		self.clear.clicked.connect(lambda: controller.execute("reference_clear"))
		self.reacquire.clicked.connect(lambda: controller.execute("origin_reacquire"))
		self.reference_status.clicked.connect(
			lambda: controller.execute("reference_status")
		)
		self.tracking_start.clicked.connect(lambda: controller.execute("tracking_start"))
		self.tracking_stop.clicked.connect(lambda: controller.execute("tracking_stop"))

		workflow_grid = QGridLayout()
		workflow_grid.addWidget(QLabel("Rigid detection indices"), 0, 0)
		workflow_grid.addWidget(self.rigid_indices, 0, 1, 1, 2)
		workflow_grid.addWidget(self.start, 1, 0, 1, 3)
		workflow_grid.addWidget(self.origin_select, 2, 0, 1, 2)
		workflow_grid.addWidget(self.origin_clear, 2, 2)
		workflow_grid.addWidget(self.preview, 3, 0, 1, 3)
		workflow_grid.addWidget(self.accept, 4, 0, 1, 2)
		workflow_grid.addWidget(self.reject, 4, 2)
		workflow_grid.addWidget(self.reacquire, 5, 0, 1, 3)
		workflow_grid.addWidget(self.origin_status, 6, 0)
		workflow_grid.addWidget(self.reference_status, 6, 1)
		workflow_grid.addWidget(self.clear, 6, 2)
		workflow_grid.addWidget(self.tracking_start, 7, 0, 1, 2)
		workflow_grid.addWidget(self.tracking_stop, 7, 2)
		workflow_box = QGroupBox("Setup workflow")
		workflow_box.setLayout(workflow_grid)

		self.path = QLineEdit()
		self.path.setPlaceholderText("Default: profile_path from camera-config.yaml")
		browse_load = QPushButton("Browse…")
		browse_save = QPushButton("Choose save path…")
		self.load = QPushButton("Load profile")
		self.save = QPushButton("Save profile")
		browse_load.clicked.connect(lambda: self._browse(False))
		browse_save.clicked.connect(lambda: self._browse(True))
		self.load.clicked.connect(self._load)
		self.save.clicked.connect(self._save)
		profile_layout = QVBoxLayout()
		profile_layout.addWidget(self.path)
		if compact:
			path_buttons = QGridLayout()
			path_buttons.addWidget(browse_load, 0, 0)
			path_buttons.addWidget(browse_save, 0, 1)
			path_buttons.addWidget(self.load, 1, 0)
			path_buttons.addWidget(self.save, 1, 1)
		else:
			path_buttons = QHBoxLayout()
			path_buttons.addWidget(browse_load)
			path_buttons.addWidget(browse_save)
			path_buttons.addWidget(self.load)
			path_buttons.addWidget(self.save)
		profile_layout.addLayout(path_buttons)
		profile_box = QGroupBox("Reference profile")
		profile_box.setLayout(profile_layout)

		self.details = QPlainTextEdit()
		self.details.setReadOnly(True)
		self.details.setPlaceholderText("RANSAC validity, fatal errors, and warnings appear here.")

		instruction = QLabel(
			"Click the live image beside these controls when origin selection or "
			"reacquisition is armed. The click is resolved against the current blob contours."
			if compact else
			"Origin clicks are made on the Camera tab. The click is resolved against "
			"the currently detected blob contours."
		)
		instruction.setWordWrap(True)

		if compact:
			self.details.setMinimumHeight(140)
			content = QWidget()
			content_layout = QVBoxLayout(content)
			content_layout.addWidget(instruction)
			content_layout.addWidget(status_box)
			content_layout.addWidget(workflow_box)
			content_layout.addWidget(profile_box)
			content_layout.addWidget(QLabel("Proposal diagnostics"))
			content_layout.addWidget(self.details)
			content_layout.addStretch()
			scroll = QScrollArea()
			scroll.setWidgetResizable(True)
			scroll.setWidget(content)
			layout = QVBoxLayout(self)
			layout.setContentsMargins(0, 0, 0, 0)
			layout.addWidget(scroll)
		else:
			left = QVBoxLayout()
			left.addWidget(status_box)
			left.addWidget(workflow_box)
			left.addWidget(profile_box)
			left.addStretch()
			layout = QHBoxLayout(self)
			layout.addLayout(left, 1)
			right = QVBoxLayout()
			right.addWidget(instruction)
			right.addWidget(QLabel("Proposal diagnostics"))
			right.addWidget(self.details, 1)
			layout.addLayout(right, 1)

		controller.state_updated.connect(self.update_state)
		controller.camera_state_changed.connect(self.update_camera_state)

	def update_camera_state(self, running: bool, _paused: bool) -> None:
		self._camera_running = running
		if self._state:
			self.update_state(self._state)

	def _start_reference(self) -> None:
		indices = self.rigid_indices.text().strip().replace(" ", "")
		if indices and any(not item.isdigit() for item in indices.split(",")):
			self.controller.log_emitted.emit(
				"error", "Rigid indices must be comma-separated non-negative integers."
			)
			return
		command = "reference_start" + (f" {indices}" if indices else "")
		self.controller.execute(command)

	def _browse(self, saving: bool) -> None:
		if saving:
			path, _ = QFileDialog.getSaveFileName(
				self, "Save reference profile", self.path.text(), "JSON (*.json)"
			)
		else:
			path, _ = QFileDialog.getOpenFileName(
				self, "Load reference profile", self.path.text(), "JSON (*.json)"
			)
		if path:
			self.path.setText(path)

	def _path_command(self, name: str) -> str:
		path = self.path.text().strip()
		return name if not path else f"{name} {shlex.quote(path)}"

	def _load(self) -> None:
		self.controller.execute(self._path_command("reference_load"))

	def _save(self) -> None:
		self.controller.execute(self._path_command("reference_save"))

	def update_state(self, state: dict) -> None:
		self._state = state
		setup = state.get("reference_setup_mode", "IDLE")
		configured = bool(state.get("reference_configured"))
		proposal = bool(state.get("proposal_available"))
		if state.get("origin_reacquisition_selection"):
			workflow_text, kind = "Waiting for reacquisition click", "warning"
		elif state.get("origin_reacquisition_required"):
			workflow_text, kind = "Waiting for origin reacquisition", "error"
		elif setup == "AWAITING_ORIGIN_CLICK":
			workflow_text, kind = "Waiting for origin click", "warning"
		elif proposal:
			workflow_text, kind = "Previewing proposal", (
				"good" if state.get("proposal_valid") else "error"
			)
		elif configured:
			workflow_text, kind = "Accepted profile", "good"
		elif setup in {"ORIGIN_REQUIRED", "READY_TO_PREVIEW"}:
			workflow_text, kind = setup.replace("_", " ").title(), "warning"
		else:
			workflow_text, kind = "Not configured", "neutral"
		self.workflow.set_status(workflow_text, kind)

		if state.get("tracking_active"):
			valid = bool(state.get("tracking_valid"))
			self.tracking.set_status(
				f"{'Successful' if valid else 'Invalid'} · "
				f"q={float(state.get('tracking_quality', 0.0)):.2f}",
				"good" if valid else "error",
			)
		else:
			self.tracking.set_status("Tracking stopped", "neutral")

		candidate = state.get("origin_candidate_index")
		if state.get("origin_reacquisition_selection"):
			self.origin.set_status("Click restored marker", "warning")
		elif state.get("origin_reacquisition_required"):
			self.origin.set_status("Lost", "error")
		elif state.get("origin_valid"):
			self.origin.set_status("Tracking", "good")
		elif state.get("reference_origin_selection"):
			self.origin.set_status("Click requested", "warning")
		elif candidate is not None:
			self.origin.set_status(f"Candidate D{candidate}", "good")
		elif configured:
			self.origin.set_status("Configured", "neutral")
		else:
			self.origin.set_status("Not selected", "neutral")

		quality = state.get("proposal_quality")
		self.quality.setText("—" if quality is None else f"{float(quality):.3f}")
		errors = state.get("proposal_errors", [])
		warnings = state.get("proposal_warnings", [])
		lines = [f"Valid: {state.get('proposal_valid')}"] if proposal else []
		lines.extend(f"FATAL: {message}" for message in errors)
		lines.extend(f"WARNING: {message}" for message in warnings)
		self.details.setPlainText("\n".join(lines))

		setup_active = state.get("mode") == "REFERENCE_SETUP"
		self.start.setEnabled(
			self._camera_running and int(state.get("detected_marker_count", 0)) > 0
		)
		self.origin_select.setEnabled(
			setup_active and setup in {"ORIGIN_REQUIRED", "READY_TO_PREVIEW"}
			and not state.get("reference_origin_selection")
		)
		self.origin_clear.setEnabled(
			setup_active and (
				state.get("reference_origin_selection") or candidate is not None
			)
		)
		self.preview.setEnabled(setup_active and setup == "READY_TO_PREVIEW")
		self.accept.setEnabled(proposal and bool(state.get("proposal_valid")))
		self.reject.setEnabled(proposal)
		self.save.setEnabled(configured)
		self.clear.setEnabled(configured or setup_active or proposal)
		self.reacquire.setEnabled(
			bool(state.get("origin_reacquisition_required"))
			and not state.get("origin_reacquisition_selection")
		)
		self.tracking_start.setEnabled(
			configured and not bool(state.get("tracking_active"))
		)
		self.tracking_stop.setEnabled(bool(state.get("tracking_active")))
