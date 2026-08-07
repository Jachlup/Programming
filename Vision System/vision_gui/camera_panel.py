"""Live camera view, runtime status, and overlay controls."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
	QCheckBox,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QVBoxLayout,
	QWidget,
)

from .common import StatusBadge, VideoLabel


DISPLAY_OPTIONS = (
	"show_points", "show_lines", "show_outliers", "show_geometry",
	"show_features", "show_circles", "show_status", "show_warnings",
	"show_mask_only",
)


class CameraPanel(QWidget):
	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self.video = VideoLabel()
		self.video.frame_clicked.connect(controller.click_frame)

		self.start_button = QPushButton("Start camera")
		self.stop_button = QPushButton("Stop camera")
		self.pause_button = QPushButton("Pause processing")
		self.pause_button.setCheckable(True)
		self.start_button.clicked.connect(controller.start_camera)
		self.stop_button.clicked.connect(controller.stop_camera)
		self.pause_button.toggled.connect(controller.set_paused)
		self.stop_button.setEnabled(False)
		self.pause_button.setEnabled(False)

		buttons = QHBoxLayout()
		buttons.addWidget(self.start_button)
		buttons.addWidget(self.stop_button)
		buttons.addWidget(self.pause_button)
		buttons.addStretch()

		self.mode = StatusBadge("DIAGNOSTIC")
		self.tracking = StatusBadge("Tracking stopped")
		self.origin = StatusBadge("Origin unavailable")
		self.markers = QLabel("0")
		self.warning = QLabel("None")
		self.warning.setWordWrap(True)
		self.error = QLabel("None")
		self.error.setWordWrap(True)
		status = QFormLayout()
		status.addRow("Mode", self.mode)
		status.addRow("Tracking", self.tracking)
		status.addRow("Origin", self.origin)
		status.addRow("Detected markers", self.markers)
		status.addRow("Warnings", self.warning)
		status.addRow("Latest error", self.error)
		status_box = QGroupBox("Runtime status")
		status_box.setLayout(status)

		overlay_grid = QGridLayout()
		self.display_checks: dict[str, QCheckBox] = {}
		for index, option in enumerate(DISPLAY_OPTIONS):
			check = QCheckBox(option.removeprefix("show_").replace("_", " ").title())
			check.toggled.connect(
				lambda enabled, name=option: controller.execute(
					f"{name} {'on' if enabled else 'off'}"
				)
			)
			self.display_checks[option] = check
			overlay_grid.addWidget(check, index // 3, index % 3)
		overlay_box = QGroupBox("Processed-frame overlays")
		overlay_box.setLayout(overlay_grid)

		side = QVBoxLayout()
		side.addWidget(status_box)
		side.addWidget(overlay_box)
		side.addStretch()
		body = QHBoxLayout()
		body.addWidget(self.video, 1)
		body.addLayout(side)

		layout = QVBoxLayout(self)
		layout.addLayout(buttons)
		layout.addLayout(body, 1)

		controller.frame_ready.connect(self.video.set_bgr_frame)
		controller.state_updated.connect(self.update_state)
		controller.camera_state_changed.connect(self.update_camera_state)
		controller.log_emitted.connect(self.update_log)

	def update_log(self, level: str, message: str) -> None:
		if level == "error":
			self.error.setText(message)
		elif level == "success" and message.startswith("Camera started"):
			self.error.setText("None")
		elif level == "warning" and self.warning.text() == "None":
			self.warning.setText(message)

	def update_camera_state(self, running: bool, paused: bool) -> None:
		self.start_button.setEnabled(not running)
		self.stop_button.setEnabled(running)
		self.pause_button.setEnabled(running)
		with QSignalBlocker(self.pause_button):
			self.pause_button.setChecked(paused)
		self.pause_button.setText("Resume processing" if paused else "Pause processing")
		if not running:
			self.video.clear_frame()

	def update_state(self, state: dict) -> None:
		mode = state.get("mode", "UNKNOWN")
		self.mode.set_status(mode, "neutral")
		if state.get("tracking_active"):
			valid = bool(state.get("tracking_valid"))
			quality = float(state.get("tracking_quality", 0.0))
			self.tracking.set_status(
				f"{'Valid' if valid else 'Invalid'} · {quality:.2f}",
				"good" if valid else "error",
			)
		else:
			self.tracking.set_status("Tracking stopped", "neutral")
		if state.get("origin_reacquisition_required"):
			self.origin.set_status("Reacquisition required", "error")
		elif state.get("origin_valid"):
			self.origin.set_status("Origin valid", "good")
		elif state.get("reference_configured"):
			self.origin.set_status("Origin not tracked", "warning")
		else:
			self.origin.set_status("Not configured", "neutral")
		self.markers.setText(str(state.get("detected_marker_count", 0)))
		warnings = state.get("runtime_warnings", [])
		proposal_errors = state.get("proposal_errors", [])
		messages = [*proposal_errors, *warnings]
		self.warning.setText("\n".join(messages[-4:]) if messages else "None")
		for name, enabled in state.get("display_options", {}).items():
			check = self.display_checks.get(name)
			if check is not None:
				with QSignalBlocker(check):
					check.setChecked(bool(enabled))
