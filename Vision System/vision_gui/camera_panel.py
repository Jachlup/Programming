"""Live camera view, runtime status, and overlay controls."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtWidgets import (
	QCheckBox,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QSlider,
	QSpinBox,
	QTabWidget,
	QVBoxLayout,
	QWidget,
)

from .common import StatusBadge, VideoLabel
from .blue_blob_panel import BlueBlobPanel
from .reference_panel import ReferencePanel


DISPLAY_OPTIONS = (
	"show_points", "show_lines", "show_outliers", "show_geometry",
	"show_features", "show_circles", "show_status", "show_warnings",
	"show_mask_only",
	"show_blue_mask_only",
)


class CameraPanel(QWidget):
	open_tuning_requested = Signal()

	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self._camera_running = False
		self._state: dict = {}
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

		self.calibration_status = StatusBadge("Calibration off")
		self.calibration_sample = QLabel("—")
		self.calibration_area = QLabel("—")
		self.calibration_range = QLabel("—")
		self.calibration_enable = QPushButton("Enable calibration")
		self.calibration_disable = QPushButton("Disable calibration")
		self.calibration_clear = QPushButton("Clear selected blob")
		self.calibration_print_status = QPushButton("Print status")
		self.calibration_tuning = QPushButton("Open tuning")
		self.calibration_save = QPushButton("Save calibration")
		self.calibration_mask_only = QPushButton("Mask-only preview")
		self.calibration_mask_only.setCheckable(True)
		self.calibration_enable.clicked.connect(
			lambda: controller.execute("calibration_start")
		)
		self.calibration_disable.clicked.connect(
			lambda: controller.execute("calibration_stop")
		)
		self.calibration_clear.clicked.connect(
			lambda: controller.execute("calibration_clear")
		)
		self.calibration_print_status.clicked.connect(
			lambda: controller.execute("calibration_status")
		)
		self.calibration_tuning.clicked.connect(self.open_tuning_requested)
		self.calibration_save.clicked.connect(self._save_blob_area_limits)
		self.calibration_mask_only.toggled.connect(
			lambda enabled: controller.execute(
				f"show_mask_only {'on' if enabled else 'off'}"
			)
		)
		calibration_grid = QGridLayout()
		calibration_grid.addWidget(self.calibration_status, 0, 0, 1, 3)
		calibration_grid.addWidget(QLabel("Sample HSV"), 1, 0)
		calibration_grid.addWidget(self.calibration_sample, 1, 1, 1, 2)
		calibration_grid.addWidget(QLabel("Selected area"), 2, 0)
		calibration_grid.addWidget(self.calibration_area, 2, 1, 1, 2)
		calibration_grid.addWidget(QLabel("Applied area range"), 3, 0)
		calibration_grid.addWidget(self.calibration_range, 3, 1, 1, 2)
		calibration_grid.addWidget(self.calibration_enable, 4, 0, 1, 2)
		calibration_grid.addWidget(self.calibration_disable, 4, 2)
		calibration_grid.addWidget(self.calibration_clear, 5, 0, 1, 2)
		calibration_grid.addWidget(self.calibration_print_status, 5, 2)
		calibration_grid.addWidget(self.calibration_mask_only, 6, 0, 1, 3)
		calibration_grid.addWidget(self.calibration_tuning, 7, 0)
		calibration_grid.addWidget(self.calibration_save, 7, 1, 1, 2)
		calibration_hint = QLabel(
			"Enable, then click a detected red blob in the video. "
			"The active preset is applied first, matching main.py."
		)
		calibration_hint.setWordWrap(True)
		calibration_grid.addWidget(calibration_hint, 8, 0, 1, 3)
		calibration_box = QGroupBox("Click calibration")
		calibration_box.setLayout(calibration_grid)

		self.minimum_area_slider = QSlider(Qt.Orientation.Horizontal)
		self.minimum_area_slider.setRange(0, 1_000)
		self.minimum_area_spin = QSpinBox()
		self.minimum_area_spin.setRange(0, 1_000)
		self.minimum_area_spin.setSuffix(" px²")
		self.maximum_area_slider = QSlider(Qt.Orientation.Horizontal)
		self.maximum_area_slider.setRange(0, 1_000)
		self.maximum_area_spin = QSpinBox()
		self.maximum_area_spin.setRange(0, 1_000)
		self.maximum_area_spin.setSpecialValueText("No maximum")
		self.maximum_area_spin.setSuffix(" px²")
		self.minimum_area_slider.valueChanged.connect(self.minimum_area_spin.setValue)
		self.minimum_area_spin.valueChanged.connect(self.minimum_area_slider.setValue)
		self.maximum_area_slider.valueChanged.connect(self.maximum_area_spin.setValue)
		self.maximum_area_spin.valueChanged.connect(self.maximum_area_slider.setValue)
		self._area_apply_timer = QTimer(self)
		self._area_apply_timer.setSingleShot(True)
		self._area_apply_timer.setInterval(80)
		self._area_apply_timer.timeout.connect(self._apply_blob_area_limits)
		self._pending_area_limits: tuple[int, int | None] | None = None
		self.minimum_area_spin.valueChanged.connect(self._queue_blob_area_limits)
		self.maximum_area_spin.valueChanged.connect(self._queue_blob_area_limits)
		area_grid = QGridLayout()
		area_grid.addWidget(QLabel("Minimum area"), 0, 0)
		area_grid.addWidget(self.minimum_area_slider, 0, 1)
		area_grid.addWidget(self.minimum_area_spin, 0, 2)
		area_grid.addWidget(QLabel("Maximum area"), 1, 0)
		area_grid.addWidget(self.maximum_area_slider, 1, 1)
		area_grid.addWidget(self.maximum_area_spin, 1, 2)
		area_note = QLabel("Set maximum to 0 to disable the upper-area limit.")
		area_note.setWordWrap(True)
		area_grid.addWidget(area_note, 2, 0, 1, 3)
		area_box = QGroupBox("Blob size filter")
		area_box.setLayout(area_grid)

		runtime_page = QWidget()
		runtime_layout = QVBoxLayout(runtime_page)
		runtime_layout.addWidget(status_box)
		runtime_layout.addWidget(overlay_box)
		runtime_layout.addStretch()
		calibration_page = QWidget()
		calibration_layout = QVBoxLayout(calibration_page)
		calibration_layout.addWidget(area_box)
		calibration_layout.addWidget(calibration_box)
		calibration_layout.addStretch()
		self.blue_blob_panel = BlueBlobPanel(controller)
		self.blue_blob_panel.open_tuning_requested.connect(
			self.open_tuning_requested.emit
		)
		self.reference_panel = ReferencePanel(controller, compact=True)
		self.control_tabs = QTabWidget()
		self.control_tabs.setMinimumWidth(460)
		self.control_tabs.setMaximumWidth(560)
		self.control_tabs.addTab(runtime_page, "Runtime")
		self.control_tabs.addTab(calibration_page, "Calibration")
		self.control_tabs.addTab(self.blue_blob_panel, "Blue blob")
		self.control_tabs.addTab(self.reference_panel, "Reference setup")
		body = QHBoxLayout()
		body.addWidget(self.video, 1)
		body.addWidget(self.control_tabs)

		layout = QVBoxLayout(self)
		layout.addLayout(buttons)
		layout.addLayout(body, 1)

		controller.frame_ready.connect(self.video.set_bgr_frame)
		controller.state_updated.connect(self.update_state)
		controller.camera_state_changed.connect(self.update_camera_state)
		controller.log_emitted.connect(self.update_log)

	def _queue_blob_area_limits(self, _value: int) -> None:
		minimum = self.minimum_area_spin.value()
		maximum_value = self.maximum_area_spin.value()
		self._pending_area_limits = (
			minimum, None if maximum_value == 0 else maximum_value
		)
		self._area_apply_timer.start()

	def _apply_blob_area_limits(self) -> None:
		if self._pending_area_limits is None:
			return
		minimum, maximum = self._pending_area_limits
		maximum_value = 0 if maximum is None else maximum
		if maximum_value != 0 and maximum_value < minimum:
			maximum_value = minimum
			with QSignalBlocker(self.maximum_area_slider):
				self.maximum_area_slider.setValue(maximum_value)
			with QSignalBlocker(self.maximum_area_spin):
				self.maximum_area_spin.setValue(maximum_value)
		self._pending_area_limits = (
			minimum, None if maximum_value == 0 else maximum_value
		)
		self.controller.set_blob_area_limits(
			minimum, None if maximum_value == 0 else maximum_value
		)

	def _save_blob_area_limits(self) -> None:
		"""Flush a pending slider edit before queueing the YAML save command."""
		if self._area_apply_timer.isActive():
			self._area_apply_timer.stop()
			self._apply_blob_area_limits()
		self.controller.execute("save_config")

	def update_log(self, level: str, message: str) -> None:
		if level == "error":
			self.error.setText(message)
		elif level == "success" and message.startswith("Camera started"):
			self.error.setText("None")
		elif level == "warning" and self.warning.text() == "None":
			self.warning.setText(message)

	def update_camera_state(self, running: bool, paused: bool) -> None:
		self._camera_running = running
		self.start_button.setEnabled(not running)
		self.stop_button.setEnabled(running)
		self.pause_button.setEnabled(running)
		with QSignalBlocker(self.pause_button):
			self.pause_button.setChecked(paused)
		self.pause_button.setText("Resume processing" if paused else "Pause processing")
		if not running:
			self.video.clear_frame()
		self.blue_blob_panel.update_camera_state(running)
		if self._state:
			self.update_state(self._state)

	def update_state(self, state: dict) -> None:
		self._state = state
		self.blue_blob_panel.update_state(state)
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
		mask_only = bool(state.get("display_options", {}).get("show_mask_only", False))
		with QSignalBlocker(self.calibration_mask_only):
			self.calibration_mask_only.setChecked(mask_only)
		self.calibration_mask_only.setText(
			"Mask-only preview: ON" if mask_only else "Mask-only preview: OFF"
		)

		calibration_mode = bool(state.get("calibration_mode"))
		selection = bool(state.get("calibration_selection_available"))
		if calibration_mode:
			self.calibration_status.set_status("Click a red blob", "warning")
		elif selection:
			self.calibration_status.set_status("Calibration applied", "good")
		else:
			self.calibration_status.set_status("Calibration off", "neutral")
		sample = state.get("calibration_sample_hsv")
		area = state.get("calibration_blob_area")
		minimum = state.get("calibration_min_area")
		maximum = state.get("calibration_max_area")
		self.calibration_sample.setText("—" if sample is None else str(tuple(sample)))
		self.calibration_area.setText("—" if area is None else f"{float(area):.1f} px²")
		self.calibration_range.setText(
			f"{minimum} .. {'unlimited' if maximum is None else maximum} px²"
		)
		minimum_value = max(0, min(int(minimum or 0), 1_000))
		maximum_value = 0 if maximum is None else max(
			minimum_value, min(int(maximum), 1_000)
		)
		confirmed_limits = (
			minimum_value, None if maximum_value == 0 else maximum_value
		)
		if self._pending_area_limits == confirmed_limits:
			self._pending_area_limits = None
		if self._pending_area_limits is None:
			for widget, value in (
				(self.minimum_area_slider, minimum_value),
				(self.minimum_area_spin, minimum_value),
				(self.maximum_area_slider, maximum_value),
				(self.maximum_area_spin, maximum_value),
			):
				with QSignalBlocker(widget):
					widget.setValue(value)
		other_mouse_mode = bool(
			state.get("reference_origin_selection")
			or state.get("origin_reacquisition_selection")
			or state.get("blue_calibration_mode")
		)
		self.calibration_enable.setEnabled(
			self._camera_running and not calibration_mode and not other_mouse_mode
		)
		self.calibration_disable.setEnabled(calibration_mode)
		self.calibration_clear.setEnabled(selection)
		self.calibration_save.setEnabled(True)
