"""Dedicated calibration and runtime controls for the single blue target."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtWidgets import (
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QLabel,
	QPushButton,
	QSlider,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from .common import StatusBadge


class BlueBlobPanel(QWidget):
	open_tuning_requested = Signal()

	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self._camera_running = False
		self._pending_area_limits: tuple[int, int | None] | None = None

		self.calibration_status = StatusBadge("Calibration off")
		self.tracking_status = StatusBadge("Target not selected")
		self.candidates = QLabel("0")
		self.image_position = QLabel("—")
		self.relative_position = QLabel("—")
		self.relative_reason = QLabel("Select the target, then track ORIGIN")
		self.relative_reason.setWordWrap(True)
		self.tracking_quality = QLabel("0.00")
		self.tracking_reason = QLabel("—")
		self.tracking_reason.setWordWrap(True)
		self.sample_hsv = QLabel("—")
		self.selected_area = QLabel("—")
		self.applied_range = QLabel("—")

		status_form = QFormLayout()
		status_form.addRow("Calibration", self.calibration_status)
		status_form.addRow("Tracking", self.tracking_status)
		status_form.addRow("Blue candidates", self.candidates)
		status_form.addRow("Image position", self.image_position)
		status_form.addRow("From ORIGIN (dx, dy)", self.relative_position)
		status_form.addRow("Unavailable reason", self.relative_reason)
		status_form.addRow("Tracking quality", self.tracking_quality)
		status_form.addRow("Invalid reason", self.tracking_reason)
		status_box = QGroupBox("Single blue target")
		status_box.setLayout(status_form)

		self.enable = QPushButton("Enable blue calibration")
		self.disable = QPushButton("Disable calibration")
		self.clear = QPushButton("Clear blue target")
		self.print_status = QPushButton("Print blue status")
		self.open_tuning = QPushButton("Open tuning")
		self.save = QPushButton("Save blue calibration")
		self.mask_only = QPushButton("Blue mask-only preview: OFF")
		self.mask_only.setCheckable(True)
		self.enable.clicked.connect(
			lambda: controller.execute("blue_calibration_start")
		)
		self.disable.clicked.connect(
			lambda: controller.execute("blue_calibration_stop")
		)
		self.clear.clicked.connect(
			lambda: controller.execute("blue_calibration_clear")
		)
		self.print_status.clicked.connect(
			lambda: controller.execute("blue_blob_status")
		)
		self.open_tuning.clicked.connect(self.open_tuning_requested)
		self.save.clicked.connect(self._save_calibration)
		self.mask_only.toggled.connect(
			lambda enabled: controller.execute(
				f"show_blue_mask_only {'on' if enabled else 'off'}"
			)
		)

		calibration_grid = QGridLayout()
		calibration_grid.addWidget(QLabel("Sample HSV"), 0, 0)
		calibration_grid.addWidget(self.sample_hsv, 0, 1, 1, 2)
		calibration_grid.addWidget(QLabel("Selected area"), 1, 0)
		calibration_grid.addWidget(self.selected_area, 1, 1, 1, 2)
		calibration_grid.addWidget(QLabel("Applied area range"), 2, 0)
		calibration_grid.addWidget(self.applied_range, 2, 1, 1, 2)
		calibration_grid.addWidget(self.enable, 3, 0, 1, 2)
		calibration_grid.addWidget(self.disable, 3, 2)
		calibration_grid.addWidget(self.clear, 4, 0, 1, 2)
		calibration_grid.addWidget(self.print_status, 4, 2)
		calibration_grid.addWidget(self.mask_only, 5, 0, 1, 3)
		calibration_grid.addWidget(self.open_tuning, 6, 0)
		calibration_grid.addWidget(self.save, 6, 1, 1, 2)
		hint = QLabel(
			"Enable calibration, then click the one blue blob in the live image. "
			"That click calibrates its color and size and starts BLUE_TARGET tracking."
		)
		hint.setWordWrap(True)
		calibration_grid.addWidget(hint, 7, 0, 1, 3)
		calibration_box = QGroupBox("Click calibration and target selection")
		calibration_box.setLayout(calibration_grid)

		self.minimum_area_slider = QSlider(Qt.Orientation.Horizontal)
		self.minimum_area_slider.setRange(0, 5_000)
		self.minimum_area_spin = QSpinBox()
		self.minimum_area_spin.setRange(0, 5_000)
		self.minimum_area_spin.setSuffix(" px²")
		self.maximum_area_slider = QSlider(Qt.Orientation.Horizontal)
		self.maximum_area_slider.setRange(0, 5_000)
		self.maximum_area_spin = QSpinBox()
		self.maximum_area_spin.setRange(0, 5_000)
		self.maximum_area_spin.setSpecialValueText("No maximum")
		self.maximum_area_spin.setSuffix(" px²")
		self.minimum_area_slider.valueChanged.connect(
			self.minimum_area_spin.setValue
		)
		self.minimum_area_spin.valueChanged.connect(
			self.minimum_area_slider.setValue
		)
		self.maximum_area_slider.valueChanged.connect(
			self.maximum_area_spin.setValue
		)
		self.maximum_area_spin.valueChanged.connect(
			self.maximum_area_slider.setValue
		)
		self._area_apply_timer = QTimer(self)
		self._area_apply_timer.setSingleShot(True)
		self._area_apply_timer.setInterval(80)
		self._area_apply_timer.timeout.connect(self._apply_area_limits)
		self.minimum_area_spin.valueChanged.connect(self._queue_area_limits)
		self.maximum_area_spin.valueChanged.connect(self._queue_area_limits)
		area_grid = QGridLayout()
		area_grid.addWidget(QLabel("Minimum area"), 0, 0)
		area_grid.addWidget(self.minimum_area_slider, 0, 1)
		area_grid.addWidget(self.minimum_area_spin, 0, 2)
		area_grid.addWidget(QLabel("Maximum area"), 1, 0)
		area_grid.addWidget(self.maximum_area_slider, 1, 1)
		area_grid.addWidget(self.maximum_area_spin, 1, 2)
		note = QLabel("Set maximum to 0 to disable the upper-area limit.")
		note.setWordWrap(True)
		area_grid.addWidget(note, 2, 0, 1, 3)
		area_box = QGroupBox("Blue blob size filter")
		area_box.setLayout(area_grid)

		layout = QVBoxLayout(self)
		layout.addWidget(status_box)
		layout.addWidget(area_box)
		layout.addWidget(calibration_box)
		layout.addStretch()

	def _queue_area_limits(self, _value: int) -> None:
		maximum = self.maximum_area_spin.value()
		self._pending_area_limits = (
			self.minimum_area_spin.value(),
			None if maximum == 0 else maximum,
		)
		self._area_apply_timer.start()

	def _apply_area_limits(self) -> None:
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
		self.controller.set_blue_blob_area_limits(
			minimum, None if maximum_value == 0 else maximum_value
		)

	def _save_calibration(self) -> None:
		if self._area_apply_timer.isActive():
			self._area_apply_timer.stop()
			self._apply_area_limits()
		self.controller.execute("save_config")

	def update_camera_state(self, running: bool) -> None:
		self._camera_running = bool(running)

	def update_state(self, state: dict) -> None:
		calibration_mode = bool(state.get("blue_calibration_mode"))
		selected = bool(state.get("blue_target_selected"))
		valid = bool(state.get("blue_tracking_valid"))
		tracking_status = str(state.get("blue_tracking_status", "UNSELECTED"))
		if calibration_mode:
			self.calibration_status.set_status("Click the blue blob", "warning")
		elif selected:
			self.calibration_status.set_status("Target calibrated", "good")
		else:
			self.calibration_status.set_status("Calibration off", "neutral")
		if valid:
			self.tracking_status.set_status(tracking_status, "good")
		elif selected:
			self.tracking_status.set_status(tracking_status, "error")
		else:
			self.tracking_status.set_status("Target not selected", "neutral")

		self.candidates.setText(str(state.get("detected_blue_candidate_count", 0)))
		position = state.get("blue_position")
		self.image_position.setText(
			"—" if position is None
			else f"({float(position[0]):.2f}, {float(position[1]):.2f}) px"
		)
		relative = state.get("blue_origin_relative_position")
		self.relative_position.setText(
			"—" if relative is None
			else f"({float(relative[0]):+.2f}, {float(relative[1]):+.2f}) px"
		)
		self.relative_reason.setText(
			"—" if relative is not None
			else str(state.get("blue_relative_unavailable_reason", "Unavailable"))
		)
		self.tracking_quality.setText(
			f"{float(state.get('blue_tracking_quality', 0.0)):.2f}"
		)
		self.tracking_reason.setText(
			str(state.get("blue_tracking_reason") or "—")
		)
		sample = state.get("blue_calibration_sample_hsv")
		area = state.get("blue_calibration_blob_area")
		minimum = int(state.get("blue_calibration_min_area", 0) or 0)
		maximum = state.get("blue_calibration_max_area")
		self.sample_hsv.setText("—" if sample is None else str(tuple(sample)))
		self.selected_area.setText(
			"—" if area is None else f"{float(area):.1f} px²"
		)
		self.applied_range.setText(
			f"{minimum} .. {'unlimited' if maximum is None else maximum} px²"
		)

		minimum_value = max(0, min(minimum, 5_000))
		maximum_value = 0 if maximum is None else max(
			minimum_value, min(int(maximum), 5_000)
		)
		confirmed = (
			minimum_value, None if maximum_value == 0 else maximum_value
		)
		if self._pending_area_limits == confirmed:
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

		blue_mask = bool(
			state.get("display_options", {}).get("show_blue_mask_only", False)
		)
		with QSignalBlocker(self.mask_only):
			self.mask_only.setChecked(blue_mask)
		self.mask_only.setText(
			"Blue mask-only preview: ON"
			if blue_mask else "Blue mask-only preview: OFF"
		)
		other_mouse_mode = bool(
			state.get("calibration_mode")
			or state.get("reference_origin_selection")
			or state.get("origin_reacquisition_selection")
		)
		self.enable.setEnabled(
			self._camera_running and not calibration_mode and not other_mouse_mode
		)
		self.disable.setEnabled(calibration_mode)
		self.clear.setEnabled(selected)
		self.save.setEnabled(True)
