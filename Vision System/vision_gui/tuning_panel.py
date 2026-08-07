"""Validated editor for existing camera-config.yaml parameters."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QPushButton,
	QScrollArea,
	QSpinBox,
	QVBoxLayout,
	QWidget,
)

from processing import config_from_mapping, config_to_mapping, load_default_config

from .common import StatusBadge


def _get_path(mapping: dict, path: tuple[str, ...]) -> Any:
	value: Any = mapping
	for part in path:
		value = value[part]
	return value


def _set_path(mapping: dict, path: tuple[str, ...], value: Any) -> None:
	target = mapping
	for part in path[:-1]:
		target = target[part]
	target[path[-1]] = value


class HsvEditor(QWidget):
	changed = Signal()

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		layout = QHBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		self.values: list[QSpinBox] = []
		for index, label in enumerate(("H", "S", "V")):
			spin = QSpinBox()
			spin.setRange(0, 180 if index == 0 else 255)
			spin.setPrefix(f"{label} ")
			spin.valueChanged.connect(self.changed)
			self.values.append(spin)
			layout.addWidget(spin)

	def value(self) -> list[int]:
		return [spin.value() for spin in self.values]

	def set_value(self, values) -> None:
		with QSignalBlocker(self):
			for spin, value in zip(self.values, values):
				spin.setValue(int(value))


class OptionalSpinBox(QSpinBox):
	def __init__(
		self, maximum: int = 100000, special_text: str = "Automatic", parent=None
	) -> None:
		super().__init__(parent)
		self.setRange(-1, maximum)
		self.setSpecialValueText(special_text)

	def optional_value(self):
		return None if self.value() < 0 else self.value()

	def set_optional_value(self, value) -> None:
		self.setValue(-1 if value is None else int(value))


@dataclass
class _Binding:
	path: tuple[str, ...]
	widget: QWidget
	read: Callable[[], Any]
	write: Callable[[Any], None]


class TuningPanel(QWidget):
	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self._bindings: list[_Binding] = []
		self._applied: dict | None = None
		self._saved: dict | None = None
		self._editor_base: dict | None = None

		self.edited_badge = StatusBadge("Waiting")
		self.applied_badge = StatusBadge("Waiting")
		self.saved_badge = StatusBadge("Waiting")
		badges = QHBoxLayout()
		badges.addWidget(QLabel("Editor"))
		badges.addWidget(self.edited_badge)
		badges.addWidget(QLabel("Runtime"))
		badges.addWidget(self.applied_badge)
		badges.addWidget(QLabel("YAML"))
		badges.addWidget(self.saved_badge)
		badges.addStretch()

		form_root = QWidget()
		form_layout = QVBoxLayout(form_root)
		form_layout.addWidget(self._camera_group())
		form_layout.addWidget(self._calibration_group())
		form_layout.addWidget(self._circle_group())
		form_layout.addWidget(self._ransac_group())
		form_layout.addWidget(self._tracking_group())
		form_layout.addWidget(self._geometry_group())
		form_layout.addWidget(self._dataset_group())
		form_layout.addStretch()
		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setWidget(form_root)

		self.apply = QPushButton("Apply")
		self.revert = QPushButton("Revert unapplied changes")
		self.reload = QPushButton("Reload from YAML")
		self.defaults = QPushButton("Restore defaults")
		self.save = QPushButton("Save configuration")
		self.apply.clicked.connect(self._apply)
		self.revert.clicked.connect(self._revert)
		self.reload.clicked.connect(controller.reload_config)
		self.defaults.clicked.connect(self._restore_defaults)
		self.save.clicked.connect(lambda: controller.execute("save_config"))
		buttons = QHBoxLayout()
		for button in (self.apply, self.revert, self.reload, self.defaults, self.save):
			buttons.addWidget(button)
		buttons.addStretch()

		warning = QLabel(
			"Camera settings restart the camera. Resolution, reference counts, versions, "
			"and geometry mode cannot be changed while a reference profile is loaded. "
			"Restore Defaults changes the editor only; press Apply to use them."
		)
		warning.setWordWrap(True)

		layout = QVBoxLayout(self)
		layout.addLayout(badges)
		layout.addWidget(warning)
		layout.addWidget(scroll, 1)
		layout.addLayout(buttons)

		controller.configuration_applied.connect(self.set_applied)
		controller.configuration_saved.connect(self.set_saved)
		self._refresh_status()

	def _group(self, title: str, rows: list[tuple[str, QWidget]]) -> QGroupBox:
		form = QFormLayout()
		for label, widget in rows:
			form.addRow(label, widget)
		box = QGroupBox(title)
		box.setLayout(form)
		return box

	def _bind(self, path: str, widget: QWidget, read, write, changed_signal) -> QWidget:
		self._bindings.append(_Binding(tuple(path.split(".")), widget, read, write))
		changed_signal.connect(self._edited)
		return widget

	def _int(self, path: str, minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
		widget = QSpinBox()
		widget.setRange(minimum, maximum)
		widget.setSuffix(suffix)
		return self._bind(path, widget, widget.value, widget.setValue, widget.valueChanged)

	def _optional_int(
		self, path: str, maximum: int, special_text: str = "Automatic"
	) -> OptionalSpinBox:
		widget = OptionalSpinBox(maximum, special_text)
		return self._bind(
			path, widget, widget.optional_value, widget.set_optional_value, widget.valueChanged
		)

	def _double(
		self, path: str, minimum: float, maximum: float,
		decimals: int = 3, suffix: str = "",
	) -> QDoubleSpinBox:
		widget = QDoubleSpinBox()
		widget.setRange(minimum, maximum)
		widget.setDecimals(decimals)
		widget.setSingleStep(0.05 if maximum <= 10 else 1.0)
		widget.setSuffix(suffix)
		return self._bind(path, widget, widget.value, widget.setValue, widget.valueChanged)

	def _bool(self, path: str) -> QCheckBox:
		widget = QCheckBox()
		return self._bind(path, widget, widget.isChecked, widget.setChecked, widget.toggled)

	def _combo(self, path: str, choices: list[tuple[str, str]]) -> QComboBox:
		widget = QComboBox()
		for label, value in choices:
			widget.addItem(label, value)

		def write(value) -> None:
			index = widget.findData(value)
			if index >= 0:
				widget.setCurrentIndex(index)

		return self._bind(
			path, widget, widget.currentData, write, widget.currentIndexChanged
		)

	def _text(self, path: str) -> QLineEdit:
		widget = QLineEdit()
		return self._bind(path, widget, widget.text, lambda value: widget.setText(str(value)), widget.textChanged)

	def _hsv(self, path: str) -> HsvEditor:
		widget = HsvEditor()
		return self._bind(path, widget, widget.value, widget.set_value, widget.changed)

	def _preset(self) -> QComboBox:
		widget = QComboBox()

		def write(value) -> None:
			if self._applied is not None:
				for name in self._applied.get("presets", {}):
					if widget.findData(name) < 0:
						widget.addItem(name, name)
			index = widget.findData(value)
			if index >= 0:
				widget.setCurrentIndex(index)

		return self._bind(
			"calibration.preset", widget, widget.currentData, write,
			widget.currentIndexChanged,
		)

	def _camera_group(self) -> QGroupBox:
		return self._group("Camera", [
			("Width", self._int("camera.width", 160, 4096, " px")),
			("Height", self._int("camera.height", 120, 2160, " px")),
			("Frame rate", self._int("camera.fps", 1, 240, " fps")),
			("Automatic exposure", self._bool("camera.auto_exposure")),
			("Exposure", self._optional_int("camera.exposure", 100000)),
			("Gain", self._optional_int("camera.gain", 10000)),
		])

	def _calibration_group(self) -> QGroupBox:
		return self._group("Blob and HSV calibration", [
			("Minimum blob area", self._int("calibration.min_area", 1, 100000, " px²")),
			("Maximum blob area", self._optional_int(
				"calibration.max_area", 100000, "No maximum"
			)),
			("Lower area ratio", self._double("calibration.area_low_ratio", 0, 10, 3)),
			("Upper area ratio", self._double("calibration.area_up_ratio", 0, 10, 3)),
			("Hue margin", self._int("calibration.hue_margin", 0, 180)),
			("Saturation margin", self._int("calibration.sat_margin", 0, 255)),
			("Value margin", self._int("calibration.val_margin", 0, 255)),
			("Preset", self._preset()),
			("Red range 1 lower", self._hsv("calibration.red_lower_1")),
			("Red range 1 upper", self._hsv("calibration.red_upper_1")),
			("Red range 2 lower", self._hsv("calibration.red_lower_2")),
			("Red range 2 upper", self._hsv("calibration.red_upper_2")),
		])

	def _circle_group(self) -> QGroupBox:
		return self._group("Hough circle detection", [
			("Accumulator resolution (dp)", self._int("hough_circles.dp", 1, 10)),
			("Minimum centre distance", self._int("hough_circles.min_dist", 1, 1000, " px")),
			("Canny threshold", self._int("hough_circles.param1", 1, 1000)),
			("Accumulator threshold", self._int("hough_circles.param2", 1, 1000)),
			("Minimum radius", self._int("hough_circles.min_radius", 0, 1000, " px")),
			("Maximum radius", self._int("hough_circles.max_radius", 0, 1000, " px")),
			("Required for reference", self._bool("hough_circles.require_validation_for_reference")),
			("Required for tracking", self._bool("hough_circles.require_validation_for_tracking")),
		])

	def _ransac_group(self) -> QGroupBox:
		return self._group("Reference RANSAC", [
			("Expected LINE_A points", self._int("reference_ransac.expected_points_line_a", 1, 100)),
			("Expected LINE_B points", self._int("reference_ransac.expected_points_line_b", 1, 100)),
			("Distance threshold", self._double("reference_ransac.distance_threshold_px", 0.01, 1000, 2, " px")),
			("Iterations", self._int("reference_ransac.iterations", 1, 100000)),
			("Minimum inliers per line", self._int("reference_ransac.minimum_inliers_per_line", 1, 100)),
			("Maximum angle difference", self._double("reference_ransac.maximum_angle_difference_deg", 0, 180, 2, "°")),
			("Maximum mean residual", self._double("reference_ransac.maximum_mean_residual_px", 0, 1000, 2, " px")),
			("Maximum outlier count", self._int("reference_ransac.maximum_outlier_count", 0, 1000)),
			("Minimum line separation", self._double("reference_ransac.minimum_line_separation_px", 0, 5000, 2, " px")),
			("Minimum order spacing", self._double("reference_ransac.minimum_order_spacing_px", 0, 1000, 2, " px")),
			("Direction", self._combo("reference_ransac.direction_rule", [
				("Base to tip", "base_to_tip"), ("Left to right", "left_to_right"),
				("Right to left", "right_to_left"), ("Top to bottom", "top_to_bottom"),
				("Bottom to top", "bottom_to_top"),
			])),
		])

	def _tracking_group(self) -> QGroupBox:
		return self._group("Tracking and origin quality", [
			("Expected deforming points", self._int("tracking.expected_deforming_point_count", 1, 200)),
			("Maximum assignment distance", self._double("tracking.maximum_assignment_distance", 0.1, 1000, 2, " px")),
			("Maximum missing frames", self._int("tracking.maximum_missing_frames", 0, 10000)),
			("Maximum optical-flow error", self._double("tracking.maximum_optical_flow_error", 0.01, 10000, 2)),
			("Tracked-only minimum quality", self._double("tracking.minimum_tracked_only_quality", 0, 1, 3)),
			("Allow tracked-only points", self._bool("tracking.allow_tracked_only_points")),
			("Maximum adjacent distance ratio", self._double("tracking.maximum_adjacent_distance_ratio", 1, 100, 3)),
			("Minimum origin quality", self._double("reference.minimum_origin_tracking_quality", 0, 1, 3)),
			("Allow tracked-only origin", self._bool("reference.allow_tracked_only_origin")),
			("Origin selection distance", self._double("reference.origin_selection_max_distance_px", 0, 1000, 2, " px")),
			("Origin assignment distance", self._double("reference.origin_maximum_assignment_distance", 0, 1000, 2, " px")),
			("Origin reacquisition distance", self._double("reference.origin_reacquisition_max_distance_px", 0, 1000, 2, " px")),
			("Origin maximum missing frames", self._int("reference.origin_maximum_missing_frames", 0, 10000)),
			("Origin reacquisition policy", self._combo("reference.origin_reacquisition_policy", [
				("Manual confirmation", "manual_confirmation"),
				("Automatic proximity", "automatic_proximity"),
			])),
		])

	def _geometry_group(self) -> QGroupBox:
		return self._group("Geometry and quality", [
			("Reference mode", self._combo("geometry.geometry_reference_mode", [
				("Translation only", "translation_only"),
				("Translation and rotation", "translation_and_rotation"),
			])),
			("Transform RANSAC threshold", self._double("geometry.transform_ransac_threshold_px", 0.01, 1000, 2, " px")),
			("Minimum deforming quality", self._double("geometry.minimum_deforming_tracking_quality", 0, 1, 3)),
			("Minimum rigid-reference quality", self._double("geometry.minimum_rigid_reference_tracking_quality", 0, 1, 3)),
			("Allow tracked-only deforming", self._bool("geometry.allow_tracked_only_deforming")),
			("Allow tracked-only rigid reference", self._bool("geometry.allow_tracked_only_rigid_reference")),
			("Allow scale compensation", self._bool("geometry.allow_scale_compensation")),
		])

	def _dataset_group(self) -> QGroupBox:
		return self._group("Dataset sampling", [
			("Dataset directory", self._text("dataset.directory")),
			("Default frames per sample", self._int("dataset.frames_per_sample", 1, 100000)),
			("Retry invalid frames", self._bool("dataset.retry_invalid_frames")),
			("Maximum invalid retries", self._int("dataset.maximum_invalid_frame_retries", 1, 100000)),
			("Save raw images", self._bool("dataset.save_images")),
		])

	def _write_widgets(self, mapping: dict) -> None:
		for binding in self._bindings:
			try:
				value = _get_path(mapping, binding.path)
			except KeyError:
				continue
			with QSignalBlocker(binding.widget):
				binding.write(value)

	def _collect(self) -> dict:
		if self._applied is None or self._editor_base is None:
			raise RuntimeError("Applied configuration is not available yet")
		mapping = copy.deepcopy(self._editor_base)
		for binding in self._bindings:
			_set_path(mapping, binding.path, binding.read())
		return mapping

	def _edited(self, *_args) -> None:
		self._refresh_status()

	def _refresh_status(self) -> None:
		if self._applied is None:
			self.edited_badge.set_status("Waiting", "neutral")
			self.applied_badge.set_status("Waiting", "neutral")
			self.saved_badge.set_status("Waiting", "neutral")
			for button in (self.apply, self.revert, self.save):
				button.setEnabled(False)
			return
		try:
			edited = self._collect()
		except Exception:
			edited = None
		dirty = edited != self._applied
		unsaved = self._saved is None or self._applied != self._saved
		self.edited_badge.set_status(
			"Unapplied edits" if dirty else "Matches runtime",
			"warning" if dirty else "good",
		)
		self.applied_badge.set_status(
			"Applied, unsaved" if unsaved else "Applied",
			"warning" if unsaved else "good",
		)
		self.saved_badge.set_status(
			"Different" if unsaved else "Matches runtime",
			"warning" if unsaved else "good",
		)
		self.apply.setEnabled(dirty)
		self.revert.setEnabled(dirty)
		self.save.setEnabled(not dirty)

	def set_applied(self, mapping: dict) -> None:
		self._applied = copy.deepcopy(mapping)
		self._editor_base = copy.deepcopy(mapping)
		self._write_widgets(self._applied)
		self._refresh_status()

	def set_saved(self, mapping: dict) -> None:
		self._saved = copy.deepcopy(mapping)
		self._refresh_status()

	def _apply(self) -> None:
		try:
			mapping = self._collect()
			config_from_mapping(mapping)
		except Exception as exc:
			self.controller.log_emitted.emit("error", f"Configuration validation failed: {exc}")
			return
		self.controller.apply_config(mapping)

	def _revert(self) -> None:
		if self._applied is not None:
			self._editor_base = copy.deepcopy(self._applied)
			self._write_widgets(self._applied)
			self._refresh_status()

	def _restore_defaults(self) -> None:
		try:
			mapping = config_to_mapping(load_default_config())
		except Exception as exc:
			self.controller.log_emitted.emit("error", f"Could not load defaults: {exc}")
			return
		self._editor_base = copy.deepcopy(mapping)
		self._write_widgets(mapping)
		self._refresh_status()
		self.controller.log_emitted.emit("info", "Factory defaults loaded into the editor.")
