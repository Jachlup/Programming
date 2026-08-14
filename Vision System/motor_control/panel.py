"""Operator-facing motor controls and detached live status display."""

from __future__ import annotations

from datetime import datetime
import time

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QMessageBox,
	QPlainTextEdit,
	QPushButton,
	QScrollArea,
	QVBoxLayout,
	QWidget,
)

from vision_gui.common import StatusBadge

from .config import MotorConfig


class MotorPanel(QWidget):
	"""Motor tab. It never owns or calls a motor backend."""

	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self.config = controller.config
		self._state: dict = {
			"state": "DISCONNECTED",
			"connected": False,
			"enabled": False,
			"homed": False,
		}

		self.connect_button = QPushButton("Connect")
		self.disconnect_button = QPushButton("Disconnect")
		self.enable_button = QPushButton("Enable drive")
		self.disable_button = QPushButton("Disable drive")
		self.clear_faults_button = QPushButton("Clear faults")
		self.connect_button.clicked.connect(controller.connect_motor)
		self.disconnect_button.clicked.connect(controller.disconnect_motor)
		self.enable_button.clicked.connect(controller.enable_drive)
		self.disable_button.clicked.connect(controller.disable_drive)
		self.clear_faults_button.clicked.connect(controller.clear_faults)
		connection_grid = QGridLayout()
		for index, button in enumerate((
			self.connect_button,
			self.disconnect_button,
			self.enable_button,
			self.disable_button,
			self.clear_faults_button,
		)):
			connection_grid.addWidget(button, index // 3, index % 3)
		connection_box = QGroupBox("Connection and drive")
		connection_box.setLayout(connection_grid)

		self.position = self._spin(
			self.config.minimum_position_rad,
			self.config.maximum_position_rad,
			self.config.open_position_rad,
			" rad",
		)
		self.velocity = self._spin(
			0.001,
			self.config.profile_velocity_rad_s,
			self.config.profile_velocity_rad_s,
			" rad/s",
		)
		self.acceleration = self._spin(
			0.001,
			self.config.profile_acceleration_rad_s2,
			self.config.profile_acceleration_rad_s2,
			" rad/s²",
		)
		self.torque = self._spin(
			-self.config.maximum_closing_torque_Nm,
			0.0,
			-0.2,
			" Nm",
		)
		self.torque.setSingleStep(0.05)
		self.torque.setToolTip(
			"Enter a negative closing torque from "
			f"-{self.config.maximum_closing_torque_Nm:g} to 0 Nm. "
			"Editing this field selects Raw motor torque mode."
		)
		force_minimum, force_maximum = (
			(-1_000_000.0, 0.0)
			if self.config.closing_direction < 0 else (0.0, 1_000_000.0)
		)
		self.finger_force = self._spin(
			force_minimum, force_maximum, 0.0, " N"
		)
		self.pulley_radius = self._spin(0.0, 10.0, 0.0, " m", decimals=6)
		self.efficiency = self._spin(0.0, 10.0, 0.0, "", decimals=6)
		self.force_mode = QComboBox()
		self.force_mode.addItem("Raw motor torque", "raw_torque")
		self.force_mode.addItem("Calculated finger force", "finger_force")
		self.calculated_torque = QLabel("—")
		self.force_mode.currentIndexChanged.connect(self._update_force_editor)
		self.torque.editingFinished.connect(self._select_raw_torque_mode)
		self.finger_force.valueChanged.connect(self._update_force_editor)
		self.pulley_radius.valueChanged.connect(self._update_force_editor)
		self.efficiency.valueChanged.connect(self._update_force_editor)

		parameters = QFormLayout()
		parameters.addRow("Target position", self.position)
		parameters.addRow("Profile velocity", self.velocity)
		parameters.addRow("Profile acceleration", self.acceleration)
		parameters.addRow("Force command mode", self.force_mode)
		parameters.addRow("Target motor torque", self.torque)
		parameters.addRow("Target finger force", self.finger_force)
		parameters.addRow("Pulley radius", self.pulley_radius)
		parameters.addRow("Efficiency", self.efficiency)
		parameters.addRow("Calculated motor torque", self.calculated_torque)
		parameter_box = QGroupBox("Command parameters")
		parameter_box.setLayout(parameters)

		self.range_minimum_position = self._spin(
			-1.0,
			0.0,
			self.config.minimum_position_rad,
			" rad",
		)
		self.range_maximum_position = self._spin(
			0.001,
			2.2,
			self.config.maximum_position_rad,
			" rad",
		)
		self.range_open_position = self._spin(
			self.config.minimum_position_rad,
			self.config.maximum_position_rad,
			self.config.open_position_rad,
			" rad",
		)
		self.range_minimum_torque = self._spin(
			-4.0,
			-abs(self.config.homing_torque_Nm),
			-self.config.maximum_closing_torque_Nm,
			" Nm",
		)
		self.range_minimum_torque.setSingleStep(0.05)
		self.range_minimum_torque.setToolTip(
			"Most-negative torque the operator may command. Its magnitude cannot "
			"exceed 4 Nm or be smaller than the configured homing-torque magnitude."
		)
		self.apply_ranges_button = QPushButton("Apply and save ranges")
		self.apply_ranges_button.clicked.connect(self._apply_configured_ranges)
		self.range_minimum_position.valueChanged.connect(self._update_range_editor)
		self.range_maximum_position.valueChanged.connect(self._update_range_editor)
		range_form = QFormLayout()
		range_form.addRow("Minimum position", self.range_minimum_position)
		range_form.addRow("Maximum position", self.range_maximum_position)
		range_form.addRow("Open-button target", self.range_open_position)
		range_form.addRow("Closing limit (most negative)", self.range_minimum_torque)
		range_form.addRow("Maximum closing torque", QLabel("0 Nm (release)"))
		range_help = QLabel(
			"Ranges can only be changed while disconnected. They are saved to "
			"motor-config.yaml and sent to the C++ bridge on the next connection. "
			"The separate controller-register ceiling remains ±10 Nm."
		)
		range_help.setWordWrap(True)
		range_layout = QVBoxLayout()
		range_layout.addLayout(range_form)
		range_layout.addWidget(range_help)
		range_layout.addWidget(self.apply_ranges_button)
		self.range_box = QGroupBox("Configured motor ranges")
		self.range_box.setLayout(range_layout)
		self._configuration_mapping = self.config.to_mapping()

		self.home_button = QPushButton("Home")
		self.open_button = QPushButton("Open gripper")
		self.move_button = QPushButton("Move to target position")
		self.torque_button = QPushButton("Close / hold force")
		self.release_button = QPushButton("Release torque")
		self.stop_button = QPushButton("STOP MOTOR")
		self.stop_button.setMinimumHeight(64)
		self.stop_button.setStyleSheet(
			"QPushButton { background: #b51624; color: white; font-size: 18px; "
			"font-weight: 800; border: 3px solid #ff6875; padding: 10px; } "
			"QPushButton:disabled { background: #67343a; color: #c8c8c8; }"
		)
		self.home_button.clicked.connect(self._home)
		self.open_button.clicked.connect(controller.open_gripper)
		self.move_button.clicked.connect(self._move)
		self.torque_button.clicked.connect(self._apply_force)
		self.release_button.clicked.connect(controller.release_torque)
		# Direct safety path: never route the prominent stop through text parsing.
		self.stop_button.clicked.connect(controller.stop_motor)
		motion_grid = QGridLayout()
		motion_grid.addWidget(self.home_button, 0, 0)
		motion_grid.addWidget(self.open_button, 0, 1)
		motion_grid.addWidget(self.move_button, 1, 0, 1, 2)
		motion_grid.addWidget(self.torque_button, 2, 0)
		motion_grid.addWidget(self.release_button, 2, 1)
		motion_grid.addWidget(self.stop_button, 3, 0, 1, 2)
		motion_box = QGroupBox("Motion")
		motion_box.setLayout(motion_grid)

		self.backend = QLabel(self.config.backend)
		self.connection = StatusBadge("Disconnected")
		self.drive = StatusBadge("Disabled")
		self.homing = StatusBadge("Not homed")
		self.motor_state = StatusBadge("DISCONNECTED")
		self.position_status = QLabel("—")
		self.velocity_status = QLabel("—")
		self.commanded_torque_status = QLabel("0 Nm")
		self.target_torque_status = QLabel("0 Nm")
		self.measured_torque_status = QLabel("—")
		self.torque_error_status = QLabel("—")
		self.torque_confirmation = StatusBadge("Not confirmed")
		self.measured_force_status = QLabel("—")
		self.temperature_status = QLabel("—")
		self.feedback_age_status = QLabel("—")
		self.elapsed_status = QLabel("0 s")
		self.warning_status = QLabel("—")
		self.warning_status.setWordWrap(True)
		self.error_status = QLabel("—")
		self.error_status.setWordWrap(True)
		status = QFormLayout()
		for label, widget in (
			("Backend", self.backend),
			("Connection", self.connection),
			("Drive", self.drive),
			("Homing", self.homing),
			("State / mode", self.motor_state),
			("Position", self.position_status),
			("Velocity", self.velocity_status),
			("Commanded torque", self.commanded_torque_status),
			("Target torque", self.target_torque_status),
			("Measured torque", self.measured_torque_status),
			("Torque error", self.torque_error_status),
			("Torque confirmation", self.torque_confirmation),
			("Calculated finger force", self.measured_force_status),
			("Temperature", self.temperature_status),
			("Feedback age", self.feedback_age_status),
			("Motion elapsed", self.elapsed_status),
			("Latest warning", self.warning_status),
			("Latest error", self.error_status),
		):
			status.addRow(label, widget)
		status_box = QGroupBox("Live motor status")
		status_box.setLayout(status)

		self.debug_toggle = QCheckBox("Show detailed motor debug trace")
		self.debug_toggle.setToolTip(
			"Show every MotorWorker request, C++ bridge command/response, native SDK "
			"message, interpreted feedback sample, state transition, and failure."
		)
		self.debug_trace = QPlainTextEdit()
		self.debug_trace.setReadOnly(True)
		self.debug_trace.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
		self.debug_trace.document().setMaximumBlockCount(10000)
		self.debug_trace.setMinimumHeight(280)
		self.debug_trace.setStyleSheet("font-family: monospace; font-size: 11px;")
		self.debug_trace.setPlaceholderText(
			"Connect or operate the motor to collect a detailed trace."
		)
		self.debug_clear_button = QPushButton("Clear trace")
		self.debug_copy_button = QPushButton("Copy trace")
		self.debug_clear_button.clicked.connect(self.debug_trace.clear)
		self.debug_copy_button.clicked.connect(self._copy_debug_trace)
		debug_buttons = QHBoxLayout()
		debug_buttons.addWidget(self.debug_clear_button)
		debug_buttons.addWidget(self.debug_copy_button)
		debug_buttons.addStretch()
		debug_help = QLabel(
			"TX = command sent to C++; RX = bridge response; NATIVE = CANdle SDK output; "
			"FEEDBACK = values interpreted by MotorWorker."
		)
		debug_help.setWordWrap(True)
		debug_layout = QVBoxLayout()
		debug_layout.addWidget(debug_help)
		debug_layout.addWidget(self.debug_trace)
		debug_layout.addLayout(debug_buttons)
		self.debug_box = QGroupBox("Motor communication debug")
		self.debug_box.setLayout(debug_layout)
		self.debug_box.setVisible(False)
		self.debug_toggle.toggled.connect(self.debug_box.setVisible)

		emergency_warning = QLabel(
			"WARNING: this software STOP cannot guarantee a physical emergency stop. "
			"A correctly installed physical emergency-stop is required for personnel "
			"and equipment safety."
		)
		emergency_warning.setWordWrap(True)
		emergency_warning.setStyleSheet(
			"color: #ff8088; background: #421a1e; border: 2px solid #b51624; "
			"padding: 8px; font-weight: 700;"
		)

		content = QWidget()
		layout = QVBoxLayout(content)
		layout.addWidget(emergency_warning)
		layout.addWidget(connection_box)
		layout.addWidget(self.range_box)
		layout.addWidget(parameter_box)
		layout.addWidget(motion_box)
		layout.addWidget(status_box)
		layout.addWidget(self.debug_toggle)
		layout.addWidget(self.debug_box)
		layout.addStretch()
		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setWidget(content)
		root = QVBoxLayout(self)
		root.addWidget(scroll)

		controller.state_updated.connect(self.update_state)
		if hasattr(controller, "debug_events_emitted"):
			controller.debug_events_emitted.connect(self.append_debug_events)
		self._update_force_editor()
		self._update_buttons()

	@staticmethod
	def _spin(minimum: float, maximum: float, value: float, suffix: str, *, decimals: int = 3) -> QDoubleSpinBox:
		widget = QDoubleSpinBox()
		widget.setDecimals(decimals)
		widget.setRange(float(minimum), float(maximum))
		widget.setValue(float(value))
		widget.setSuffix(suffix)
		return widget

	def _confirm_action(self, title: str, text: str) -> bool:
		return QMessageBox.question(
			self,
			title,
			text,
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
			QMessageBox.StandardButton.No,
		) == QMessageBox.StandardButton.Yes

	def _home(self) -> None:
		if self._confirm_action(
			"Confirm homing",
			f"Home in the confirmed closing direction using {self.config.homing_torque_Nm:g} Nm?",
		):
			self.controller.home()

	def _move(self) -> None:
		self.controller.move_to_position(
			self.position.value(),
			self.velocity.value(),
			self.acceleration.value(),
		)

	def _calculated_motor_torque(self) -> float | None:
		radius = self.pulley_radius.value()
		efficiency = self.efficiency.value()
		if radius <= 0.0 or efficiency <= 0.0:
			return None
		return self.finger_force.value() * radius / efficiency

	def _update_force_editor(self, *_args) -> None:
		calculated = self.force_mode.currentData() == "finger_force"
		for widget in (self.finger_force, self.pulley_radius, self.efficiency):
			widget.setEnabled(calculated)
		torque = self._calculated_motor_torque() if calculated else self.torque.value()
		self.calculated_torque.setText("—" if torque is None else f"{torque:.4g} Nm")
		self._update_measured_force()

	def _select_raw_torque_mode(self) -> None:
		index = self.force_mode.findData("raw_torque")
		if index >= 0 and self.force_mode.currentIndex() != index:
			self.force_mode.setCurrentIndex(index)

	def _copy_debug_trace(self) -> None:
		QApplication.clipboard().setText(self.debug_trace.toPlainText())

	def _update_range_editor(self, *_args) -> None:
		self.range_open_position.setRange(
			self.range_minimum_position.value(),
			self.range_maximum_position.value(),
		)

	def _apply_configured_ranges(self) -> None:
		minimum_position = self.range_minimum_position.value()
		maximum_position = self.range_maximum_position.value()
		open_position = self.range_open_position.value()
		minimum_torque = self.range_minimum_torque.value()
		if not self._confirm_action(
			"Apply motor ranges",
			f"Save position range {minimum_position:g}..{maximum_position:g} rad "
			f"with open target "
			f"{open_position:g} rad, and closing torque range "
			f"{minimum_torque:g}..0 Nm?",
		):
			return
		self.controller.configure_ranges(
			minimum_position,
			maximum_position,
			open_position,
			minimum_torque,
		)

	def _apply_runtime_configuration(self, config: MotorConfig) -> None:
		self.config = config
		with QSignalBlocker(self.range_minimum_position):
			self.range_minimum_position.setValue(config.minimum_position_rad)
		with QSignalBlocker(self.range_maximum_position):
			self.range_maximum_position.setValue(config.maximum_position_rad)
		with QSignalBlocker(self.range_open_position):
			self.range_open_position.setRange(
				config.minimum_position_rad,
				config.maximum_position_rad,
			)
			self.range_open_position.setValue(config.open_position_rad)
		with QSignalBlocker(self.range_minimum_torque):
			self.range_minimum_torque.setRange(
				-4.0,
				-abs(config.homing_torque_Nm),
			)
			self.range_minimum_torque.setValue(-config.maximum_closing_torque_Nm)
		with QSignalBlocker(self.position):
			self.position.setRange(
				config.minimum_position_rad,
				config.maximum_position_rad,
			)
		with QSignalBlocker(self.torque):
			self.torque.setRange(-config.maximum_closing_torque_Nm, 0.0)
		self._configuration_mapping = config.to_mapping()

	def append_debug_events(self, events) -> None:
		for event in events or ():
			try:
				stamp = datetime.fromtimestamp(
					float(event.get("wall_time", 0.0))
				).strftime("%H:%M:%S.%f")[:-3]
			except (TypeError, ValueError, OverflowError, OSError):
				stamp = "--:--:--.---"
			source = str(event.get("source", "motor"))
			category = str(event.get("category", "DEBUG"))
			message = str(event.get("message", "")).replace("\r", "").replace("\n", " ↳ ")
			self.debug_trace.appendPlainText(
				f"{stamp} [{source}] [{category}] {message}"
			)

	def _apply_force(self) -> None:
		updating = self._state.get("state") == "HOLDING_TORQUE"
		if self.force_mode.currentData() == "finger_force":
			torque = self._calculated_motor_torque()
			if torque is None:
				self.controller.log_emitted.emit(
					"error", "Pulley radius and efficiency must be positive."
				)
				return
			description = (
				f"Apply {self.finger_force.value():g} N calculated finger force "
				f"({torque:g} Nm motor torque)?"
			)
		else:
			torque = self.torque.value()
			description = f"Apply {torque:g} Nm closing torque?"
		if torque == 0.0:
			self.controller.release_torque()
			return
		if not self._confirm_action("Confirm nonzero force", description):
			return
		if self.force_mode.currentData() == "finger_force":
			action = (
				self.controller.update_finger_force
				if updating else self.controller.hold_finger_force
			)
			action(
				self.finger_force.value(),
				self.pulley_radius.value(),
				self.efficiency.value(),
			)
		else:
			action = self.controller.update_torque if updating else self.controller.hold_torque
			action(torque)

	def _format(self, value, suffix: str) -> str:
		return "—" if value is None else f"{float(value):.4g}{suffix}"

	def update_state(self, state: dict) -> None:
		self._state = dict(state)
		configuration = state.get("configuration")
		if (
			isinstance(configuration, dict)
			and configuration != self._configuration_mapping
		):
			try:
				self._apply_runtime_configuration(MotorConfig(**configuration))
			except (TypeError, ValueError) as exc:
				self.controller.log_emitted.emit(
					"error", f"Invalid motor configuration received by panel: {exc}"
				)
		connected = bool(state.get("connected"))
		enabled = bool(state.get("enabled"))
		homed = bool(state.get("homed"))
		mode = str(state.get("state", "UNKNOWN"))
		self.backend.setText(str(state.get("backend_type", self.config.backend)))
		self.connection.set_status(
			"Connected" if connected else "Disconnected",
			"good" if connected else "neutral",
		)
		self.drive.set_status(
			"Enabled" if enabled else "Disabled",
			"good" if enabled else "neutral",
		)
		self.homing.set_status(
			"Homed" if homed else "Not homed",
			"good" if homed else "warning" if connected else "neutral",
		)
		kind = "error" if mode == "FAULT" else "warning" if mode in {
			"HOMING", "MOVING", "HOLDING_TORQUE", "STOPPING", "CONNECTING"
		} else "neutral"
		self.motor_state.set_status(mode, kind)
		self.position_status.setText(self._format(state.get("position_rad"), " rad"))
		self.velocity_status.setText(self._format(state.get("velocity_rad_s"), " rad/s"))
		self.commanded_torque_status.setText(
			self._format(state.get("commanded_torque_Nm", 0.0), " Nm")
		)
		self.target_torque_status.setText(
			self._format(state.get("target_torque_Nm", 0.0), " Nm")
		)
		self.measured_torque_status.setText(
			self._format(state.get("measured_torque_Nm"), " Nm")
		)
		measured_torque = state.get("measured_torque_Nm")
		target_torque = state.get("target_torque_Nm")
		if measured_torque is None or target_torque is None:
			self.torque_error_status.setText("—")
		else:
			self.torque_error_status.setText(
				f"{float(measured_torque) - float(target_torque):.4g} Nm"
			)
		confirmed = bool(state.get("torque_confirmed"))
		command_at_target = bool(state.get("torque_command_at_target"))
		within_tolerance = bool(state.get("torque_within_tolerance"))
		if confirmed:
			self.torque_confirmation.set_status("Confirmed", "good")
		elif mode == "HOLDING_TORQUE":
			if not command_at_target:
				label = "Ramping"
			elif within_tolerance:
				label = "Within tolerance; confirming"
			else:
				label = "Waiting for measured torque"
			self.torque_confirmation.set_status(label, "warning")
		else:
			self.torque_confirmation.set_status("Not confirmed", "neutral")
		self.temperature_status.setText(self._format(state.get("temperature_C"), " °C"))
		feedback_time = state.get("feedback_monotonic")
		if feedback_time is None:
			self.feedback_age_status.setText("—")
		else:
			self.feedback_age_status.setText(
				f"{max(0.0, time.monotonic() - float(feedback_time)):.3f} s"
			)
		self.elapsed_status.setText(self._format(state.get("motion_elapsed_s", 0.0), " s"))
		self.warning_status.setText(str(state.get("latest_warning") or "—"))
		self.error_status.setText(str(state.get("latest_error") or "—"))
		self.torque_button.setText(
			"Update held force" if mode == "HOLDING_TORQUE" else "Close / hold force"
		)
		self._update_measured_force()
		self._update_buttons()

	def _update_measured_force(self) -> None:
		measured = self._state.get("measured_torque_Nm")
		radius = self.pulley_radius.value()
		efficiency = self.efficiency.value()
		if measured is None or radius <= 0.0 or efficiency <= 0.0:
			self.measured_force_status.setText("—")
		else:
			force = float(measured) / radius * efficiency
			self.measured_force_status.setText(f"{force:.4g} N")

	def _update_buttons(self) -> None:
		connected = bool(self._state.get("connected"))
		enabled = bool(self._state.get("enabled"))
		homed = bool(self._state.get("homed"))
		state = str(self._state.get("state", "DISCONNECTED"))
		idle = state == "IDLE"
		active = state in {"HOMING", "MOVING", "HOLDING_TORQUE", "STOPPING"}
		self.connect_button.setEnabled(not connected and state != "CONNECTING")
		self.disconnect_button.setEnabled(connected and not active)
		self.enable_button.setEnabled(connected and idle and not enabled)
		self.disable_button.setEnabled(connected and (enabled or active))
		self.clear_faults_button.setEnabled(connected and not enabled and state in {"IDLE", "FAULT"})
		self.home_button.setEnabled(connected and enabled and idle)
		self.open_button.setEnabled(connected and enabled and idle and (homed or not self.config.require_homing))
		self.move_button.setEnabled(connected and enabled and idle and (homed or not self.config.require_homing))
		self.torque_button.setEnabled(
			connected and enabled and (
				(state == "HOLDING_TORQUE")
				or (idle and (homed or not self.config.require_homing))
			)
		)
		self.release_button.setEnabled(connected and state == "HOLDING_TORQUE")
		self.stop_button.setEnabled(connected)
		self.range_box.setEnabled(not connected and state != "CONNECTING")
