"""Top-level window for the vision system."""

from __future__ import annotations

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDockWidget, QMainWindow, QMessageBox, QTabWidget

from experiment_control import ExperimentController
from motor_control.panel import MotorPanel
from training_control import TrainingController
from motor_control.runtime import MotorController

from .camera_panel import CameraPanel
from .collection_panel import CollectionPanel
from .command_console import CommandConsole
from .dataset_panel import DatasetPanel
from .runtime import ApplicationController
from .training_panel import TrainingPanel
from .tuning_panel import TuningPanel


class MainWindow(QMainWindow):
	def __init__(
		self,
		controller: ApplicationController | None = None,
		motor_controller: MotorController | None = None,
		experiment_controller: ExperimentController | None = None,
		training_controller: TrainingController | None = None,
		*,
		auto_start_camera: bool = True,
		parent=None,
	) -> None:
		super().__init__(parent)
		self.controller = controller or ApplicationController(parent=self)
		self.motor_controller = motor_controller or MotorController(parent=self)
		self.experiment_controller = experiment_controller or ExperimentController(
			self.controller, self.motor_controller, parent=self
		)
		self.training_controller = training_controller or TrainingController(
			self.controller, parent=self
		)
		self.setWindowTitle("Fin Ray Vision System")
		self.setMinimumSize(1080, 720)

		self.tabs = QTabWidget()
		# Keep one live image widget for both clicking/calibration and display.
		# It lives in a persistent dock so changing tabs never hides the feed.
		self.camera_panel = CameraPanel(self.controller, embed_video=False)
		self.reference_panel = self.camera_panel.reference_panel
		self.tuning_panel = TuningPanel(self.controller)
		self.dataset_panel = DatasetPanel(self.controller)
		self.motor_panel = MotorPanel(self.motor_controller)
		self.collection_panel = CollectionPanel(self.experiment_controller)
		self.training_panel = TrainingPanel(self.training_controller)
		self.tabs.addTab(self.camera_panel, "Camera")
		self.tabs.addTab(self.tuning_panel, "Tuning")
		self.tabs.addTab(self.dataset_panel, "Dataset")
		self.tabs.addTab(self.motor_panel, "Motor")
		self.tabs.addTab(self.collection_panel, "Collection")
		self.tabs.addTab(self.training_panel, "Training")
		self.camera_panel.open_tuning_requested.connect(
			lambda: self.tabs.setCurrentWidget(self.tuning_panel)
		)
		self.setCentralWidget(self.tabs)

		self.camera_preview_dock = QDockWidget("Live camera preview", self)
		self.camera_preview_dock.setObjectName("camera-preview-dock")
		self.camera_preview_dock.setAllowedAreas(
			Qt.DockWidgetArea.LeftDockWidgetArea
			| Qt.DockWidgetArea.RightDockWidgetArea
		)
		self.camera_preview_dock.setFeatures(
			QDockWidget.DockWidgetFeature.DockWidgetMovable
			| QDockWidget.DockWidgetFeature.DockWidgetFloatable
		)
		self.camera_preview_dock.setWidget(self.camera_panel.video)
		self.addDockWidget(
			Qt.DockWidgetArea.RightDockWidgetArea,
			self.camera_preview_dock,
		)
		self.resizeDocks(
			[self.camera_preview_dock],
			[520],
			Qt.Orientation.Horizontal,
		)

		self.console = CommandConsole(self.controller)
		console_dock = QDockWidget("Command console and logs", self)
		console_dock.setObjectName("command-console-dock")
		console_dock.setAllowedAreas(
			Qt.DockWidgetArea.BottomDockWidgetArea
			| Qt.DockWidgetArea.TopDockWidgetArea
		)
		console_dock.setWidget(self.console)
		self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, console_dock)
		self.resizeDocks([console_dock], [180], Qt.Orientation.Vertical)

		self.controller.log_emitted.connect(self._show_log_status)
		self.motor_controller.log_emitted.connect(self.console.append_log)
		self.motor_controller.log_emitted.connect(self._show_log_status)
		self.experiment_controller.log_emitted.connect(self.console.append_log)
		self.experiment_controller.log_emitted.connect(self._show_log_status)
		self.training_controller.log_emitted.connect(self.console.append_log)
		self.training_controller.log_emitted.connect(self._show_log_status)
		self.motor_controller.state_updated.connect(
			self.controller.update_motor_telemetry
		)
		self._settings = QSettings("SoftRoboticsLab", "FinRayVision")
		geometry = self._settings.value("windowGeometry")
		if geometry is not None:
			self.restoreGeometry(geometry)
		window_state = self._settings.value("windowState")
		if window_state is not None:
			self.restoreState(window_state)

		if auto_start_camera:
			QTimer.singleShot(0, self.controller.start_camera)

	def _show_log_status(self, level: str, message: str) -> None:
		if level in {"success", "warning", "error"}:
			self.statusBar().showMessage(message, 7000)

	def closeEvent(self, event: QCloseEvent) -> None:
		self._settings.setValue("windowGeometry", self.saveGeometry())
		self._settings.setValue("windowState", self.saveState())
		self.experiment_controller.shutdown()
		training_ok = self.training_controller.shutdown()
		motor_ok = self.motor_controller.shutdown()
		camera_ok = self.controller.shutdown()
		if not training_ok or not motor_ok or not camera_ok:
			failed = []
			if not training_ok:
				failed.append("training")
			if not motor_ok:
				failed.append("motor")
			if not camera_ok:
				failed.append("camera")
			QMessageBox.critical(
				self,
				"Worker shutdown failed",
				"The following worker(s) did not stop cleanly: "
				+ ", ".join(failed)
				+ ". The window will remain open.",
			)
			event.ignore()
			return
		event.accept()
