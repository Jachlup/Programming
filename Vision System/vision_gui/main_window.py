"""Top-level four-tab window for the vision system."""

from __future__ import annotations

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDockWidget, QMainWindow, QMessageBox, QTabWidget

from .camera_panel import CameraPanel
from .command_console import CommandConsole
from .dataset_panel import DatasetPanel
from .reference_panel import ReferencePanel
from .runtime import ApplicationController
from .tuning_panel import TuningPanel


class MainWindow(QMainWindow):
	def __init__(
		self,
		controller: ApplicationController | None = None,
		*,
		auto_start_camera: bool = True,
		parent=None,
	) -> None:
		super().__init__(parent)
		self.controller = controller or ApplicationController(parent=self)
		self.setWindowTitle("Fin Ray Vision System")
		self.setMinimumSize(1080, 720)

		self.tabs = QTabWidget()
		self.camera_panel = CameraPanel(self.controller)
		self.reference_panel = ReferencePanel(self.controller)
		self.tuning_panel = TuningPanel(self.controller)
		self.dataset_panel = DatasetPanel(self.controller)
		self.tabs.addTab(self.camera_panel, "Camera")
		self.tabs.addTab(self.reference_panel, "Reference setup")
		self.tabs.addTab(self.tuning_panel, "Tuning")
		self.tabs.addTab(self.dataset_panel, "Dataset")
		self.setCentralWidget(self.tabs)

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
		if not self.controller.shutdown():
			QMessageBox.critical(
				self,
				"Camera worker did not stop",
				"The camera worker did not stop cleanly. The window will remain open.",
			)
			event.ignore()
			return
		event.accept()
