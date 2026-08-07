"""PySide6 entry point for the Fin Ray camera vision system."""

from __future__ import annotations

import argparse
import os
import sys

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--no-camera",
		action="store_true",
		help="Open the GUI without automatically starting the RealSense camera.",
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	args = build_argument_parser().parse_args(argv)
	# Importing cv2 sets this variable to OpenCV's private Qt plugin directory.
	# PySide6 must initialise its own Qt runtime first or the incompatible OpenCV
	# xcb plugin is selected and QApplication aborts during startup.
	plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
	if "/cv2/qt/plugins" in plugin_path.replace("\\", "/"):
		os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
	QCoreApplication.setOrganizationName("SoftRoboticsLab")
	QCoreApplication.setApplicationName("FinRayVision")
	app = QApplication(sys.argv[:1])
	app.setStyle("Fusion")
	# Import the vision modules only after QApplication has loaded PySide6's
	# platform plugin; the runtime legitimately imports OpenCV for processing.
	from vision_gui import MainWindow

	window = MainWindow(auto_start_camera=not args.no_camera)
	window.show()
	return app.exec()


if __name__ == "__main__":
	raise SystemExit(main())
