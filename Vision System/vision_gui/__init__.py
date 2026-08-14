"""PySide6 interface for the camera-based Fin Ray vision system."""

__all__ = ["ApplicationController", "CameraWorker", "MainWindow"]


def __getattr__(name):
	"""Load top-level GUI classes lazily so shared widgets remain reusable."""
	if name == "MainWindow":
		from .main_window import MainWindow

		return MainWindow
	if name in {"ApplicationController", "CameraWorker"}:
		from .runtime import ApplicationController, CameraWorker

		return {
			"ApplicationController": ApplicationController,
			"CameraWorker": CameraWorker,
		}[name]
	raise AttributeError(name)
