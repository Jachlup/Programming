"""Threaded camera runtime and GUI-facing application controller."""

from __future__ import annotations

import contextlib
import copy
import io
import shlex
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from application import ApplicationState, DisplayOptions
from camera import create_pipeline, poll_frame, stop_pipeline
from commands import execute_command
from processing import (
	cfg,
	config_from_mapping,
	config_to_mapping,
	load_config,
	update_config_in_place,
)


class _SignalWriter(io.TextIOBase):
	"""Turn legacy print output into structured Qt log messages."""

	def __init__(self, callback, level: str) -> None:
		super().__init__()
		self._callback = callback
		self._level = level
		self._buffer = ""

	def write(self, value: str) -> int:
		self._buffer += str(value)
		while "\n" in self._buffer:
			line, self._buffer = self._buffer.split("\n", 1)
			if line.strip():
				self._callback(self._level, line.rstrip())
		return len(value)

	def flush(self) -> None:
		if self._buffer.strip():
			self._callback(self._level, self._buffer.rstrip())
		self._buffer = ""


class CameraWorker(QObject):
	"""Own the camera and the only mutable ``ApplicationState`` instance."""

	frame_ready = Signal(object)
	state_updated = Signal(object)
	camera_state_changed = Signal(bool, bool)
	log_emitted = Signal(str, str)
	command_completed = Signal(str, bool)
	configuration_applied = Signal(object)
	configuration_saved = Signal(object)
	shutdown_complete = Signal()

	def __init__(self, state: ApplicationState | None = None) -> None:
		super().__init__()
		self.state = state or ApplicationState()
		self._pipeline = None
		self._paused = False
		self._timer = QTimer(self)
		self._timer.setInterval(5)
		self._timer.timeout.connect(self._process_once)

	def _log(self, level: str, message: str) -> None:
		self.log_emitted.emit(level, str(message))

	@contextlib.contextmanager
	def _captured_output(self):
		stdout = _SignalWriter(self._log, "info")
		stderr = _SignalWriter(self._log, "warning")
		with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
			yield
		stdout.flush()
		stderr.flush()

	def _emit_state(self) -> None:
		self.state_updated.emit(self.state.status_snapshot())

	def _emit_error(self, context: str, exc: BaseException) -> None:
		self._log("error", f"{context}: {exc}")
		self._log("debug", "".join(traceback.format_exception(exc)).rstrip())

	@Slot()
	def emit_initial_state(self) -> None:
		self._emit_state()
		self.configuration_applied.emit(config_to_mapping(self.state.config))
		try:
			self.configuration_saved.emit(config_to_mapping(load_config()))
		except Exception as exc:
			self._emit_error("Could not read saved configuration", exc)
		self.camera_state_changed.emit(self._pipeline is not None, self._paused)

	@Slot()
	def start_camera(self) -> None:
		if self._pipeline is not None:
			self._log("warning", "Camera is already running.")
			return
		camera = self.state.config.camera
		try:
			self._pipeline = create_pipeline(
				width=int(camera.get("width", 640)),
				height=int(camera.get("height", 480)),
				fps=int(camera.get("fps", 60)),
				camera_options=camera,
			)
			self._paused = False
			self._timer.start()
			self._log("success", "Camera started.")
		except Exception as exc:
			self._pipeline = None
			self._emit_error("Camera start failed", exc)
		self.camera_state_changed.emit(self._pipeline is not None, self._paused)

	@Slot()
	def stop_camera(self) -> None:
		self._timer.stop()
		pipeline, self._pipeline = self._pipeline, None
		if pipeline is not None:
			try:
				stop_pipeline(pipeline)
				self._log("success", "Camera stopped and released.")
			except Exception as exc:
				self._emit_error("Camera stop failed", exc)
		self._paused = False
		self.state.current_raw_frame = None
		self.state.current_gray_frame = None
		self.state.current_detections = []
		self.state.current_circles = []
		if self.state.tracker is not None:
			self.state.tracker.previous_gray = None
		self._emit_state()
		self.camera_state_changed.emit(False, False)

	@Slot(bool)
	def set_paused(self, paused: bool) -> None:
		if self._pipeline is None:
			self._log("warning", "Start the camera before pausing processing.")
			return
		self._paused = bool(paused)
		self._log("info", "Processing paused." if self._paused else "Processing resumed.")
		self.camera_state_changed.emit(True, self._paused)

	@Slot()
	def _process_once(self) -> None:
		if self._pipeline is None:
			return
		try:
			color_frame = poll_frame(self._pipeline)
			if color_frame is None or self._paused:
				return
			raw_frame = np.asanyarray(color_frame.get_data()).copy()
			with self._captured_output():
				display_frame = self.state.update_frame(raw_frame)
			self.frame_ready.emit(display_frame.copy())
			self._emit_state()
		except Exception as exc:
			self._paused = True
			self._emit_error("Frame processing failed; processing was paused", exc)
			self.camera_state_changed.emit(True, True)

	@Slot(str)
	def run_command(self, command_line: str) -> None:
		line = command_line.strip()
		if not line:
			self.command_completed.emit(command_line, False)
			return
		self._log("command", f"> {line}")
		try:
			with self._captured_output():
				success = execute_command(line, self.state)
		except Exception as exc:
			success = False
			self._emit_error("Command raised an unexpected exception", exc)
		self._emit_state()
		if success:
			try:
				command_name = shlex.split(line)[0]
			except ValueError:
				command_name = ""
			if command_name == "save_config":
				self.configuration_saved.emit(config_to_mapping(self.state.config))
		self.command_completed.emit(line, success)

	@Slot(int, int)
	def handle_frame_click(self, x: int, y: int) -> None:
		try:
			with self._captured_output():
				handled = self.state.handle_mouse_click(
					cv2.EVENT_LBUTTONDOWN, int(x), int(y)
				)
			if not handled:
				self._log("warning", f"Frame click at ({x}, {y}) was not accepted.")
		except Exception as exc:
			self._emit_error("Frame click failed", exc)
		self._emit_state()

	@staticmethod
	def _profile_contract(mapping: dict) -> tuple:
		ransac = mapping["reference_ransac"]
		geometry = mapping["geometry"]
		tracking = mapping["tracking"]
		return (
			mapping["camera"].get("width"),
			mapping["camera"].get("height"),
			ransac.get("expected_points_line_a"),
			ransac.get("expected_points_line_b"),
			ransac.get("configuration_version"),
			tracking.get("expected_deforming_point_count"),
			tracking.get("expected_reference_origin_point_count"),
			tracking.get("expected_rigid_reference_point_count"),
			geometry.get("geometry_reference_mode"),
			geometry.get("structural_connection_version"),
			tuple(mapping["reference"].get("rigid_reference_point_ids", [])),
		)

	@Slot(object)
	def apply_configuration(self, mapping: dict) -> None:
		try:
			candidate = config_from_mapping(copy.deepcopy(mapping))
			current_mapping = config_to_mapping(self.state.config)
			candidate_mapping = config_to_mapping(candidate)
			if (
				self.state.reference_profile is not None
				and self._profile_contract(current_mapping)
				!= self._profile_contract(candidate_mapping)
			):
				raise RuntimeError(
					"These structural settings cannot change while a reference profile "
					"is loaded. Clear the reference first."
				)
			camera_changed = current_mapping["camera"] != candidate_mapping["camera"]
			was_running = self._pipeline is not None
			update_config_in_place(self.state.config, candidate)
			if self.state.config is not cfg:
				update_config_in_place(cfg, candidate)
			if self.state.tracker is not None:
				self.state.tracker.config = self.state._tracking_config()
			self.state.display_options = DisplayOptions.from_mapping(
				self.state.config.display
			)
			self.configuration_applied.emit(config_to_mapping(self.state.config))
			self._emit_state()
			self._log("success", "Edited configuration applied.")
			if camera_changed and was_running:
				self._log("info", "Camera settings changed; restarting the camera.")
				self.stop_camera()
				self.start_camera()
		except Exception as exc:
			self._emit_error("Configuration apply failed", exc)

	@Slot()
	def reload_configuration(self) -> None:
		try:
			loaded = load_config()
		except Exception as exc:
			self._emit_error("Configuration reload failed", exc)
			return
		self.apply_configuration(config_to_mapping(loaded))
		self.configuration_saved.emit(config_to_mapping(loaded))

	@Slot()
	def shutdown(self) -> None:
		self.stop_camera()
		self.shutdown_complete.emit()
		QThread.currentThread().quit()


class ApplicationController(QObject):
	"""GUI-thread facade that queues every state mutation to ``CameraWorker``."""

	frame_ready = Signal(object)
	state_updated = Signal(object)
	camera_state_changed = Signal(bool, bool)
	log_emitted = Signal(str, str)
	command_completed = Signal(str, bool)
	configuration_applied = Signal(object)
	configuration_saved = Signal(object)

	_start_camera = Signal()
	_stop_camera = Signal()
	_pause_camera = Signal(bool)
	_run_command = Signal(str)
	_frame_click = Signal(int, int)
	_apply_configuration = Signal(object)
	_reload_configuration = Signal()
	_initial_state = Signal()
	_shutdown = Signal()

	def __init__(self, state: ApplicationState | None = None, parent=None) -> None:
		super().__init__(parent)
		self._thread = QThread(self)
		self._thread.setObjectName("vision-camera-worker")
		self._worker = CameraWorker(state)
		self._worker.moveToThread(self._thread)

		self._start_camera.connect(self._worker.start_camera)
		self._stop_camera.connect(self._worker.stop_camera)
		self._pause_camera.connect(self._worker.set_paused)
		self._run_command.connect(self._worker.run_command)
		self._frame_click.connect(self._worker.handle_frame_click)
		self._apply_configuration.connect(self._worker.apply_configuration)
		self._reload_configuration.connect(self._worker.reload_configuration)
		self._initial_state.connect(self._worker.emit_initial_state)
		self._shutdown.connect(self._worker.shutdown)

		self._worker.frame_ready.connect(self.frame_ready)
		self._worker.state_updated.connect(self.state_updated)
		self._worker.camera_state_changed.connect(self.camera_state_changed)
		self._worker.log_emitted.connect(self.log_emitted)
		self._worker.command_completed.connect(self.command_completed)
		self._worker.configuration_applied.connect(self.configuration_applied)
		self._worker.configuration_saved.connect(self.configuration_saved)
		self._thread.started.connect(self._initial_state)
		self._thread.start()

	def start_camera(self) -> None:
		self._start_camera.emit()

	def stop_camera(self) -> None:
		self._stop_camera.emit()

	def set_paused(self, paused: bool) -> None:
		self._pause_camera.emit(paused)

	def execute(self, command_line: str) -> None:
		self._run_command.emit(command_line)

	def click_frame(self, x: int, y: int) -> None:
		self._frame_click.emit(x, y)

	def apply_config(self, mapping: dict) -> None:
		self._apply_configuration.emit(copy.deepcopy(mapping))

	def reload_config(self) -> None:
		self._reload_configuration.emit()

	def shutdown(self, timeout_ms: int = 5000) -> bool:
		if not self._thread.isRunning():
			return True
		self._shutdown.emit()
		if self._thread.wait(timeout_ms):
			return True
		self.log_emitted.emit("error", "Camera worker did not stop within the timeout.")
		return False
