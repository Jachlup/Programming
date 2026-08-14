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
	experiment_batch_queued = Signal(str, bool, str)
	model_activation_completed = Signal(str, bool, str)
	shutdown_complete = Signal()

	def __init__(self, state: ApplicationState | None = None) -> None:
		super().__init__()
		self.state = state or ApplicationState()
		self._latest_motor_telemetry: dict | None = None
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
				warning_callback=lambda message: self._log("warning", message),
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
		self.state.current_blue_detections = []
		self.state.current_circles = []
		if self.state.tracker is not None:
			self.state.tracker.previous_gray = None
		if self.state.blue_tracker is not None:
			self.state.blue_tracker.previous_gray = None
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
				display_frame = self.state.update_frame(
					raw_frame,
					motor_telemetry=self._latest_motor_telemetry,
				)
			self.frame_ready.emit(display_frame.copy())
			self._emit_state()
		except Exception as exc:
			self._paused = True
			self._emit_error("Frame processing failed; processing was paused", exc)
			self.camera_state_changed.emit(True, True)

	@Slot(object)
	def set_motor_telemetry(self, snapshot) -> None:
		"""Keep a detached scalar snapshot outside ``ApplicationState`` ownership."""
		if not isinstance(snapshot, dict):
			self._latest_motor_telemetry = None
			return
		keys = (
			"position_rad",
			"velocity_rad_s",
			"target_torque_Nm",
			"measured_torque_Nm",
			"temperature_C",
			"state",
			"feedback_monotonic",
		)
		self._latest_motor_telemetry = {
			key: copy.deepcopy(snapshot.get(key)) for key in keys
		}

	@Slot(object)
	def queue_experiment_batch(self, specification) -> None:
		step_id = ""
		try:
			if not isinstance(specification, dict):
				raise ValueError("Experiment batch specification must be a mapping")
			step_id = str(specification.get("force_step_id", ""))
			self.state.queue_dataset_batch(**copy.deepcopy(specification))
			message = f"Queued experiment batch {step_id}."
			self._log("success", message)
			self.experiment_batch_queued.emit(step_id, True, message)
		except Exception as exc:
			message = str(exc)
			self._emit_error("Experiment batch could not be queued", exc)
			self.experiment_batch_queued.emit(step_id, False, message)
		self._emit_state()

	@Slot()
	def abort_experiment_batch(self) -> None:
		batch = self.state.abort_pending_dataset_batch()
		if batch is not None:
			self._log("warning", f"Aborted pending batch {batch.force_step_id}.")
		self._emit_state()

	@Slot(str, str)
	def activate_force_model(self, model_path: str, metadata_path: str) -> None:
		try:
			model = self.state.activate_force_model(
				model_path, metadata_path or None
			)
			message = (
				f"Activated force model {model.metadata.model_name} "
				f"version {model.metadata.model_version}."
			)
			self._log("success", message)
			self.model_activation_completed.emit(model_path, True, message)
		except Exception as exc:
			self._emit_error("Force model activation failed; prior model retained", exc)
			self.model_activation_completed.emit(model_path, False, str(exc))
		self._emit_state()

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
			if command_name == "calibration_start":
				self.configuration_applied.emit(config_to_mapping(self.state.config))
		self.command_completed.emit(line, success)

	@Slot(int, int)
	def handle_frame_click(self, x: int, y: int) -> None:
		calibration_was_active = (
			self.state.calibration_mode or self.state.blue_calibration_mode
		)
		try:
			with self._captured_output():
				handled = self.state.handle_mouse_click(
					cv2.EVENT_LBUTTONDOWN, int(x), int(y)
				)
			if not handled and not self.state.last_click_rejection_reason:
				self._log("warning", f"Frame click at ({x}, {y}) was not accepted.")
			elif calibration_was_active:
				self.configuration_applied.emit(config_to_mapping(self.state.config))
		except Exception as exc:
			self._emit_error("Frame click failed", exc)
		self._emit_state()

	@Slot(int, object)
	def set_blob_area_limits(self, minimum: int, maximum) -> None:
		"""Apply live blob-area limits from the GUI calibration sliders."""
		try:
			minimum_value = int(minimum)
			maximum_value = None if maximum is None else int(maximum)
			if minimum_value < 0:
				raise ValueError("Minimum blob area cannot be negative")
			if maximum_value is not None and maximum_value < minimum_value:
				raise ValueError("Maximum blob area must be at least the minimum")
			self.state.config.calibration.min_area = minimum_value
			self.state.config.calibration.max_area = maximum_value
			if self.state.config is not cfg:
				cfg.calibration.min_area = minimum_value
				cfg.calibration.max_area = maximum_value
			self.configuration_applied.emit(config_to_mapping(self.state.config))
			self._emit_state()
		except Exception as exc:
			self._emit_error("Blob-area update failed", exc)

	@Slot(int, object)
	def set_blue_blob_area_limits(self, minimum: int, maximum) -> None:
		"""Apply live blue-target area limits from its dedicated GUI tab."""
		try:
			minimum_value = int(minimum)
			maximum_value = None if maximum is None else int(maximum)
			if minimum_value < 0:
				raise ValueError("Minimum blue blob area cannot be negative")
			if maximum_value is not None and maximum_value < minimum_value:
				raise ValueError("Maximum blue blob area must be at least the minimum")
			self.state.config.blue_calibration.min_area = minimum_value
			self.state.config.blue_calibration.max_area = maximum_value
			if self.state.config is not cfg:
				cfg.blue_calibration.min_area = minimum_value
				cfg.blue_calibration.max_area = maximum_value
			self.configuration_applied.emit(config_to_mapping(self.state.config))
			self._emit_state()
		except Exception as exc:
			self._emit_error("Blue blob-area update failed", exc)

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
			if self.state.blue_tracker is not None:
				self.state.blue_tracker.config = self.state.config.blue_tracking
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
	experiment_batch_queued = Signal(str, bool, str)
	model_activation_completed = Signal(str, bool, str)

	_start_camera = Signal()
	_stop_camera = Signal()
	_pause_camera = Signal(bool)
	_run_command = Signal(str)
	_frame_click = Signal(int, int)
	_blob_area_limits = Signal(int, object)
	_blue_blob_area_limits = Signal(int, object)
	_motor_telemetry = Signal(object)
	_apply_configuration = Signal(object)
	_reload_configuration = Signal()
	_queue_experiment_batch = Signal(object)
	_abort_experiment_batch = Signal()
	_activate_force_model = Signal(str, str)
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
		self._blob_area_limits.connect(self._worker.set_blob_area_limits)
		self._blue_blob_area_limits.connect(
			self._worker.set_blue_blob_area_limits
		)
		self._motor_telemetry.connect(self._worker.set_motor_telemetry)
		self._apply_configuration.connect(self._worker.apply_configuration)
		self._reload_configuration.connect(self._worker.reload_configuration)
		self._queue_experiment_batch.connect(self._worker.queue_experiment_batch)
		self._abort_experiment_batch.connect(self._worker.abort_experiment_batch)
		self._activate_force_model.connect(self._worker.activate_force_model)
		self._initial_state.connect(self._worker.emit_initial_state)
		self._shutdown.connect(self._worker.shutdown)

		self._worker.frame_ready.connect(self.frame_ready)
		self._worker.state_updated.connect(self.state_updated)
		self._worker.camera_state_changed.connect(self.camera_state_changed)
		self._worker.log_emitted.connect(self.log_emitted)
		self._worker.command_completed.connect(self.command_completed)
		self._worker.configuration_applied.connect(self.configuration_applied)
		self._worker.configuration_saved.connect(self.configuration_saved)
		self._worker.experiment_batch_queued.connect(self.experiment_batch_queued)
		self._worker.model_activation_completed.connect(
			self.model_activation_completed
		)
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

	def set_blob_area_limits(self, minimum: int, maximum: int | None) -> None:
		self._blob_area_limits.emit(int(minimum), maximum)

	def set_blue_blob_area_limits(
		self,
		minimum: int,
		maximum: int | None,
	) -> None:
		self._blue_blob_area_limits.emit(int(minimum), maximum)

	@Slot(object)
	def update_motor_telemetry(self, snapshot) -> None:
		self._motor_telemetry.emit(copy.deepcopy(snapshot))

	def apply_config(self, mapping: dict) -> None:
		self._apply_configuration.emit(copy.deepcopy(mapping))

	def reload_config(self) -> None:
		self._reload_configuration.emit()

	def queue_experiment_batch(self, specification: dict) -> None:
		self._queue_experiment_batch.emit(copy.deepcopy(specification))

	def abort_experiment_batch(self) -> None:
		self._abort_experiment_batch.emit()

	def activate_force_model(
		self,
		model_path: str,
		metadata_path: str | None = None,
	) -> None:
		self._activate_force_model.emit(str(model_path), str(metadata_path or ""))

	def shutdown(self, timeout_ms: int = 5000) -> bool:
		if not self._thread.isRunning():
			return True
		self._shutdown.emit()
		if self._thread.wait(timeout_ms):
			return True
		self.log_emitted.emit("error", "Camera worker did not stop within the timeout.")
		return False
