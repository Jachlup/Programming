"""Camera setup and frame acquisition helpers for the RealSense stream."""

from __future__ import annotations

import pyrealsense2 as rs


def _apply_sensor_options(profile: rs.pipeline_profile, options: dict | None) -> None:
	"""Apply configured color-sensor controls when the device supports them."""
	if not options:
		return
	color_sensor = profile.get_device().first_color_sensor()
	auto_exposure = bool(options.get("auto_exposure", True))
	if color_sensor.supports(rs.option.enable_auto_exposure):
		color_sensor.set_option(rs.option.enable_auto_exposure, float(auto_exposure))
	if auto_exposure:
		return
	for key, option in (("exposure", rs.option.exposure), ("gain", rs.option.gain)):
		value = options.get(key)
		if value is not None and color_sensor.supports(option):
			color_sensor.set_option(option, float(value))


def create_pipeline(
	width: int = 640,
	height: int = 480,
	fps: int = 60,
	camera_options: dict | None = None,
) -> rs.pipeline:
	"""Create and start the RealSense pipeline for the color stream."""
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
	profile = pipeline.start(config)
	try:
		_apply_sensor_options(profile, camera_options)
	except Exception:
		pipeline.stop()
		raise
	return pipeline


def get_frame(pipeline: rs.pipeline):
	"""Return the latest color frame as a NumPy image, or None if unavailable."""
	frames = pipeline.wait_for_frames()
	color_frame = frames.get_color_frame()
	if not color_frame:
		return None
	return color_frame


def poll_frame(pipeline: rs.pipeline):
	"""Return the newest color frame without blocking, or ``None``."""
	frames = pipeline.poll_for_frames()
	if not frames:
		return None
	color_frame = frames.get_color_frame()
	return color_frame or None


def stop_pipeline(pipeline: rs.pipeline) -> None:
	"""Stop the RealSense pipeline safely."""
	if pipeline is not None:
		pipeline.stop()
