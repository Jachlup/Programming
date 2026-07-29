"""Camera setup and frame acquisition helpers for the RealSense stream."""

from __future__ import annotations

import pyrealsense2 as rs


def create_pipeline(width: int = 640, height: int = 480, fps: int = 60) -> rs.pipeline:
	"""Create and start the RealSense pipeline for the color stream."""
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
	pipeline.start(config)
	return pipeline


def get_frame(pipeline: rs.pipeline):
	"""Return the latest color frame as a NumPy image, or None if unavailable."""
	frames = pipeline.wait_for_frames()
	color_frame = frames.get_color_frame()
	if not color_frame:
		return None
	return color_frame


def stop_pipeline(pipeline: rs.pipeline) -> None:
	"""Stop the RealSense pipeline safely."""
	pipeline.stop()
