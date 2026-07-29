"""Minimal standalone RealSense color-camera check."""

import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

try:
	pipeline.start(config)
	for _ in range(10):
		frames = pipeline.wait_for_frames(10_000)
		color = frames.get_color_frame()
		if color:
			print(
				f"Camera works: {color.get_width()}x{color.get_height()}, "
				f"frame {color.get_frame_number()}"
			)
			break
	else:
		print("Camera opened, but no color frame was received.")
finally:
	pipeline.stop()
