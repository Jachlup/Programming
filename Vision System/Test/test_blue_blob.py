"""Hardware-free coverage for blue calibration, tracking, and ORIGIN offsets."""

from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blue_tracking
from application import ApplicationState
from blue_tracking import BlueBlobTracker, BlueTrackStatus
from commands import execute_command
from processing import (
	cfg,
	config_from_mapping,
	config_to_mapping,
	find_blue_blob_data,
)


def _config():
	return config_from_mapping(config_to_mapping(cfg))


def _blue_frame(center=(80, 60), radius=12):
	frame = np.zeros((120, 180, 3), dtype=np.uint8)
	cv2.circle(frame, center, radius, (255, 0, 0), thickness=-1)
	return frame


def _detection(center=(80.0, 60.0), quality=1.0):
	return {"center": center, "detection_quality": quality}


def test_old_configuration_without_blue_sections_gets_safe_defaults() -> None:
	mapping = config_to_mapping(cfg)
	mapping.pop("blue_calibration")
	mapping.pop("blue_tracking")
	config = config_from_mapping(mapping)
	assert config.blue_calibration.blue_lower_1.tolist() == [90, 40, 30]
	assert config.blue_tracking["maximum_assignment_distance_px"] == 40.0


def test_blue_detector_rejects_red_and_returns_blue_metadata() -> None:
	config = _config()
	config.blue_calibration.min_area = 10
	config.blue_calibration.max_area = 1_000
	frame = _blue_frame()
	cv2.circle(frame, (140, 60), 12, (0, 0, 255), thickness=-1)
	blobs = find_blue_blob_data(frame, config=config)
	assert len(blobs) == 1
	assert np.allclose(blobs[0]["center"], (80.0, 60.0), atol=0.1)
	assert blobs[0]["area"] > 300
	assert blobs[0]["detection_quality"] > 0.5


def test_blue_calibration_click_binds_the_single_target() -> None:
	state = ApplicationState(config=_config())
	frame = _blue_frame()
	state.update_frame(frame, detections=[])
	state.start_blue_calibration()
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 80, 60)
	assert state.selected_blue_calibration_blob is not None
	assert state.blue_tracker is not None
	assert state.blue_tracker.selected
	assert state.blue_tracker.track.status == BlueTrackStatus.DETECTED
	assert state.blue_tracker.track.current_position == (80.0, 60.0)
	assert state.config.blue_calibration.blue_lower_1[0] <= 120
	assert state.config.blue_calibration.blue_upper_1[0] >= 120
	state.stop_blue_calibration()
	assert state.blue_tracker.selected


def test_selected_blue_identity_follows_nearest_consistent_detection() -> None:
	state = ApplicationState(config=_config())
	frame = _blue_frame(center=(45, 60))
	cv2.circle(frame, (140, 60), 12, (255, 0, 0), thickness=-1)
	state.update_frame(frame, detections=[])
	state.start_blue_calibration()
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 45, 60)

	moved = _blue_frame(center=(55, 65))
	cv2.circle(moved, (140, 60), 12, (255, 0, 0), thickness=-1)
	state.update_frame(moved, detections=[])
	assert state.blue_tracker is not None
	assert np.allclose(
		state.blue_tracker.track.current_position, (55.0, 65.0), atol=0.1
	)


def test_blue_tracker_uses_optical_flow_during_short_detection_gap(monkeypatch) -> None:
	tracker = BlueBlobTracker({
		**cfg.blue_tracking,
		"minimum_tracking_quality": 0.1,
		"maximum_missing_frames": 2,
	})
	gray = np.zeros((80, 100), dtype=np.uint8)
	tracker.bind(gray, _detection((30.0, 40.0)))

	def fake_flow(*_args, **_kwargs):
		return (
			np.array([[[34.0, 43.0]]], dtype=np.float32),
			np.array([[1]], dtype=np.uint8),
			np.array([[0.0]], dtype=np.float32),
		)

	monkeypatch.setattr(blue_tracking.cv2, "calcOpticalFlowPyrLK", fake_flow)
	track = tracker.update(gray, [])
	assert track.status == BlueTrackStatus.TRACKED_ONLY
	assert track.current_position == (34.0, 43.0)
	assert tracker.valid


def test_invalid_track_explains_filters_flow_and_missing_limit(monkeypatch) -> None:
	tracker = BlueBlobTracker({
		**cfg.blue_tracking,
		"maximum_missing_frames": 0,
	})
	gray = np.zeros((80, 100), dtype=np.uint8)
	tracker.bind(gray, _detection((30.0, 40.0)))
	monkeypatch.setattr(
		blue_tracking.cv2,
		"calcOpticalFlowPyrLK",
		lambda *_args, **_kwargs: (None, None, None),
	)
	track = tracker.update(gray, [])
	assert track.status == BlueTrackStatus.INVALID
	assert not tracker.valid
	assert "No blue blob passed the active HSV and area filters" in tracker.invalid_reason
	assert "optical flow could not produce a valid point" in tracker.invalid_reason
	assert "maximum missing-frame limit 0 was exceeded" in tracker.invalid_reason


def test_invalid_track_explains_assignment_distance_and_low_quality(monkeypatch) -> None:
	tracker = BlueBlobTracker({
		**cfg.blue_tracking,
		"maximum_assignment_distance_px": 5.0,
		"maximum_missing_frames": 0,
		"minimum_tracking_quality": 0.2,
	})
	gray = np.zeros((80, 100), dtype=np.uint8)
	tracker.bind(gray, _detection((10.0, 10.0)))
	monkeypatch.setattr(
		blue_tracking.cv2,
		"calcOpticalFlowPyrLK",
		lambda *_args, **_kwargs: (None, None, None),
	)
	tracker.update(gray, [_detection((30.0, 10.0))])
	assert "outside the 5.00px assignment limit" in tracker.invalid_reason

	tracker.bind(gray, _detection((10.0, 10.0)))
	tracker.update(gray, [_detection((10.0, 10.0), quality=0.05)])
	assert not tracker.valid
	assert "quality 0.050 is below the configured minimum 0.200" in tracker.invalid_reason


class _OriginTracker:
	origin_point_id = "REFERENCE_ORIGIN"
	origin_reacquisition_required = False
	valid = True
	quality = 1.0

	def __init__(self, position):
		self.origin = SimpleNamespace(current_position=position)

	def get_point(self, point_id):
		return self.origin if point_id == self.origin_point_id else None


def test_blue_position_is_reported_relative_to_valid_origin(capsys) -> None:
	state = ApplicationState(config=_config())
	gray = np.zeros((80, 100), dtype=np.uint8)
	assert state.blue_tracker is not None
	state.blue_tracker.bind(gray, _detection((75.0, 42.0)))
	state.tracker = _OriginTracker((20.0, 12.0))
	state.tracking_enabled = True
	state.current_geometry = SimpleNamespace(
		origin_valid=True,
		valid=True,
		quality=1.0,
	)
	snapshot = state.status_snapshot()
	assert snapshot["blue_origin_relative_position"] == (55.0, 30.0)
	assert snapshot["blue_relative_position_valid"]
	assert execute_command("blue_blob_status", state)
	assert "dx=+55.00px, dy=+30.00px" in capsys.readouterr().out

	state.current_geometry.origin_valid = False
	snapshot = state.status_snapshot()
	assert snapshot["blue_origin_relative_position"] is None
	assert snapshot["blue_relative_unavailable_reason"] == "Reference origin is invalid"


def test_clear_blue_target_removes_the_only_bound_track() -> None:
	state = ApplicationState(config=_config())
	assert state.blue_tracker is not None
	state.blue_tracker.bind(
		np.zeros((80, 100), dtype=np.uint8), _detection()
	)
	state.clear_blue_target()
	assert not state.blue_tracker.selected
	assert state.blue_tracker.track.status == BlueTrackStatus.UNSELECTED
	assert state.blue_origin_relative_position is None
