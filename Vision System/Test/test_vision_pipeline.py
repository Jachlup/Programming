"""Hardware-free tests for reference identity and geometric feature ordering."""

from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import extract_geometry
from tracking import PointTracker, initialise_reference, make_reference_profile


def _configuration():
	with (Path(__file__).resolve().parents[1] / "camera-config.yaml").open() as stream:
		return yaml.safe_load(stream)


def _detections():
	result = [
		{"center": (float(x), 100.0 + float(x) * 0.01), "detection_quality": 1.0}
		for x in np.linspace(50, 450, 9)
	]
	result.extend(
		{"center": (float(x), 200.0 + float(x) * 0.01), "detection_quality": 1.0}
		for x in np.linspace(50, 450, 11)
	)
	result.extend([
		{"center": (20.0, 20.0), "detection_quality": 1.0},
		{"center": (20.0, 400.0), "detection_quality": 1.0},
	])
	return result


def test_unequal_line_counts_and_permanent_ids():
	config = _configuration()
	result = initialise_reference(
		_detections(), {**config["reference_ransac"], **config["reference"]},
		(640, 480), [20, 21],
	)
	assert result.valid
	assert [len(line.point_ids) for line in result.lines] == [9, 11]
	assert result.lines[0].point_ids == [f"LINE_A_{i:02d}" for i in range(9)]
	assert result.lines[1].point_ids == [f"LINE_B_{i:02d}" for i in range(11)]


def test_tracking_and_feature_order_are_valid():
	config = _configuration()
	result = initialise_reference(
		_detections(), {**config["reference_ransac"], **config["reference"]},
		(640, 480), [20, 21],
	)
	profile = make_reference_profile(
		result, (640, 480), config["reference_ransac"], config["geometry"]
	)
	tracker = PointTracker(profile, config["tracking"])
	points = tracker.update(np.zeros((480, 640), np.uint8), _detections())
	geometry = extract_geometry(profile, points, config["geometry"])
	assert tracker.valid
	assert geometry.valid
	assert geometry.feature_names[:4] == [
		"LINE_A_00.dx_px", "LINE_A_00.dy_px",
		"LINE_A_01.dx_px", "LINE_A_01.dy_px",
	]
	assert len(geometry.feature_names) == len(geometry.feature_vector)
