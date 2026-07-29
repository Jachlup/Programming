"""Portable configuration smoke tests (no camera hardware required)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing import load_config


def test_camera_configuration_loads() -> None:
	config = load_config()
	assert config.camera["width"] > 0
	assert config.camera["height"] > 0
	assert config.reference_ransac["expected_points_line_a"] > 0
	assert config.reference_ransac["expected_points_line_b"] > 0
	assert config.geometry["geometry_reference_mode"] in {
		"translation_only",
		"translation_and_rotation",
	}
