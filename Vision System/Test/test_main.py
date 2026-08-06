"""Portable configuration smoke tests (no camera hardware required)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing import load_config
from main import CommandConsole


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


def test_continuous_command_console_queues_input_without_camera_blocking() -> None:
	responses = iter(["tracking_status"])
	prompts: list[str] = []

	def fake_input(prompt: str) -> str:
		prompts.append(prompt)
		try:
			return next(responses)
		except StopIteration as exc:
			raise EOFError from exc

	console = CommandConsole(input_function=fake_input)
	console.start()
	item = console.get_pending(timeout=1.0)
	assert item is not None
	command_line, completed = item
	assert command_line == "tracking_status"
	assert prompts == ["Command> "]
	completed.set()
	console.stop()
