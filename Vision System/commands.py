"""Validated text-command surface for the application coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Callable

from dataset import DatasetSession
from force_model import ForceModel
from processing import cfg, save_config


@dataclass(frozen=True)
class Command:
	name: str
	description: str
	handler: Callable


def _no_args(args, name):
	if args:
		raise ValueError(f"Usage: {name}")


def reference_start(args, state):
	indices = []
	if args:
		indices = [int(value) for value in args[0].split(",") if value]
	state.reference_start(indices)
	print("Reference setup started. Click detections to toggle rigid markers, then use reference_preview.")


def reference_preview(args, state):
	_no_args(args, "reference_preview")
	state.reference_preview()
	print(f"Reference valid={state.ransac_result.valid}, quality={state.ransac_result.quality:.3f}")
	for warning in state.ransac_result.warnings:
		print(f"  warning: {warning}")


def reference_accept(args, state):
	_no_args(args, "reference_accept")
	state.reference_accept()
	print(f"Accepted {state.reference_profile.profile_id}")


def reference_reject(args, state):
	_no_args(args, "reference_reject")
	state.ransac_result = None
	print("Reference proposal rejected.")


def reference_clear(args, state):
	_no_args(args, "reference_clear")
	state.clear_reference()


def reference_save(args, state):
	state.reference_save(args[0] if args else None)
	print("Reference profile saved.")


def reference_load(args, state):
	state.reference_load(args[0] if args else None)
	print(f"Loaded {state.reference_profile.profile_id}")


def reference_status(args, state):
	_no_args(args, "reference_status")
	print(state.reference_profile or state.ransac_result or "No reference")


def tracking_start(args, state):
	_no_args(args, "tracking_start")
	if state.reference_profile is None:
		raise RuntimeError("Load or accept a reference first")
	if state.tracker_state is None:
		from tracking import PointTracker
		state.tracker_state = PointTracker(state.reference_profile, cfg.tracking)
	print("Tracking active.")


def tracking_stop(args, state):
	_no_args(args, "tracking_stop")
	state.tracker_state = None


def tracking_status(args, state):
	_no_args(args, "tracking_status")
	tracker = state.tracker_state
	print("Tracking stopped" if tracker is None else f"valid={tracker.valid}, quality={tracker.quality:.3f}")


def _show(option):
	def handler(args, state):
		if len(args) > 1 or (args and args[0] not in ("on", "off")):
			raise ValueError(f"Usage: show_{option} [on|off]")
		attr = f"show_{option}"
		value = not getattr(state.display_options, attr) if not args else args[0] == "on"
		setattr(state.display_options, attr, value)
		print(f"{attr}={value}")
	return handler


def dataset_start(args, state):
	if len(args) != 2:
		raise ValueError("Usage: dataset_start EXPERIMENT_ID KNOWN_FORCE_N")
	if state.reference_profile is None or state.tracker_state is None:
		raise RuntimeError("Valid reference and tracking are required")
	state.known_reference_force = float(args[1])
	state.dataset_session = DatasetSession.start(
		cfg.dataset.get("directory", "datasets"), args[0], state.reference_profile,
		str(cfg.geometry.get("geometry_configuration_version", "1")),
		bool(cfg.dataset.get("save_images", True)),
	)
	state.mode = type(state.mode).DATA_COLLECTION


def dataset_sample(args, state):
	if len(args) > 1:
		raise ValueError("Usage: dataset_sample [FRAME_COUNT]")
	if state.dataset_session is None:
		raise RuntimeError("No dataset session")
	count = int(args[0]) if args else int(cfg.dataset.get("frames_per_sample", 1))
	if count < 1:
		raise ValueError("FRAME_COUNT must be positive")
	state.pending_dataset_frames = count
	print(f"Queued {count} frame(s); invalid geometries will be rejected.")


def dataset_stop(args, state):
	_no_args(args, "dataset_stop")
	if state.dataset_session:
		print(f"Accepted={state.dataset_session.accepted}, rejected={state.dataset_session.rejected}")
	state.dataset_session = None
	state.mode = type(state.mode).DIAGNOSTIC


def dataset_status(args, state):
	_no_args(args, "dataset_status")
	print(state.dataset_session or "No dataset session")


def model_load(args, state):
	if len(args) not in (1, 2):
		raise ValueError("Usage: model_load MODEL.pkl [metadata.json]")
	state.loaded_model = ForceModel.load(args[0], args[1] if len(args) == 2 else None)
	print(f"Loaded model {state.loaded_model.metadata.model_name}")


def model_info(args, state):
	_no_args(args, "model_info")
	print(state.loaded_model.metadata if state.loaded_model else "No model loaded")


def force_start(args, state):
	_no_args(args, "force_start")
	if not all((state.reference_profile, state.tracker_state, state.loaded_model, state.current_geometry)):
		raise RuntimeError("Reference, tracking, valid geometry, and model are required")
	if not state.tracker_state.valid or not state.current_geometry.valid:
		raise RuntimeError("Tracking and rigid-reference geometry must be valid")
	state.loaded_model.validate_features(state.feature_names)
	state.mode = type(state.mode).FORCE_MEASUREMENT


def force_stop(args, state):
	_no_args(args, "force_stop")
	state.mode = type(state.mode).DIAGNOSTIC
	state.predicted_force = None


def force_status(args, state):
	_no_args(args, "force_status")
	print(state.predicted_force or f"Mode={state.mode.value}")


def save_configuration(args, state):
	_no_args(args, "save_config")
	save_config(cfg)
	print("Configuration saved.")


def help_command(args, state):
	_no_args(args, "help")
	for command in COMMANDS.values():
		print(f"{command.name:<20} {command.description}")


COMMANDS = {
	"reference_start": Command("reference_start", "Start setup; optional rigid indices: 0,1", reference_start),
	"reference_preview": Command("reference_preview", "Fit and validate the two reference lines", reference_preview),
	"reference_accept": Command("reference_accept", "Accept proposed permanent IDs", reference_accept),
	"reference_reject": Command("reference_reject", "Reject the current proposal", reference_reject),
	"reference_clear": Command("reference_clear", "Clear accepted/proposed reference", reference_clear),
	"reference_save": Command("reference_save", "Save profile [path]", reference_save),
	"reference_load": Command("reference_load", "Load profile [path]", reference_load),
	"reference_status": Command("reference_status", "Show reference status", reference_status),
	"tracking_start": Command("tracking_start", "Start tracking", tracking_start),
	"tracking_stop": Command("tracking_stop", "Stop tracking", tracking_stop),
	"tracking_status": Command("tracking_status", "Show tracker quality", tracking_status),
	"show_points": Command("show_points", "Toggle point overlays [on|off]", _show("points")),
	"show_lines": Command("show_lines", "Toggle reference lines [on|off]", _show("lines")),
	"show_outliers": Command("show_outliers", "Toggle rejected outliers [on|off]", _show("outliers")),
	"show_geometry": Command("show_geometry", "Toggle structural geometry [on|off]", _show("geometry")),
	"show_features": Command("show_features", "Toggle feature display [on|off]", _show("features")),
	"dataset_start": Command("dataset_start", "Start EXPERIMENT_ID KNOWN_FORCE_N", dataset_start),
	"dataset_sample": Command("dataset_sample", "Save a sample batch [FRAME_COUNT]", dataset_sample),
	"dataset_stop": Command("dataset_stop", "Stop and report dataset session", dataset_stop),
	"dataset_status": Command("dataset_status", "Show dataset status", dataset_status),
	"model_load": Command("model_load", "Load MODEL.pkl [metadata.json]", model_load),
	"model_info": Command("model_info", "Show model metadata", model_info),
	"force_start": Command("force_start", "Begin validated inference", force_start),
	"force_stop": Command("force_stop", "Stop inference", force_stop),
	"force_status": Command("force_status", "Show latest prediction", force_status),
	"save_config": Command("save_config", "Save YAML configuration", save_configuration),
	"help": Command("help", "Show commands", help_command),
}


def execute_command(command_line, state) -> bool:
	try:
		tokens = shlex.split(command_line)
		if not tokens:
			return False
		command = COMMANDS.get(tokens[0])
		if command is None:
			raise ValueError(f"Unknown command: {tokens[0]}")
		command.handler(tokens[1:], state)
		return True
	except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
		print(f"Command failed: {exc}")
		return False
