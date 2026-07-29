"""Validated text-command surface for the application coordinator."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
import shlex
from typing import Callable

import yaml

from processing import cfg, save_config


@dataclass(frozen=True)
class Command:
	name: str
	description: str
	handler: Callable[[list[str], object], None]


def _no_args(args: list[str], name: str) -> None:
	if args:
		raise ValueError(f"Usage: {name}")


def _optional_path(args: list[str], name: str) -> str | None:
	if len(args) > 1:
		raise ValueError(f"Usage: {name} [PATH]")
	return args[0] if args else None


def _parse_rigid_indices(args: list[str]) -> tuple[int, ...]:
	if len(args) > 1:
		raise ValueError("Usage: reference_start [RIGID_INDICES]")
	if not args:
		return ()
	try:
		indices = tuple(int(value.strip()) for value in args[0].split(",") if value.strip())
	except ValueError as exc:
		raise ValueError("Rigid detection indices must be comma-separated integers") from exc
	if any(index < 0 for index in indices):
		raise ValueError("Rigid detection indices must be non-negative")
	if len(set(indices)) != len(indices):
		raise ValueError("Rigid detection indices must be unique")
	return indices


def _positive_count(value: str) -> int:
	try:
		count = int(value)
	except ValueError as exc:
		raise ValueError("FRAME_COUNT must be a positive integer") from exc
	if count < 1:
		raise ValueError("FRAME_COUNT must be positive")
	return count


def reference_start(args, state):
	indices = _parse_rigid_indices(args)
	state.reference_start(indices=indices)
	print(
		"Reference setup started. Run reference_origin_select, click the origin blob, "
		"then run reference_preview. Optional rigid indices use the displayed D0, D1, ... labels."
	)


def reference_origin_select(args, state):
	_no_args(args, "reference_origin_select")
	state.begin_reference_origin_selection()
	print("Click the detected blob that will become the moving reference origin.")


def reference_origin_clear(args, state):
	_no_args(args, "reference_origin_clear")
	state.clear_reference_origin()
	print("Reference-origin candidate cleared.")


def reference_origin_status(args, state):
	_no_args(args, "reference_origin_status")
	profile = state.reference_profile
	origin_id = getattr(profile, "reference_origin_point_id", None) if profile is not None else None
	setup_active = getattr(getattr(state, "mode", None), "value", None) == "REFERENCE_SETUP"
	proposal = getattr(state, "ransac_result", None)
	candidate_index = getattr(state, "selected_origin_detection_index", None)
	selection_pending = bool(
		getattr(
			state,
			"reference_origin_selection_mode",
			getattr(state, "origin_selection_mode", False),
		)
	)
	if (
		setup_active
		and proposal is not None
		and proposal.reference_origin_point is not None
	):
		print(f"Proposed reference origin: {proposal.reference_origin_point.id}")
	elif candidate_index is not None:
		print(f"Reference-origin candidate detection index: {candidate_index}")
	elif selection_pending:
		print("Waiting for a reference-origin click.")
	elif setup_active:
		print("No reference origin selected for the active setup.")
	elif origin_id is not None:
		print(f"Accepted reference origin: {origin_id}")
	else:
		print("No reference origin selected.")


def reference_preview(args, state):
	_no_args(args, "reference_preview")
	result = state.reference_preview()
	if result is None:
		result = state.ransac_result
	if result is None:
		raise RuntimeError("Reference preview did not produce a proposal")
	print(f"Reference valid={result.valid}, quality={result.quality:.3f}")
	for error in getattr(result, "fatal_errors", ()):
		print(f"  error: {error}")
	for warning in result.warnings:
		print(f"  warning: {warning}")


def reference_accept(args, state):
	_no_args(args, "reference_accept")
	profile = state.reference_accept()
	if profile is None:
		profile = state.reference_profile
	if profile is None:
		raise RuntimeError("Reference acceptance did not produce a profile")
	print(f"Accepted {profile.profile_id}")


def reference_reject(args, state):
	_no_args(args, "reference_reject")
	state.reject_reference()
	print("Reference proposal rejected.")


def reference_clear(args, state):
	_no_args(args, "reference_clear")
	state.clear_reference()
	print("Reference setup and accepted profile cleared.")


def reference_save(args, state):
	path = _optional_path(args, "reference_save")
	state.reference_save(path)
	print("Reference profile saved.")


def reference_load(args, state):
	path = _optional_path(args, "reference_load")
	profile = state.reference_load(path)
	if profile is None:
		profile = state.reference_profile
	if profile is None:
		raise RuntimeError("Reference loading did not produce a profile")
	print(f"Loaded {profile.profile_id}")


def reference_status(args, state):
	_no_args(args, "reference_status")
	setup_active = getattr(getattr(state, "mode", None), "value", None) == "REFERENCE_SETUP"
	if setup_active and state.ransac_result is not None:
		result = state.ransac_result
		print(f"Proposed reference valid={result.valid}, quality={result.quality:.3f}")
		for error in getattr(result, "fatal_errors", ()):
			print(f"  error: {error}")
		for warning in result.warnings:
			print(f"  warning: {warning}")
	elif setup_active:
		candidate = getattr(state, "selected_origin_detection_index", None)
		print(
			"Reference setup active; "
			f"origin candidate={candidate if candidate is not None else 'not selected'}"
		)
	elif state.reference_profile is not None:
		profile = state.reference_profile
		origin_id = getattr(profile, "reference_origin_point_id", None)
		print(f"Accepted profile={profile.profile_id}, origin={origin_id or 'not configured'}")
	elif state.ransac_result is not None:
		result = state.ransac_result
		print(f"Proposed reference valid={result.valid}, quality={result.quality:.3f}")
		for error in getattr(result, "fatal_errors", ()):
			print(f"  error: {error}")
		for warning in result.warnings:
			print(f"  warning: {warning}")
	else:
		print("No reference")


def tracking_start(args, state):
	_no_args(args, "tracking_start")
	state.start_tracking()
	print("Tracking active.")


def tracking_stop(args, state):
	_no_args(args, "tracking_stop")
	state.stop_tracking()
	print("Tracking stopped.")


def tracking_status(args, state):
	_no_args(args, "tracking_status")
	tracker = state.tracker_state
	if tracker is None:
		print("Tracking stopped")
	else:
		print(f"Tracking valid={tracker.valid}, quality={tracker.quality:.3f}")


def origin_reacquire(args, state):
	_no_args(args, "origin_reacquire")
	state.begin_origin_reacquisition()
	print(
		"Click the restored physical origin marker. No detection can inherit "
		"REFERENCE_ORIGIN until this confirmation."
	)


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
	known_force = float(args[1])
	state.start_dataset(args[0], known_force)
	print(f"Dataset session started: experiment={args[0]}, known_force_N={known_force:g}")


def dataset_force(args, state):
	if len(args) != 1:
		raise ValueError("Usage: dataset_force KNOWN_FORCE_N")
	known_force = float(args[0])
	state.set_dataset_force(known_force)
	print(f"Known dataset force set to {known_force:g} N.")


def dataset_sample(args, state):
	if len(args) > 1:
		raise ValueError("Usage: dataset_sample [FRAME_COUNT]")
	default_count = int(cfg.dataset.get("frames_per_sample", 1))
	count = _positive_count(args[0]) if args else _positive_count(str(default_count))
	state.queue_dataset_samples(count)
	print(f"Queued {count} frame(s); invalid frames will remain pending for retry.")


def dataset_stop(args, state):
	_no_args(args, "dataset_stop")
	session = state.dataset_session
	state.stop_dataset()
	if session is None:
		print("No dataset session was active.")
	else:
		print(f"Dataset stopped: accepted={session.accepted}, rejected={session.rejected}")


def dataset_status(args, state):
	_no_args(args, "dataset_status")
	session = state.dataset_session
	if session is None:
		print("No dataset session")
		return
	print(
		f"Dataset experiment={session.experiment_id}, force_N={state.known_reference_force:g}, "
		f"accepted={session.accepted}, rejected={session.rejected}, "
		f"pending={state.pending_dataset_frames}"
	)


def model_load(args, state):
	if len(args) not in (1, 2):
		raise ValueError("Usage: model_load MODEL.pkl [metadata.json]")
	model = state.load_force_model(args[0], args[1] if len(args) == 2 else None)
	if model is None:
		model = state.loaded_model
	if model is None:
		raise RuntimeError("Model loading did not produce a force model")
	print(f"Loaded model {model.metadata.model_name}")


def model_info(args, state):
	_no_args(args, "model_info")
	print(state.loaded_model.metadata if state.loaded_model else "No model loaded")


def force_start(args, state):
	_no_args(args, "force_start")
	state.start_force_inference()
	print("Force inference active.")


def force_stop(args, state):
	_no_args(args, "force_stop")
	state.stop_force_inference()
	print("Force inference stopped.")


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
		print(f"{command.name:<24} {command.description}")


COMMANDS = {
	"reference_start": Command(
		"reference_start", "Start setup [comma-separated D-label rigid indices]", reference_start
	),
	"reference_origin_select": Command(
		"reference_origin_select", "Make the next valid click select the moving origin", reference_origin_select
	),
	"reference_origin_clear": Command(
		"reference_origin_clear", "Clear the proposed origin marker", reference_origin_clear
	),
	"reference_origin_status": Command(
		"reference_origin_status", "Show origin-selection status", reference_origin_status
	),
	"reference_preview": Command(
		"reference_preview", "Fit and validate the two deforming lines", reference_preview
	),
	"reference_accept": Command(
		"reference_accept", "Accept the origin and proposed permanent IDs", reference_accept
	),
	"reference_reject": Command("reference_reject", "Reject the current proposal", reference_reject),
	"reference_clear": Command("reference_clear", "Clear accepted/proposed reference", reference_clear),
	"reference_save": Command("reference_save", "Save profile [path]", reference_save),
	"reference_load": Command("reference_load", "Load profile [path]", reference_load),
	"reference_status": Command("reference_status", "Show reference status", reference_status),
	"tracking_start": Command("tracking_start", "Start tracking", tracking_start),
	"tracking_stop": Command("tracking_stop", "Stop tracking", tracking_stop),
	"tracking_status": Command("tracking_status", "Show tracker quality", tracking_status),
	"origin_reacquire": Command(
		"origin_reacquire",
		"Arm a click to confirm a lost origin's permanent identity",
		origin_reacquire,
	),
	"show_points": Command("show_points", "Toggle point overlays [on|off]", _show("points")),
	"show_lines": Command("show_lines", "Toggle reference lines [on|off]", _show("lines")),
	"show_outliers": Command("show_outliers", "Toggle rejected outliers [on|off]", _show("outliers")),
	"show_geometry": Command("show_geometry", "Toggle structural geometry [on|off]", _show("geometry")),
	"show_features": Command("show_features", "Toggle feature display [on|off]", _show("features")),
	"show_circles": Command("show_circles", "Toggle Hough-circle overlays [on|off]", _show("circles")),
	"show_status": Command("show_status", "Toggle runtime status overlays [on|off]", _show("status")),
	"show_warnings": Command("show_warnings", "Toggle warning overlays [on|off]", _show("warnings")),
	"show_mask_only": Command("show_mask_only", "Toggle mask-only display [on|off]", _show("mask_only")),
	"dataset_start": Command("dataset_start", "Start EXPERIMENT_ID KNOWN_FORCE_N", dataset_start),
	"dataset_force": Command("dataset_force", "Set KNOWN_FORCE_N for later samples", dataset_force),
	"dataset_sample": Command("dataset_sample", "Queue a sample batch [FRAME_COUNT]", dataset_sample),
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


EXPECTED_COMMAND_ERRORS = (
	ValueError,
	RuntimeError,
	OSError,
	KeyError,
	json.JSONDecodeError,
	yaml.YAMLError,
	pickle.UnpicklingError,
	EOFError,
)


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
	except EXPECTED_COMMAND_ERRORS as exc:
		print(f"Command failed: {exc}")
		return False
