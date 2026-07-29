"""Hardware-free coverage for the moving-reference-origin force pipeline."""

from __future__ import annotations

import copy
import csv
import inspect
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application import ApplicationState
from commands import execute_command
from force_model import ForceModel, ModelMetadata
from geometry import extract_geometry
import main
from processing import cfg
from tracking import (
	PointGroup,
	PointStatus,
	PointTracker,
	initialise_reference,
	load_reference_profile,
	make_reference_profile,
	save_reference_profile,
)
from train_force_model import train


class ZeroEstimator:
	def predict(self, features):
		return np.zeros(len(features), dtype=float)


class ConstantEstimator:
	def __init__(self, value: float):
		self.value = float(value)

	def predict(self, features):
		return np.full(len(features), self.value, dtype=float)


def _detection(x: float, y: float, *, label: str = "") -> dict:
	x_i, y_i = int(round(x)), int(round(y))
	contour = np.asarray(
		[[[x_i - 4, y_i - 4]], [[x_i + 4, y_i - 4]],
		 [[x_i + 4, y_i + 4]], [[x_i - 4, y_i + 4]]],
		dtype=np.int32,
	)
	return {
		"center": (float(x), float(y)),
		"contour": contour,
		"area": 64.0,
		"detection_quality": 1.0,
		"label": label,
	}


def _detections(*, rigid: bool = False) -> list[dict]:
	result = [
		_detection(x, 100.0, label=f"A{index}")
		for index, x in enumerate((30.0, 60.0, 90.0))
	]
	result.extend(
		_detection(x, 160.0, label=f"B{index}")
		for index, x in enumerate((30.0, 60.0, 90.0))
	)
	result.append(_detection(140.0, 70.0, label="origin"))
	if rigid:
		result.extend([
			_detection(20.0, 20.0, label="rigid-left"),
			_detection(110.0, 20.0, label="rigid-right"),
		])
	return result


def _reference_config(mode: str = "translation_only") -> dict:
	return {
		"expected_line_count": 2,
		"expected_points_line_a": 3,
		"expected_points_line_b": 3,
		"expected_deforming_point_count": 6,
		"distance_threshold_px": 2.0,
		"iterations": 100,
		"minimum_inliers_per_line": 2,
		"direction_rule": "left_to_right",
		"maximum_angle_difference_deg": 5.0,
		"minimum_line_separation_px": 20.0,
		"maximum_line_separation_px": 100.0,
		"maximum_outlier_count": 0,
		"minimum_order_spacing_px": 5.0,
		"maximum_mean_residual_px": 1.0,
		"minimum_valid_rigid_reference_points": 2,
		"geometry_reference_mode": mode,
		"configuration_version": "test",
	}


def _geometry_config(mode: str = "translation_only") -> dict:
	return {
		"geometry_reference_mode": mode,
		"minimum_valid_rigid_reference_points": 2,
		"minimum_origin_tracking_quality": 0.1,
		"allow_tracked_only_origin": False,
		"minimum_deforming_tracking_quality": 0.0,
		"enabled_feature_groups": [
			"point_displacements",
			"length_changes",
			"relative_length_changes",
			"angle_changes",
		],
		"structural_connections": [
			["LINE_A_00", "LINE_A_01"],
			["LINE_B_00", "LINE_B_01"],
		],
		"structural_connection_version": "test",
		"geometry_configuration_version": "test",
		"feature_schema_version": "test",
	}


def _profile(*, rigid: bool = False, mode: str = "translation_only"):
	detections = _detections(rigid=rigid)
	origin_index = next(
		index for index, detection in enumerate(detections)
		if detection["label"] == "origin"
	)
	rigid_indices = [
		index for index, detection in enumerate(detections)
		if detection["label"].startswith("rigid-")
	]
	result = initialise_reference(
		detections,
		_reference_config(mode),
		(200, 200),
		rigid_indices,
		reference_origin_detection_index=origin_index,
	)
	assert result.valid, result.fatal_errors
	return make_reference_profile(
		result, (200, 200), _reference_config(mode), _geometry_config(mode)
	)


def _tracker_config() -> dict:
	return {
		"maximum_assignment_distance": 18.0,
		"origin_identity_lock_distance_px": 2.0,
		"origin_maximum_assignment_distance": 5.0,
		"origin_reacquisition_max_distance_px": 5.0,
		"origin_reacquisition_policy": "manual_confirmation",
		"origin_initialization_max_distance_px": 2.0,
		"origin_no_flow_max_distance_px": 0.25,
		"origin_maximum_missing_frames": 2,
		"maximum_missing_frames": 2,
		"maximum_optical_flow_error": 0.01,
		"maximum_adjacent_distance_ratio": 3.0,
		"minimum_origin_tracking_quality": 0.1,
		"allow_tracked_only_origin": False,
		"allow_tracked_only_points": False,
	}


def _application_config(tmp_path: Path | None = None):
	config = copy.deepcopy(cfg)
	config.reference_ransac = _reference_config()
	config.reference.update({
		"origin_selection_max_distance_px": 12.0,
		"minimum_origin_tracking_quality": 0.1,
		"allow_tracked_only_origin": False,
		"origin_maximum_missing_frames": 2,
		"origin_reacquisition_max_distance_px": 5.0,
	})
	config.tracking.update(_tracker_config())
	config.geometry = _geometry_config()
	config.dataset.update({
		"directory": str(tmp_path or "datasets"),
		"save_images": False,
		"retry_invalid_frames": True,
		"maximum_invalid_frame_retries": 2,
	})
	config.display["show_circles"] = False
	return config


def _accepted_application(tmp_path: Path | None = None) -> tuple[ApplicationState, np.ndarray]:
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	state = ApplicationState(config=_application_config(tmp_path))
	state.update_frame(frame, detections=_detections(), timestamp=1.0)
	assert execute_command("reference_start", state)
	assert execute_command("reference_origin_select", state)
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 140, 70)
	assert execute_command("reference_preview", state)
	assert execute_command("reference_accept", state)
	state.update_frame(frame, detections=_detections(), timestamp=2.0)
	assert state.tracker is not None and state.tracker.valid
	assert state.current_geometry is not None and state.current_geometry.valid
	return state, frame


def test_origin_click_is_contour_first_and_distant_click_is_rejected() -> None:
	state = ApplicationState(config=_application_config())
	state.current_detections = _detections()
	state.reference_start()
	state.begin_reference_origin_selection()
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 143, 72)
	assert state.selected_origin_detection_index == 6

	state.clear_reference_origin()
	state.begin_reference_origin_selection()
	assert not state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 195, 195)
	assert state.selected_origin_detection_index is None
	assert state.reference_origin_selection_mode


def test_reference_preview_follows_selected_center_not_the_original_click() -> None:
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	state = ApplicationState(config=_application_config())
	state.update_frame(frame, detections=_detections())
	state.reference_start()
	state.begin_reference_origin_selection()
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 143, 70)

	moved = [
		_detection(143.0, 70.0, label="distractor"),
		*[
			detection for detection in _detections()
			if detection["label"] != "origin"
		],
		_detection(141.0, 70.0, label="origin"),
	]
	state.update_frame(frame, detections=moved)
	assert moved[state.selected_origin_detection_index]["label"] == "origin"
	result = state.reference_preview()
	assert result.reference_origin_point.reference_position == (141.0, 70.0)


def test_origin_is_permanent_excluded_and_line_ids_are_deterministic() -> None:
	detections = _detections()
	result = initialise_reference(
		detections, _reference_config(), (200, 200),
		reference_origin_detection_index=6,
	)
	assert result.valid
	assert result.reference_origin_point.id == "REFERENCE_ORIGIN"
	assert result.reference_origin_point.point_group == PointGroup.REFERENCE_ORIGIN
	assert all(
		"REFERENCE_ORIGIN" not in line.point_ids for line in result.lines
	)
	assert result.lines[0].point_ids == [f"LINE_A_{index:02d}" for index in range(3)]
	assert result.lines[1].point_ids == [f"LINE_B_{index:02d}" for index in range(3)]

	shuffled = [detections[index] for index in (5, 2, 6, 0, 4, 1, 3)]
	origin_index = next(
		index for index, detection in enumerate(shuffled)
		if detection["label"] == "origin"
	)
	second = initialise_reference(
		shuffled, _reference_config(), (200, 200),
		reference_origin_detection_index=origin_index,
	)
	assert [line.point_ids for line in second.lines] == [
		line.point_ids for line in result.lines
	]


def test_vertical_line_ids_are_deterministic_across_detection_order() -> None:
	detections = [
		*[_detection(30.0, y, label=f"A{index}") for index, y in enumerate((40, 70, 100))],
		*[_detection(90.0, y, label=f"B{index}") for index, y in enumerate((40, 70, 100))],
		_detection(150.0, 150.0, label="origin"),
	]
	expected_mapping = None
	rng = np.random.default_rng(7)
	for _ in range(20):
		order = rng.permutation(len(detections))
		shuffled = [detections[int(index)] for index in order]
		origin_index = next(
			index for index, detection in enumerate(shuffled)
			if detection["label"] == "origin"
		)
		result = initialise_reference(
			shuffled,
			_reference_config(),
			(200, 200),
			reference_origin_detection_index=origin_index,
		)
		assert result.valid, result.fatal_errors
		mapping = {
			point.id: point.reference_position
			for point in result.points if point.point_group == PointGroup.DEFORMING
		}
		if expected_mapping is None:
			expected_mapping = mapping
		else:
			assert mapping == expected_mapping


def test_profile_round_trip_preserves_the_real_origin(tmp_path: Path) -> None:
	profile = _profile()
	path = tmp_path / "profile.json"
	save_reference_profile(profile, path)
	loaded = load_reference_profile(path)
	assert loaded.reference_origin_point_id == "REFERENCE_ORIGIN"
	assert loaded.reference_origin_position == (140.0, 70.0)
	assert sum(
		point.point_group == PointGroup.REFERENCE_ORIGIN
		for point in loaded.points
	) == 1


def test_translation_only_origin_frame_is_zero_and_translation_invariant() -> None:
	profile = _profile()
	baseline_points = copy.deepcopy(profile.points)
	baseline = extract_geometry(profile, baseline_points, _geometry_config())
	assert baseline.valid
	assert baseline.transform.mode == "translation_only"
	assert not baseline.transform.applied
	assert baseline.origin_relative_coordinates["REFERENCE_ORIGIN"] == (0.0, 0.0)
	assert baseline.reference_origin_relative_coordinates["REFERENCE_ORIGIN"] == (0.0, 0.0)

	translated = copy.deepcopy(profile.points)
	for point in translated:
		point.current_position = (
			point.current_position[0] + 17.25,
			point.current_position[1] - 9.5,
		)
	translated_geometry = extract_geometry(profile, translated, _geometry_config())
	np.testing.assert_allclose(
		translated_geometry.feature_vector, baseline.feature_vector, atol=1e-8
	)
	assert translated_geometry.origin_relative_coordinates["REFERENCE_ORIGIN"] == (0.0, 0.0)


def test_rotation_is_not_inferred_from_origin_but_rigid_mode_works() -> None:
	translation_profile = _profile()
	failed = extract_geometry(
		translation_profile,
		copy.deepcopy(translation_profile.points),
		_geometry_config("translation_and_rotation"),
	)
	assert not failed.valid
	assert "rigid reference" in " ".join(failed.fatal_errors).lower()

	rigid_profile = _profile(rigid=True, mode="translation_and_rotation")
	current = copy.deepcopy(rigid_profile.points)
	angle = np.deg2rad(12.0)
	rotation = np.asarray([
		[np.cos(angle), -np.sin(angle)],
		[np.sin(angle), np.cos(angle)],
	])
	for point in current:
		value = rotation @ np.asarray(point.reference_position) + np.asarray([8.0, -4.0])
		point.current_position = (float(value[0]), float(value[1]))
	rigid_geometry = extract_geometry(
		rigid_profile, current, _geometry_config("translation_and_rotation")
	)
	assert rigid_geometry.valid, rigid_geometry.fatal_errors
	assert rigid_geometry.transform.applied
	np.testing.assert_allclose(rigid_geometry.feature_vector, 0.0, atol=1e-4)


def test_lost_origin_is_not_substituted_and_confirmed_reacquisition_restores_geometry() -> None:
	profile = _profile()
	tracker = PointTracker(profile, _tracker_config())
	gray = np.zeros((200, 200), dtype=np.uint8)
	tracker.update(gray, _detections())

	without_origin = [
		detection for detection in _detections()
		if detection["label"] != "origin"
	]
	without_origin.append(_detection(141.0, 70.0, label="distractor"))
	points = tracker.update(gray, without_origin)
	origin = tracker.get_point("REFERENCE_ORIGIN")
	assert origin.status in {PointStatus.MISSING, PointStatus.INVALID}
	assert origin.current_position != (141.0, 70.0)
	lost = extract_geometry(profile, points, _geometry_config())
	assert not lost.origin_valid
	assert not lost.valid

	points = tracker.update(gray, _detections())
	assert tracker.get_point("REFERENCE_ORIGIN").status in {
		PointStatus.MISSING,
		PointStatus.INVALID,
	}
	origin_detection = next(
		detection for detection in _detections()
		if detection["label"] == "origin"
	)
	tracker.confirm_origin_reacquisition(origin_detection)
	assert tracker.get_point("REFERENCE_ORIGIN").status == PointStatus.RECOVERED
	recovered = extract_geometry(profile, points, _geometry_config())
	assert recovered.origin_valid
	assert recovered.valid


def test_adjacent_stretch_validation_checks_both_deforming_lines() -> None:
	tracker = PointTracker(_profile(), _tracker_config())
	gray = np.zeros((200, 200), dtype=np.uint8)
	tracker.update(gray, _detections())
	for x_position in range(100, 161, 10):
		detections = _detections()
		b2 = next(
			detection for detection in detections
			if detection["label"] == "B2"
		)
		b2["center"] = (float(x_position), 160.0)
		tracker.update(gray, detections)
	assert tracker.get_point("LINE_B_01").status == PointStatus.INVALID
	assert tracker.get_point("LINE_B_02").status == PointStatus.INVALID
	assert not tracker.valid


def test_origin_reacquire_command_requires_a_confirming_click(
	tmp_path: Path,
) -> None:
	state, frame = _accepted_application(tmp_path)
	distractor_frame = [
		detection for detection in _detections()
		if detection["label"] != "origin"
	]
	distractor_frame.append(_detection(141.0, 70.0, label="distractor"))
	state.update_frame(frame, detections=distractor_frame, timestamp=3.0)
	assert state.tracker.origin_reacquisition_required
	assert state.tracker.get_point("REFERENCE_ORIGIN").current_position != (141.0, 70.0)

	state.update_frame(frame, detections=_detections(), timestamp=4.0)
	assert state.tracker.origin_reacquisition_required
	assert execute_command("origin_reacquire", state)
	assert state.origin_reacquisition_selection_mode
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 140, 70)
	assert not state.origin_reacquisition_selection_mode
	assert state.tracker.get_point("REFERENCE_ORIGIN").status == PointStatus.RECOVERED
	assert state.tracker.valid
	state.update_frame(frame, detections=_detections(), timestamp=5.0)
	assert state.current_geometry.origin_valid
	assert state.current_geometry.valid


def test_dataset_commands_write_rows_and_invalid_frames_retry(tmp_path: Path) -> None:
	state, frame = _accepted_application(tmp_path)
	assert execute_command("dataset_start experiment-a 7.5", state)
	assert execute_command("dataset_sample 1", state)
	state.update_frame(frame, detections=_detections(), timestamp=3.0)
	assert state.dataset_session.accepted == 1
	rows = list(csv.DictReader(state.dataset_session.root.joinpath("samples.csv").open()))
	assert len(rows) == 1
	assert rows[0]["reference_origin_point_id"] == "REFERENCE_ORIGIN"
	assert json.loads(rows[0]["origin_relative_point_coordinates"])["REFERENCE_ORIGIN"] == [0.0, 0.0]
	assert rows[0]["tracking_valid"] == "True"

	assert execute_command("dataset_sample 1", state)
	missing = [
		detection for detection in _detections()
		if detection["label"] != "origin"
	]
	state.update_frame(frame, detections=missing, timestamp=4.0)
	assert state.pending_dataset_frames == 1
	assert state.dataset_session.rejected == 1
	state.update_frame(frame, detections=missing, timestamp=5.0)
	assert state.pending_dataset_frames == 0
	assert state.dataset_session.rejected == 2


def test_dataset_validation_rejection_is_printed_and_shown_in_status(
	tmp_path: Path,
	capsys,
) -> None:
	state, _ = _accepted_application(tmp_path)
	state.start_dataset("rejection-report", 4.0)
	state.queue_dataset_samples(1)
	state.current_geometry.feature_vector[0] = np.nan
	assert state.process_pending_dataset_sample() is False
	assert "feature values are not finite" in state.last_dataset_result
	assert "rejected" in capsys.readouterr().out.lower()
	assert any(
		text.startswith("DATA LAST: REJECTED")
		for text, _ in state._status_lines()
	)


def test_feature_order_and_model_metadata_are_exact() -> None:
	profile = _profile()
	points = copy.deepcopy(profile.points)
	first = extract_geometry(profile, points, _geometry_config())
	second = extract_geometry(profile, list(reversed(points)), _geometry_config())
	assert first.feature_names == second.feature_names
	np.testing.assert_allclose(first.feature_vector, second.feature_vector)

	model = ForceModel(
		ZeroEstimator(),
		ModelMetadata(
			model_type="zero",
			feature_names=first.feature_names,
			calibrated_min_force_N=-1.0,
			calibrated_max_force_N=1.0,
			reference_profile_compatibility=profile.profile_id,
			geometry_configuration_version="test",
			feature_schema_version="test",
		),
	)
	bad = model.predict(
		first.feature_vector,
		list(reversed(first.feature_names)),
		tracking_quality=1.0,
		reference_quality=1.0,
		geometry_quality=1.0,
		reference_profile_id=profile.profile_id,
		geometry_version="test",
		feature_schema_version="test",
	)
	assert not bad.valid
	assert bad.raw_force_N is None


def test_live_inference_runs_in_frame_loop_and_pauses_on_origin_loss(tmp_path: Path) -> None:
	state, frame = _accepted_application(tmp_path)
	state.loaded_force_model = ForceModel(
		ZeroEstimator(),
		ModelMetadata(
			model_type="zero",
			feature_names=state.current_feature_names,
			calibrated_min_force_N=-1.0,
			calibrated_max_force_N=1.0,
			reference_profile_compatibility=state.reference_profile.profile_id,
			geometry_configuration_version="test",
			feature_schema_version="test",
		),
	)
	assert execute_command("force_start", state)
	state.update_frame(frame, detections=_detections(), timestamp=3.0)
	assert state.current_force_prediction.valid
	assert state.current_force_prediction.raw_force_N == 0.0

	missing = [
		detection for detection in _detections()
		if detection["label"] != "origin"
	]
	state.update_frame(frame, detections=missing, timestamp=4.0)
	assert not state.current_force_prediction.valid
	assert state.current_force_prediction.raw_force_N is None


def test_force_overlay_reports_model_qualities_and_out_of_range_value(
	tmp_path: Path,
) -> None:
	state, frame = _accepted_application(tmp_path)
	state.loaded_force_model = ForceModel(
		ConstantEstimator(3.0),
		ModelMetadata(
			model_type="constant",
			feature_names=state.current_feature_names,
			calibrated_min_force_N=-1.0,
			calibrated_max_force_N=1.0,
			reference_profile_compatibility=state.reference_profile.profile_id,
			geometry_configuration_version="test",
			feature_schema_version="test",
			model_name="constant-test",
		),
	)
	state.start_force_inference()
	state.update_frame(frame, detections=_detections(), timestamp=3.0)
	prediction = state.current_force_prediction
	assert prediction.raw_force_N == 3.0
	assert not prediction.valid
	lines = [text for text, _ in state._status_lines()]
	assert any(text == "FORCE RAW: 3.000 N" for text in lines)
	assert any(text == "MODEL: constant-test" for text in lines)
	assert any(text.startswith("QUALITY: tracking=") for text in lines)
	assert any("Outside calibrated force range" in text for text in lines)


def test_hough_validation_is_an_opt_in_reference_and_tracking_qualifier() -> None:
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	state = ApplicationState(config=_application_config())
	state.current_raw_frame = frame
	state.current_detections = _detections()
	for detection in state.current_detections:
		detection["circle_validation"] = True
	state.reference_start()
	state.begin_reference_origin_selection()
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 140, 70)
	state.config.circles.require_validation_for_reference = True
	state.current_detections[6]["circle_validation"] = False
	try:
		state.reference_preview()
	except ValueError as exc:
		assert "Hough-circle" in str(exc)
	else:
		raise AssertionError("A required origin Hough validation was ignored")

	tracking_state, _ = _accepted_application()
	tracking_state.config.circles.require_validation_for_tracking = True
	tracking_state.update_frame(frame, detections=_detections(), timestamp=3.0)
	assert not tracking_state.tracker.valid
	assert any(
		"Hough validation rejected" in warning
		for warning in tracking_state.runtime_warnings
	)


def test_reference_proposal_errors_are_carried_into_frame_warnings() -> None:
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	state = ApplicationState(config=_application_config())
	state.update_frame(frame, detections=_detections())
	state.reference_start()
	state.begin_reference_origin_selection()
	assert state.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 140, 70)
	incomplete = [
		detection for detection in _detections()
		if detection["label"] != "B2"
	]
	state.current_detections = incomplete
	result = state.reference_preview()
	assert not result.valid
	state.update_frame(frame, detections=incomplete)
	assert set(result.fatal_errors).issubset(state.runtime_warnings)


def test_malformed_artifacts_are_recoverable_command_failures(
	tmp_path: Path,
	capsys,
) -> None:
	state = ApplicationState(config=_application_config(tmp_path))
	profile_path = tmp_path / "bad-profile.json"
	profile_path.write_text('{"points": [null]}', encoding="utf-8")
	assert not execute_command(f"reference_load {profile_path}", state)

	model_path = tmp_path / "bad-model.pkl"
	model_path.write_bytes(pickle.dumps(7))
	model_path.with_suffix(".json").write_text(
		json.dumps({"model_type": "bad", "feature_names": ["x"]}),
		encoding="utf-8",
	)
	assert not execute_command(f"model_load {model_path}", state)

	bad_metadata_model = tmp_path / "bad-metadata.pkl"
	bad_metadata_model.write_bytes(pickle.dumps(ZeroEstimator()))
	bad_metadata_model.with_suffix(".json").write_text(
		json.dumps({
			"model_type": "bad",
			"feature_names": ["x"],
			"calibrated_min_force_N": {},
		}),
		encoding="utf-8",
	)
	assert not execute_command(f"model_load {bad_metadata_model}", state)
	output = capsys.readouterr().out
	assert output.count("Command failed:") == 3


def test_reference_status_prioritises_the_active_setup(
	tmp_path: Path,
	capsys,
) -> None:
	state, _ = _accepted_application(tmp_path)
	state.reference_start()
	assert execute_command("reference_status", state)
	output = capsys.readouterr().out
	assert "Reference setup active" in output
	assert "Accepted profile=" not in output


def test_training_uses_grouped_samples_and_refuses_implicit_overwrite(tmp_path: Path) -> None:
	header = [
		"tracking_valid", "origin_valid", "geometry_valid", "origin_status",
		"reference_origin_point_id", "known_force_N", "feature_names",
		"feature_values", "reference_profile_identifier",
		"geometry_configuration_identifier", "dataset_schema_version",
		"feature_schema_version", "experiment_id", "acquisition_batch_id",
		"force_step_id",
	]
	csv_path = tmp_path / "samples.csv"
	with csv_path.open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=header)
		writer.writeheader()
		for batch, force, values in (
			("batch-a", 0.0, [0.0, 0.0]),
			("batch-a", 0.0, [0.1, 0.0]),
			("batch-b", 10.0, [1.0, 1.0]),
			("batch-b", 10.0, [1.1, 1.0]),
		):
			writer.writerow({
				"tracking_valid": True,
				"origin_valid": True,
				"geometry_valid": True,
				"origin_status": "DETECTED",
				"reference_origin_point_id": "REFERENCE_ORIGIN",
				"known_force_N": force,
				"feature_names": json.dumps(["x", "y"]),
				"feature_values": json.dumps(values),
				"reference_profile_identifier": "profile-a",
				"geometry_configuration_identifier": "test",
				"dataset_schema_version": "2",
				"feature_schema_version": "test",
				"experiment_id": "experiment-a",
				"acquisition_batch_id": batch,
				"force_step_id": batch,
			})
	model_path = tmp_path / "force.pkl"
	_, metadata, report = train(
		[csv_path],
		model_path=model_path,
		validation_fraction=0.5,
		random_seed=7,
		split_group="acquisition_batch",
	)
	assert model_path.exists()
	assert model_path.with_suffix(".json").exists()
	assert metadata.training_metrics["sample_count"] == 2
	assert report["validation_metrics"]["sample_count"] == 2
	try:
		train([csv_path], model_path=model_path, split_group="acquisition_batch")
	except FileExistsError:
		pass
	else:
		raise AssertionError("Training silently overwrote an existing model")


def test_runtime_has_no_artificial_point_pipeline_dependency() -> None:
	runtime_source = inspect.getsource(main)
	assert "compute_and_draw_artificial_point" not in runtime_source
	assert "update_and_draw_tracked_points" not in runtime_source
	assert "filter_and_collect_points" not in runtime_source
