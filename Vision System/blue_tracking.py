"""Detector-corrected point tracking for the single calibrated blue target."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import cv2
import numpy as np


class BlueTrackStatus(str, Enum):
	UNSELECTED = "UNSELECTED"
	DETECTED = "DETECTED"
	RECOVERED = "RECOVERED"
	TRACKED_ONLY = "TRACKED_ONLY"
	MISSING = "MISSING"
	INVALID = "INVALID"


@dataclass
class BlueBlobTrack:
	selected: bool = False
	current_position: tuple[float, float] | None = None
	predicted_position: tuple[float, float] | None = None
	detected_position: tuple[float, float] | None = None
	status: BlueTrackStatus = BlueTrackStatus.UNSELECTED
	quality: float = 0.0
	missing_frame_count: int = 0
	status_reason: str = "Click a blue blob during calibration to select the target"


class BlueBlobTracker:
	"""Track one user-selected blue blob with LK flow and color redetection."""

	def __init__(self, config: dict[str, Any]):
		self.config = config
		self.track = BlueBlobTrack()
		self.previous_gray: np.ndarray | None = None

	@property
	def selected(self) -> bool:
		return self.track.selected

	@property
	def valid(self) -> bool:
		track = self.track
		if not track.selected or track.current_position is None:
			return False
		if track.status in (
			BlueTrackStatus.UNSELECTED,
			BlueTrackStatus.MISSING,
			BlueTrackStatus.INVALID,
		):
			return False
		if (
			track.status == BlueTrackStatus.TRACKED_ONLY
			and not bool(self.config.get("allow_tracked_only", True))
		):
			return False
		return track.quality >= float(
			self.config.get("minimum_tracking_quality", 0.0)
		)

	@property
	def invalid_reason(self) -> str:
		"""Return a specific user-facing explanation when the track is invalid."""
		track = self.track
		if not track.selected or track.current_position is None:
			return "Blue target is not selected"
		if track.status in (BlueTrackStatus.MISSING, BlueTrackStatus.INVALID):
			return track.status_reason
		if (
			track.status == BlueTrackStatus.TRACKED_ONLY
			and not bool(self.config.get("allow_tracked_only", True))
		):
			return (
				f"Color detection is missing and tracked-only output is disabled; "
				f"{track.status_reason}"
			)
		minimum_quality = float(self.config.get("minimum_tracking_quality", 0.0))
		if track.quality < minimum_quality:
			return (
				f"Blue tracking quality {track.quality:.3f} is below the configured "
				f"minimum {minimum_quality:.3f}; {track.status_reason}"
			)
		return ""

	def clear(self) -> None:
		self.track = BlueBlobTrack()
		self.previous_gray = None

	def bind(self, gray: np.ndarray, detection: dict[str, Any]) -> BlueBlobTrack:
		"""Bind the permanent BLUE_TARGET identity to a clicked detection."""
		try:
			position_array = np.asarray(detection["center"], dtype=np.float64)
			quality = float(detection.get("detection_quality", 1.0))
		except (KeyError, TypeError, ValueError, OverflowError) as exc:
			raise ValueError("The selected blue detection is invalid") from exc
		if (
			gray is None
			or position_array.shape != (2,)
			or not np.all(np.isfinite(position_array))
			or not math.isfinite(quality)
			or quality <= 0.0
		):
			raise ValueError("The selected blue detection is invalid")
		position = (float(position_array[0]), float(position_array[1]))
		self.track = BlueBlobTrack(
			selected=True,
			current_position=position,
			predicted_position=position,
			detected_position=position,
			status=BlueTrackStatus.DETECTED,
			quality=float(np.clip(quality, 0.0, 1.0)),
			missing_frame_count=0,
			status_reason="Blue target was selected from a color detection",
		)
		self.previous_gray = gray.copy()
		return self.track

	def _optical_flow_prediction(
		self,
		gray: np.ndarray,
	) -> tuple[np.ndarray, bool, float]:
		current = np.asarray(self.track.current_position, dtype=np.float32)
		if self.previous_gray is None or self.previous_gray.shape != gray.shape:
			return current, False, math.inf
		criteria = (
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
			int(self.config.get("termination_count", 20)),
			float(self.config.get("termination_epsilon", 0.03)),
		)
		next_points, status, error = cv2.calcOpticalFlowPyrLK(
			self.previous_gray,
			gray,
			current.reshape(1, 1, 2),
			None,
			winSize=tuple(int(value) for value in self.config.get("window_size", [21, 21])),
			maxLevel=int(self.config.get("maximum_pyramid_level", 3)),
			criteria=criteria,
		)
		if next_points is None or status is None or not bool(status.reshape(-1)[0]):
			return current, False, math.inf
		prediction = next_points.reshape(-1, 2)[0]
		flow_error = (
			math.inf if error is None else float(error.reshape(-1)[0])
		)
		if not np.all(np.isfinite(prediction)) or not math.isfinite(flow_error):
			return current, False, math.inf
		return prediction, True, flow_error

	@staticmethod
	def _valid_detections(
		detections: list[dict[str, Any]],
	) -> list[tuple[np.ndarray, float]]:
		result: list[tuple[np.ndarray, float]] = []
		for detection in detections:
			try:
				position = np.asarray(detection["center"], dtype=np.float64)
				quality = float(detection.get("detection_quality", 1.0))
			except (KeyError, TypeError, ValueError, OverflowError):
				continue
			if (
				position.shape == (2,)
				and np.all(np.isfinite(position))
				and math.isfinite(quality)
				and quality > 0.0
			):
				result.append((position, float(np.clip(quality, 0.0, 1.0))))
		return result

	def update(
		self,
		gray: np.ndarray,
		detections: list[dict[str, Any]],
	) -> BlueBlobTrack:
		if gray is None:
			raise ValueError("gray frame cannot be None")
		if not self.track.selected or self.track.current_position is None:
			self.previous_gray = gray.copy()
			return self.track

		track = self.track
		prediction, flow_ok, flow_error = self._optical_flow_prediction(gray)
		track.predicted_position = tuple(float(value) for value in prediction)
		gate = max(
			float(self.config.get("maximum_assignment_distance_px", 40.0)),
			1e-6,
		)
		candidates = self._valid_detections(detections)
		match: tuple[np.ndarray, float] | None = None
		match_distance = math.inf
		for position, detection_quality in candidates:
			distance = float(np.linalg.norm(position - prediction))
			if distance <= gate and distance < match_distance:
				match = (position, detection_quality)
				match_distance = distance

		was_missing = track.missing_frame_count > 0 or track.status in (
			BlueTrackStatus.MISSING,
			BlueTrackStatus.INVALID,
		)
		if match is not None:
			position, detection_quality = match
			resolved = tuple(float(value) for value in position)
			track.current_position = resolved
			track.detected_position = resolved
			track.status = (
				BlueTrackStatus.RECOVERED if was_missing else BlueTrackStatus.DETECTED
			)
			track.missing_frame_count = 0
			distance_quality = 1.0 - min(match_distance / gate, 1.0)
			track.quality = float(detection_quality * distance_quality)
			track.status_reason = (
				f"Blue color detection matched {match_distance:.2f}px from the "
				f"predicted position (limit {gate:.2f}px)"
			)
		else:
			maximum_flow_error = max(
				float(self.config.get("maximum_optical_flow_error", 25.0)),
				1e-6,
			)
			maximum_missing = int(self.config.get("maximum_missing_frames", 5))
			can_track_only = (
				flow_ok
				and flow_error <= maximum_flow_error
				and track.missing_frame_count < maximum_missing
			)
			track.detected_position = None
			track.missing_frame_count += 1
			if not candidates:
				detection_reason = (
					"No blue blob passed the active HSV and area filters"
				)
			else:
				nearest_distance = min(
					float(np.linalg.norm(position - prediction))
					for position, _quality in candidates
				)
				detection_reason = (
					f"Nearest blue candidate was {nearest_distance:.2f}px from the "
					f"predicted position, outside the {gate:.2f}px assignment limit"
				)
			if not flow_ok:
				flow_reason = "optical flow could not produce a valid point"
			elif flow_error > maximum_flow_error:
				flow_reason = (
					f"optical-flow error {flow_error:.2f} exceeded the configured "
					f"maximum {maximum_flow_error:.2f}"
				)
			else:
				flow_reason = f"optical-flow error was {flow_error:.2f}"
			if can_track_only:
				track.current_position = track.predicted_position
				track.status = BlueTrackStatus.TRACKED_ONLY
				track.quality = float(
					max(0.0, 1.0 - flow_error / maximum_flow_error) * 0.5
				)
				track.status_reason = (
					f"{detection_reason}; using optical flow for missing frame "
					f"{track.missing_frame_count}/{maximum_missing} ({flow_reason})"
				)
			else:
				track.status = (
					BlueTrackStatus.MISSING
					if track.missing_frame_count <= maximum_missing
					else BlueTrackStatus.INVALID
				)
				track.quality = 0.0
				missing_limit_reason = (
					f"; maximum missing-frame limit {maximum_missing} was exceeded"
					if track.status == BlueTrackStatus.INVALID else ""
				)
				track.status_reason = (
					f"{detection_reason}; {flow_reason}; missing frame "
					f"{track.missing_frame_count}/{maximum_missing}"
					f"{missing_limit_reason}"
				)
		self.previous_gray = gray.copy()
		return track
