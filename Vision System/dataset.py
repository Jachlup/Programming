"""Experiment-oriented storage for force-calibration samples."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import yaml

from geometry import GeometryResult
from tracking import ReferenceProfile, TrackedPoint


@dataclass
class DatasetSession:
	root: Path
	experiment_id: str
	profile_id: str
	geometry_id: str
	save_images: bool = True
	accepted: int = 0
	rejected: int = 0
	_samples: list[dict[str, Any]] = field(default_factory=list)

	@classmethod
	def start(cls, dataset_directory: str, experiment_id: str,
			  profile: ReferenceProfile, geometry_id: str, save_images: bool = True) -> "DatasetSession":
		root = Path(dataset_directory) / experiment_id
		(root / "images").mkdir(parents=True, exist_ok=True)
		metadata = {
			"experiment_id": experiment_id, "created_at": datetime.now(timezone.utc).isoformat(),
			"reference_profile_identifier": profile.profile_id,
			"geometry_configuration_identifier": geometry_id,
		}
		(root / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
		return cls(root, experiment_id, profile.profile_id, geometry_id, save_images)

	def add_sample(self, *, timestamp: float, frame_number: int, known_force_N: float,
				   points: list[TrackedPoint], geometry: GeometryResult, raw_frame) -> bool:
		if not geometry.valid:
			self.rejected += 1
			return False
		sample_id = self.accepted + 1
		image_name = f"frame_{frame_number:06d}.png" if self.save_images else ""
		if image_name:
			cv2.imwrite(str(self.root / "images" / image_name), raw_frame)
		row = {
			"timestamp": timestamp, "frame_number": frame_number,
			"experiment_id": self.experiment_id, "sample_id": sample_id,
			"known_force_N": known_force_N, "tracking_valid": True,
			"tracking_quality": min((p.tracking_quality for p in points), default=0.0),
			"reference_transformation_valid": geometry.transform.valid,
			"reference_quality": geometry.transform.quality,
			"raw_point_coordinates": json.dumps({p.id: p.current_position for p in points}),
			"compensated_point_coordinates": json.dumps(geometry.compensated_points),
			"point_statuses": json.dumps({p.id: p.status.value for p in points}),
			"feature_names": json.dumps(geometry.feature_names),
			"feature_values": json.dumps(geometry.feature_vector.tolist()),
			"image_filename": image_name, "reference_profile_identifier": self.profile_id,
			"geometry_configuration_identifier": self.geometry_id,
		}
		self._samples.append(row)
		self.accepted += 1
		self._flush()
		return True

	def _flush(self) -> None:
		if not self._samples:
			return
		with (self.root / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
			writer = csv.DictWriter(stream, fieldnames=list(self._samples[0]))
			writer.writeheader()
			writer.writerows(self._samples)
