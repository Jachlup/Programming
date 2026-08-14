"""Small atomic registry for trained and explicitly activated force models."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from force_model import ModelMetadata


class ModelRegistry:
	def __init__(self, path: str | Path) -> None:
		self.path = Path(path)

	def load(self) -> dict[str, Any]:
		if not self.path.exists():
			return {"schema_version": 1, "active_model": None, "history": [], "models": []}
		raw = json.loads(self.path.read_text(encoding="utf-8"))
		if not isinstance(raw, dict) or raw.get("schema_version") != 1:
			raise ValueError("Model registry has an unsupported schema")
		for key, default in (
			("active_model", None), ("history", []), ("models", []),
		):
			raw.setdefault(key, default)
		return raw

	def _save(self, mapping: dict[str, Any]) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		temporary_name = ""
		try:
			with tempfile.NamedTemporaryFile(
				"w", encoding="utf-8", dir=self.path.parent,
				prefix=f".{self.path.name}-", delete=False,
			) as stream:
				temporary_name = stream.name
				json.dump(mapping, stream, indent=2, allow_nan=False)
				stream.write("\n")
			os.replace(temporary_name, self.path)
		finally:
			if temporary_name:
				Path(temporary_name).unlink(missing_ok=True)

	@staticmethod
	def _identifier(model_path: Path, metadata: ModelMetadata) -> str:
		return f"{metadata.model_name}:{metadata.model_version}:{model_path.resolve()}"

	def register(
		self,
		model_path: str | Path,
		metadata_path: str | Path,
		metadata: ModelMetadata,
	) -> str:
		model = Path(model_path).resolve()
		meta = Path(metadata_path).resolve()
		identifier = self._identifier(model, metadata)
		mapping = self.load()
		entry = {
			"id": identifier,
			"model_name": metadata.model_name,
			"model_version": metadata.model_version,
			"model_path": str(model),
			"metadata_path": str(meta),
			"training_date": metadata.training_date,
			"registered_at": datetime.now(timezone.utc).isoformat(),
			"validation_metrics": metadata.validation_metrics,
		}
		mapping["models"] = [
			item for item in mapping["models"] if item.get("id") != identifier
		] + [entry]
		self._save(mapping)
		return identifier

	def mark_active(self, identifier: str) -> None:
		mapping = self.load()
		if not any(item.get("id") == identifier for item in mapping["models"]):
			raise ValueError("Cannot activate an unregistered model")
		previous = mapping.get("active_model")
		if previous and previous != identifier:
			mapping["history"].append(previous)
		mapping["active_model"] = identifier
		self._save(mapping)

	def previous_model(self) -> dict[str, Any] | None:
		mapping = self.load()
		if not mapping["history"]:
			return None
		identifier = mapping["history"][-1]
		return next(
			(item for item in mapping["models"] if item.get("id") == identifier),
			None,
		)

	def mark_rollback(self) -> dict[str, Any]:
		mapping = self.load()
		if not mapping["history"]:
			raise ValueError("No previously active model is available")
		previous = mapping["history"].pop()
		current = mapping.get("active_model")
		if current and current != previous:
			mapping["history"].append(current)
		mapping["active_model"] = previous
		entry = next(
			(item for item in mapping["models"] if item.get("id") == previous),
			None,
		)
		if entry is None:
			raise ValueError("Previous model is missing from the registry")
		self._save(mapping)
		return entry
