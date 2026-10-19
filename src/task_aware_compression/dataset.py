from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterator
import numpy as np
from PIL import Image
from .types import BoundingBox, Detection

@dataclass
class DetectionSample:
    image: np.ndarray
    ground_truth: list[Detection]
    name: str
    path: Path
    metadata: dict[str, float | int | str] = field(default_factory=dict)

class YoloPersonDataset:
    """Load images and normalized YOLO labels for person-class evaluation."""
    def __init__(self, images: str | Path, labels: str | Path, person_class: int = 0, metadata_dir: str | Path | None = None):
        self.images = Path(images); self.labels = Path(labels); self.person_class = person_class
        inferred_metadata = self.labels.parent.parent / "metadata" / self.labels.name
        self.metadata_dir = Path(metadata_dir) if metadata_dir is not None else (inferred_metadata if inferred_metadata.exists() else None)
        if not self.images.exists(): raise FileNotFoundError(self.images)
        if not self.labels.exists(): raise FileNotFoundError(self.labels)
    def paths(self) -> list[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted(path for path in self.images.rglob("*") if path.suffix.lower() in extensions)
    def _labels_for(self, image_path: Path, width: int, height: int) -> list[Detection]:
        label_path = self.labels / image_path.relative_to(self.images).with_suffix(".txt")
        if not label_path.exists(): return []
        result: list[Detection] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.strip().split()
            if len(values) < 5 or int(float(values[0])) != self.person_class: continue
            center_x, center_y, box_width, box_height = map(float, values[1:5])
            result.append(Detection(BoundingBox((center_x - box_width / 2) * width, (center_y - box_height / 2) * height, (center_x + box_width / 2) * width, (center_y + box_height / 2) * height), 1.0))
        return result
    def _metadata_for(self, image_path: Path) -> dict[str, float | int | str]:
        if self.metadata_dir is None:
            return {}
        path = self.metadata_dir / image_path.relative_to(self.images).with_suffix(".json")
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        annotations = data.get("persons", [])
        if not isinstance(annotations, list):
            return {}
        occlusion = [float(item["occlusion"]) for item in annotations if isinstance(item, dict) and item.get("occlusion") is not None]
        truncation = [float(item["truncation"]) for item in annotations if isinstance(item, dict) and item.get("truncation") is not None]
        return {
            "max_occlusion": max(occlusion) if occlusion else "",
            "mean_occlusion": float(np.mean(occlusion)) if occlusion else "",
            "max_truncation": max(truncation) if truncation else "",
            "mean_truncation": float(np.mean(truncation)) if truncation else "",
        }
    def __iter__(self) -> Iterator[DetectionSample]:
        for path in self.paths():
            with Image.open(path) as image: array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            height, width = array.shape[:2]
            yield DetectionSample(array, self._labels_for(path, width, height), path.stem, path, self._metadata_for(path))
