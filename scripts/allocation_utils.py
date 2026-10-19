"""Shared V2 preparation and paired-metric helpers; no pilot entry point."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from task_aware_compression.dataset import DetectionSample, YoloPersonDataset
from task_aware_compression.types import Detection
from strict_evaluation import _bootstrap_delta, _cluster_bootstrap_delta, _micro

def sequence(path: Path | str) -> str:
    return Path(path).stem.split("_")[0]


def choose_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(paths):
        groups.setdefault(sequence(path), []).append(path)
    rng = np.random.default_rng(seed)
    names = list(groups)
    rng.shuffle(names)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    for frame in range(2):
        for name in names:
            if len(chosen) >= count:
                return chosen
            if frame < len(groups[name]):
                chosen.append(groups[name][frame])
    if len(chosen) < count:
        raise ValueError(f"requested {count} images but only {len(chosen)} are available at <=2 frames per sequence")
    return chosen


def load_sample(dataset: YoloPersonDataset, path: Path) -> DetectionSample:
    if not (dataset.labels / path.relative_to(dataset.images).with_suffix(".txt")).is_file():
        raise FileNotFoundError(f"missing label for {path}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    return DetectionSample(array, dataset._labels_for(path, width, height), path.stem, path, dataset._metadata_for(path))


def rgb_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    # The project's NumPy predict interface is treated as BGR by Ultralytics.
    return torch.from_numpy(array[..., ::-1].copy()).permute(2, 0, 1).float().to(device) / 255


def serialize_detections(detections: list[Detection]) -> list[dict]:
    return [{"box": asdict(det.box), "confidence": det.confidence, "label": det.label} for det in detections]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def paired_report(rows: list[dict], left: str, right: str, seed: int, repeats: int) -> dict:
    lookup = {(row["sample"], row["setting"]): row for row in rows}
    pairs = [(row, lookup[(row["sample"], right)]) for row in rows if row["setting"] == left]
    errors = np.array([abs(a["file_size_bytes"] / b["file_size_bytes"] - 1) for a, b in pairs])
    result = {
        "delta_micro_recall": _bootstrap_delta(pairs, "micro_recall", repeats, seed),
        "sequence_ci": _cluster_bootstrap_delta(pairs, "micro_recall", repeats, seed),
        "delta_micro_precision": _micro([a for a, _ in pairs])["precision"] - _micro([b for _, b in pairs])["precision"],
        "size_error_median": float(np.median(errors)), "size_error_p95": float(np.percentile(errors, 95)),
        "size_error_max": float(errors.max()), "within_2pct": int((errors <= .02).sum()),
        "within_5pct": int((errors <= .05).sum()),
    }
    for threshold in (.02, .05):
        subset = [pair for pair, error in zip(pairs, errors) if error <= threshold]
        result[f"within_{int(threshold * 100)}pct_sensitivity"] = {
            "n": len(subset),
            "delta_micro_recall": _bootstrap_delta(subset, "micro_recall", repeats, seed) if subset else None,
            "sequence_ci": _cluster_bootstrap_delta(subset, "micro_recall", repeats, seed) if subset else None,
            "delta_micro_precision": _micro([a for a, _ in subset])["precision"] - _micro([b for _, b in subset])["precision"] if subset else None,
        }
    return result
