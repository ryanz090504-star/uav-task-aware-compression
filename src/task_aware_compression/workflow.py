from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .benchmark import _detect, run_yolo_dataset
from .compressor import compress_adaptive, save_preview
from .dataset import YoloPersonDataset
from .detector import Detector, draw_detections
from .types import CompressionConfig
from .visualization import (
    save_compression_strength_visualization,
    save_importance_visualization,
    save_metric_visualizations,
    save_small_person_zoom,
    save_tile_quality_visualization,
    save_workflow_diagram,
    write_workflow_manifest,
)


def run_workflow(
    images: str | Path,
    labels: str | Path,
    output_dir: str | Path,
    detector: Detector,
    limit: int | None = None,
    person_class: int = 0,
    oracle_importance: bool = False,
    visual_samples: int = 4,
    tile_size: int = 64,
    include_hevc: bool = True,
    metadata_dir: str | Path | None = None,
    uniform_qualities: Iterable[int] = (90, 60, 35, 15),
    pairing_qualities: Iterable[int] = tuple(range(1, 101)),
    detection_batch_size: int = 1,
    tile_workers: int = 1,
    importance_semantics: str = "historical_full_gaussian",
) -> dict[str, Any]:
    """Run the real-data benchmark and generate paper-ready inspection figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_yolo_dataset(
        images,
        labels,
        output_dir / "benchmark",
        detector,
        limit,
        person_class,
        oracle_importance,
        research_methods=True,
        adaptive_tile_size=tile_size,
        include_hevc=include_hevc,
        metadata_dir=metadata_dir,
        uniform_qualities=uniform_qualities,
        pairing_qualities=pairing_qualities,
        detection_batch_size=detection_batch_size,
        tile_workers=tile_workers,
        importance_semantics=importance_semantics,
    )
    save_metric_visualizations(output_dir / "benchmark" / "records.csv", output_dir)
    save_workflow_diagram(output_dir / "workflow_diagram.png")
    visual_dir = output_dir / "visual_samples"
    dataset = YoloPersonDataset(images, labels, person_class, metadata_dir)
    for index, sample in enumerate(dataset):
        if index >= visual_samples or (limit is not None and index >= limit):
            break
        sample_dir = visual_dir / sample.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        predictions, detection_ms = _detect(detector, sample.image)
        importance_detections = sample.ground_truth if oracle_importance else predictions
        config = CompressionConfig(tile_size=tile_size, tile_workers=tile_workers, background_quality=5, person_quality=75, importance_semantics=importance_semantics)
        result = compress_adaptive(sample.image, importance_detections, config)
        Image.fromarray(sample.image).save(sample_dir / "original.jpg")
        draw_detections(sample.image, predictions, sample_dir / "candidate_detections.jpg")
        save_importance_visualization(sample.image, result.importance, importance_detections, sample_dir / "importance_visualization.png")
        save_tile_quality_visualization(sample.image, result.importance, config, result.tile_qualities, sample_dir / "tile_quality_visualization.png")
        save_compression_strength_visualization(sample.image, predictions, sample_dir / "compression_strength.png", tile_size, importance_semantics=importance_semantics)
        save_small_person_zoom(sample.image, sample.ground_truth, result.image, sample_dir / "small_person_zoom.png")
        save_preview(result, sample_dir / "adaptive_preview.jpg")
        (sample_dir / "sample_metrics.json").write_text(json.dumps({"sample": sample.name, "candidate_count": len(predictions), "ground_truth_count": len(sample.ground_truth), "adaptive_file_size_bytes": result.file_size, "adaptive_bits_per_pixel": result.bits_per_pixel, "adaptive_encode_ms": result.encode_ms, "adaptive_decode_ms": result.decode_ms, "candidate_detection_ms": detection_ms, "oracle_importance": oracle_importance}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_workflow_manifest(output_dir, summary)
    return summary
