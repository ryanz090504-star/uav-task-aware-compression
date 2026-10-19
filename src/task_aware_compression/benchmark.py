from __future__ import annotations
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
import numpy as np
from PIL import Image
from .compressor import compress_adaptive, compress_uniform, save_encoded, save_importance_map, save_preview
from .detector import Detector, draw_detections
from .metrics import evaluate_detections
from .synthetic import SyntheticSample, make_synthetic_dataset
from .types import BoundingBox, CompressionConfig, Detection

def _image_read(path: str | Path) -> np.ndarray:
    with Image.open(path) as image: return np.asarray(image.convert("RGB"), dtype=np.uint8)

DEFAULT_UNIFORM_QUALITIES = (90, 60, 35, 15)
DEFAULT_PAIRING_QUALITIES = tuple(range(1, 101))


def _normalize_qualities(qualities: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(quality) for quality in qualities}))
    if not normalized or normalized[0] < 1 or normalized[-1] > 100:
        raise ValueError("JPEG qualities must contain values in [1, 100]")
    return normalized


def _match_uniform_quality(target_bytes: int, candidates: dict[int, Any]) -> tuple[int, Any]:
    quality, result = min(candidates.items(), key=lambda item: abs(item[1].file_size - target_bytes))
    return quality, result

def _sample_context(sample: Any) -> dict[str, float | int | str]:
    image_area = max(1, sample.image.shape[0] * sample.image.shape[1])
    areas = [detection.box.area / image_area for detection in sample.ground_truth]
    metadata = getattr(sample, "metadata", {}) or {}
    return {
        "target_count": len(areas),
        "small_target_count": sum(area <= 0.018 for area in areas),
        "mean_target_area": float(np.mean(areas)) if areas else 0.0,
        "max_occlusion": metadata.get("max_occlusion", ""),
        "mean_occlusion": metadata.get("mean_occlusion", ""),
        "max_truncation": metadata.get("max_truncation", ""),
        "mean_truncation": metadata.get("mean_truncation", ""),
    }


def _record(
    sample: Any,
    mode: str,
    setting: str,
    result: Any,
    predictions: list[Detection],
    detection_ms: float,
    paired_quality: int | None = None,
    candidate_detection_ms: float = 0.0,
    paired_target_file_size_bytes: int | None = None,
    paired_size_error_bytes: int | None = None,
) -> dict[str, Any]:
    metrics = evaluate_detections(predictions, sample.ground_truth, result.image.shape[:2])
    incremental_ms = result.encode_ms + result.decode_ms + detection_ms
    full_pipeline_ms = candidate_detection_ms + incremental_ms
    target_bytes = paired_target_file_size_bytes if paired_target_file_size_bytes is not None else ""
    error_bytes = paired_size_error_bytes if paired_size_error_bytes is not None else ""
    error_fraction = (
        abs(paired_size_error_bytes) / max(1, paired_target_file_size_bytes)
        if paired_target_file_size_bytes is not None and paired_size_error_bytes is not None
        else ""
    )
    return {
        "sample": sample.name,
        "mode": mode,
        "setting": setting,
        "paired_uniform_quality": paired_quality if paired_quality is not None else "",
        "paired_target_file_size_bytes": target_bytes,
        "paired_size_error_bytes": error_bytes,
        "paired_size_error_fraction": error_fraction,
        "file_size_bytes": result.file_size,
        "raw_size_bytes": result.raw_size,
        "compression_ratio": result.compression_ratio,
        "bits_per_pixel": result.bits_per_pixel,
        "candidate_detection_ms": candidate_detection_ms,
        "encode_ms": result.encode_ms,
        "decode_ms": result.decode_ms,
        "detection_ms": detection_ms,
        "incremental_pipeline_ms": incremental_ms,
        "end_to_end_ms": full_pipeline_ms,
        "fps": 1000.0 / max(1e-6, full_pipeline_ms),
        **_sample_context(sample),
        **metrics,
    }

def _detect(detector: Detector, image: np.ndarray) -> tuple[list[Detection], float]:
    start = time.perf_counter(); predictions = detector.predict(image); elapsed = (time.perf_counter() - start) * 1000
    return predictions, elapsed


def _detect_batch(detector: Detector, images: list[np.ndarray], batch_size: int = 4) -> tuple[list[list[Detection]], float]:
    if not images:
        return [], 0.0
    start = time.perf_counter()
    predict_batch = getattr(detector, "predict_batch", None)
    if predict_batch is None:
        predictions = [detector.predict(image) for image in images]
    else:
        predictions = predict_batch(images, batch_size=batch_size)
    elapsed = (time.perf_counter() - start) * 1000 / len(images)
    return predictions, elapsed


def _write_checkpoint(output_dir: Path, records: list[dict[str, Any]], processed: int, total: int) -> None:
    if not records:
        return
    fields = list(records[0].keys())
    with (output_dir / "records.partial.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "progress.json").write_text(
        json.dumps({"processed_images": processed, "total_images": total, "record_count": len(records)}, indent=2),
        encoding="utf-8",
    )

def run_demo(
    output_dir: str | Path = "outputs/demo",
    sample_count: int = 6,
    seed: int = 7,
    detector: Detector | None = None,
    pairing_qualities: Iterable[int] = DEFAULT_PAIRING_QUALITIES,
) -> dict[str, Any]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    detector = detector or __import__("task_aware_compression.detector", fromlist=["ColorBlobPersonDetector"]).ColorBlobPersonDetector()
    samples = make_synthetic_dataset(sample_count, seed=seed)
    records: list[dict[str, Any]] = []
    uniform_qualities = DEFAULT_UNIFORM_QUALITIES
    pairing_qualities = _normalize_qualities(pairing_qualities)
    adaptive_configs = [("adaptive_bg5_fg75_t256", CompressionConfig(tile_size=256, background_quality=5, person_quality=75)), ("adaptive_bg8_fg82_t256", CompressionConfig(tile_size=256, background_quality=8, person_quality=82))]
    for sample in samples:
        sample_dir = output_dir / sample.name; sample_dir.mkdir(exist_ok=True)
        Image.fromarray(sample.image).save(sample_dir / "original.png")
        original_predictions, candidate_detection_ms = _detect(detector, sample.image)
        draw_detections(sample.image, original_predictions, sample_dir / "detections.png")
        for quality in uniform_qualities:
            result = compress_uniform(sample.image, quality)
            predictions, detection_ms = _detect(detector, result.image)
            save_encoded(result, sample_dir / f"uniform_q{quality}.jpg")
            save_preview(result, sample_dir / f"uniform_q{quality}_preview.jpg")
            records.append(_record(sample, "uniform", f"q{quality}", result, predictions, detection_ms))
        candidates = {quality: compress_uniform(sample.image, quality) for quality in pairing_qualities}
        for name, config in adaptive_configs:
            result = compress_adaptive(sample.image, original_predictions, config)
            predictions, detection_ms = _detect(detector, result.image)
            save_encoded(result, sample_dir / f"{name}.tac")
            save_preview(result, sample_dir / f"{name}_preview.jpg")
            save_importance_map(result.importance, sample_dir / f"{name}_importance.png")
            paired_quality, paired = _match_uniform_quality(result.file_size, candidates)
            pairing_error = paired.file_size - result.file_size
            records.append(_record(sample, "adaptive", name, result, predictions, detection_ms, paired_quality, candidate_detection_ms, result.file_size, pairing_error))
            paired_predictions, paired_detection_ms = _detect(detector, paired.image)
            records.append(_record(sample, "uniform_paired", f"paired_{name}", paired, paired_predictions, paired_detection_ms, paired_quality, 0.0, result.file_size, pairing_error))
    fields = list(records[0].keys()) if records else []
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    summary = summarize_records(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "experiment_notes.md").write_text(render_experiment_notes(summary), encoding="utf-8")
    return summary

def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records: groups.setdefault(record["setting"], []).append(record)
    summary: dict[str, Any] = {"record_count": len(records), "groups": {}}
    numeric = ["recall", "precision", "mAP50", "ap_small", "small_recall", "miss_rate", "bits_per_pixel", "compression_ratio", "file_size_bytes", "candidate_detection_ms", "encode_ms", "decode_ms", "detection_ms", "incremental_pipeline_ms", "end_to_end_ms", "fps"]
    for name, group in groups.items():
        summary["groups"][name] = {key: round(float(sum(float(row[key]) for row in group) / len(group)), 6) for key in numeric if group}
    adaptive = [row for row in records if row["mode"] == "adaptive"]
    paired = [row for row in records if row["mode"] == "uniform_paired"]
    if adaptive and paired:
        summary["paired_comparison"] = {"adaptive_recall": round(sum(row["recall"] for row in adaptive) / len(adaptive), 6), "uniform_recall": round(sum(row["recall"] for row in paired) / len(paired), 6), "adaptive_ap_small": round(sum(row["ap_small"] for row in adaptive) / len(adaptive), 6), "uniform_ap_small": round(sum(row["ap_small"] for row in paired) / len(paired), 6), "adaptive_small_recall": round(sum(row["small_recall"] for row in adaptive) / len(adaptive), 6), "uniform_small_recall": round(sum(row["small_recall"] for row in paired) / len(paired), 6), "adaptive_file_size": round(sum(row["file_size_bytes"] for row in adaptive) / len(adaptive), 2), "uniform_file_size": round(sum(row["file_size_bytes"] for row in paired) / len(paired), 2)}
        pairing: dict[str, dict[str, float]] = {}
        for setting in sorted({str(row["setting"]) for row in adaptive}):
            errors = [float(row["paired_size_error_fraction"]) for row in adaptive if row["setting"] == setting and row["paired_size_error_fraction"] != ""]
            if errors:
                median_error = float(np.median(errors))
                p95_error = float(np.percentile(errors, 95))
                pairing[setting] = {
                    "n": len(errors),
                    "mean_absolute_size_error_fraction": round(float(np.mean(errors)), 6),
                    "median_absolute_size_error_fraction": round(median_error, 6),
                    "p95_absolute_size_error_fraction": round(p95_error, 6),
                    "max_absolute_size_error_fraction": round(float(np.max(errors)), 6),
                    "meets_strict_rate_match_threshold": median_error <= 0.02 and p95_error <= 0.05,
                }
        if pairing:
            summary["paired_rate_matching"] = pairing
        strata = _paired_strata(records)
        if strata:
            summary["paired_strata"] = strata
    return summary


def _bucket_target_count(value: Any) -> str:
    count = int(float(value))
    return "1" if count == 1 else "2" if count == 2 else "3+"


def _bucket_target_size(value: Any) -> str:
    area = float(value)
    return "small<=1.8%" if area <= 0.018 else "large>1.8%"


def _bucket_occlusion(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return f"occlusion_{int(float(value))}"


def _paired_strata(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    lookup = {(str(row["sample"]), str(row["setting"])): row for row in records if row["mode"] == "uniform_paired"}
    definitions = {
        "target_count": ("target_count", _bucket_target_count),
        "mean_target_area": ("mean_target_area", _bucket_target_size),
        "max_occlusion": ("max_occlusion", _bucket_occlusion),
    }
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for adaptive in (row for row in records if row["mode"] == "adaptive"):
        paired = lookup.get((str(adaptive["sample"]), f"paired_{adaptive['setting']}"))
        if paired is None:
            continue
        for name, (field, bucket_fn) in definitions.items():
            try:
                bucket = bucket_fn(adaptive.get(field))
            except (TypeError, ValueError):
                bucket = None
            if bucket is not None:
                grouped.setdefault(str(adaptive["setting"]), {}).setdefault(name, {}).setdefault(bucket, []).append({"adaptive": adaptive, "paired": paired})
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for setting, dimensions in grouped.items():
        result[setting] = {}
        for dimension, buckets in dimensions.items():
            result[setting][dimension] = {}
            for bucket, pairs in buckets.items():
                recall_deltas = [float(pair["adaptive"]["recall"]) - float(pair["paired"]["recall"]) for pair in pairs]
                result[setting][dimension][bucket] = {
                    "n": len(pairs),
                    "mean_recall_delta": round(float(np.mean(recall_deltas)), 6),
                    "mean_bpp_delta": round(float(np.mean([float(pair["adaptive"]["bits_per_pixel"]) - float(pair["paired"]["bits_per_pixel"]) for pair in pairs])), 6),
                }
    return result

def render_experiment_notes(summary: dict[str, Any]) -> str:
    comparison = summary.get("paired_comparison", {})
    return """# UAV task-aware compression demo\n\nThis directory contains a reproducible dependency-light smoke benchmark based on the paper specification.\n\n- `records.csv`: per-image results for uniform and adaptive compression.\n- `summary.json`: aggregate detection, compression, and latency metrics.\n- `sample_*/`: source image, detection overlay, importance map, encoded payload, and decoded preview.\n\nThe paired experiment selects a uniform JPEG quality closest to each adaptive payload size.\n\n## Paired result\n\n- Adaptive human recall: {adaptive_recall}\n- Matched uniform human recall: {uniform_recall}\n- Adaptive small-person recall: {adaptive_small_recall}\n- Matched uniform small-person recall: {uniform_small_recall}\n- Adaptive mean file size: {adaptive_file_size} bytes\n- Matched uniform mean file size: {uniform_file_size} bytes\n\nThese are engineering smoke-test results on the included synthetic data, not a replacement for VisDrone/UAV benchmark results.\n""".format(adaptive_recall=comparison.get("adaptive_recall", "n/a"), uniform_recall=comparison.get("uniform_recall", "n/a"), adaptive_small_recall=comparison.get("adaptive_small_recall", "n/a"), uniform_small_recall=comparison.get("uniform_small_recall", "n/a"), adaptive_file_size=comparison.get("adaptive_file_size", "n/a"), uniform_file_size=comparison.get("uniform_file_size", "n/a"))

def compress_file(input_path: str | Path, output_path: str | Path, mode: str, detections: list[Detection] | None = None, quality: int = 50, config: CompressionConfig | None = None) -> dict[str, Any]:
    image = _image_read(input_path)
    if mode == "uniform": result = compress_uniform(image, quality)
    elif mode == "adaptive": result = compress_adaptive(image, detections or [], config)
    else: raise ValueError("mode must be uniform or adaptive")
    save_encoded(result, output_path)
    preview_path = Path(output_path).with_name(Path(output_path).stem + "_preview.jpg")
    save_preview(result, preview_path)
    payload = {"mode": result.mode, "output": str(output_path), "preview": str(preview_path), "file_size_bytes": result.file_size, "bits_per_pixel": result.bits_per_pixel, "compression_ratio": result.compression_ratio, "encode_ms": result.encode_ms, "decode_ms": result.decode_ms}
    if result.importance is not None:
        importance_path = Path(output_path).with_name(Path(output_path).stem + "_importance.png"); save_importance_map(result.importance, importance_path); payload["importance_map"] = str(importance_path)
    return payload

def load_detections(path: str | Path) -> list[Detection]:
    data = json.loads(Path(path).read_text(encoding="utf-8")); values = data.get("detections", data)
    return [Detection(BoundingBox(*item["box"]), float(item.get("confidence", 1.0)), item.get("label", "person")) for item in values]


def _fixed_roi_detections(detections: list[Detection]) -> list[Detection]:
    """Remove confidence and size-risk weighting for the Human ROI baseline."""
    return [Detection(item.box, 1.0, item.label) for item in detections]


def _random_roi_detections(sample_name: str, image_shape: tuple[int, int], detections: list[Detection]) -> list[Detection]:
    """Create a deterministic random-ROI baseline with matching box dimensions."""
    height, width = image_shape
    seed = int.from_bytes(hashlib.sha256(sample_name.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng(seed)
    templates = detections or [Detection(BoundingBox(0, 0, width * 0.06, height * 0.12), 1.0)]
    result: list[Detection] = []
    for item in templates:
        box_width = min(width, max(2.0, item.box.width))
        box_height = min(height, max(3.0, item.box.height))
        x1 = float(generator.uniform(0, max(1.0, width - box_width)))
        y1 = float(generator.uniform(0, max(1.0, height - box_height)))
        result.append(Detection(BoundingBox(x1, y1, x1 + box_width, y1 + box_height), 1.0, "random_roi"))
    return result



def run_yolo_dataset(
    images: str | Path,
    labels: str | Path,
    output_dir: str | Path,
    detector: Detector,
    limit: int | None = None,
    person_class: int = 0,
    oracle_importance: bool = False,
    research_methods: bool = False,
    adaptive_tile_size: int = 256,
    include_hevc: bool = False,
    metadata_dir: str | Path | None = None,
    uniform_qualities: Iterable[int] = DEFAULT_UNIFORM_QUALITIES,
    pairing_qualities: Iterable[int] = DEFAULT_PAIRING_QUALITIES,
    detection_batch_size: int = 1,
    tile_workers: int = 1,
    importance_semantics: str = "historical_full_gaussian",
) -> dict[str, Any]:
    """Run uniform/adaptive comparisons on a YOLO-labelled image folder."""
    from .dataset import YoloPersonDataset
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    uniform_qualities = _normalize_qualities(uniform_qualities)
    pairing_qualities = _normalize_qualities(pairing_qualities)
    if detection_batch_size < 1:
        raise ValueError("detection_batch_size must be positive")
    if tile_workers < 1:
        raise ValueError("tile_workers must be positive")
    dataset = YoloPersonDataset(images, labels, person_class, metadata_dir)
    records: list[dict[str, Any]] = []
    adaptive_configs = [(f"adaptive_bg5_fg75_t{adaptive_tile_size}", CompressionConfig(tile_size=adaptive_tile_size, tile_workers=tile_workers, background_quality=5, person_quality=75, importance_semantics=importance_semantics)), (f"adaptive_bg8_fg82_t{adaptive_tile_size}", CompressionConfig(tile_size=adaptive_tile_size, tile_workers=tile_workers, background_quality=8, person_quality=82, importance_semantics=importance_semantics))]
    dataset_count = len(dataset.paths())
    total = min(dataset_count, limit) if limit is not None else dataset_count
    for index, sample in enumerate(dataset):
        if limit is not None and index >= limit: break
        pre_predictions, pre_detection_ms = _detect(detector, sample.image)
        importance_detections = sample.ground_truth if oracle_importance else pre_predictions
        pixels = sample.image.shape[0] * sample.image.shape[1]
        source_size = sample.path.stat().st_size
        original = SimpleNamespace(image=sample.image, file_size=source_size, raw_size=sample.image.nbytes, compression_ratio=sample.image.nbytes / max(1, source_size), bits_per_pixel=source_size * 8 / pixels, encode_ms=0.0, decode_ms=0.0)
        records.append(_record(sample, "original", "original", original, pre_predictions, pre_detection_ms))
        uniform_results = {quality: compress_uniform(sample.image, quality) for quality in uniform_qualities}
        uniform_predictions, uniform_detection_ms = _detect_batch(
            detector, [result.image for result in uniform_results.values()], batch_size=detection_batch_size
        )
        for (quality, result), predictions in zip(uniform_results.items(), uniform_predictions):
            records.append(_record(sample, "uniform", f"q{quality}", result, predictions, uniform_detection_ms))
        if include_hevc:
            from .hevc import compress_hevc

            for crf in [25, 35, 45, 51]:
                result = compress_hevc(sample.image, crf)
                predictions, detection_ms = _detect(detector, result.image)
                records.append(_record(sample, "hevc", f"hevc_crf{crf}", result, predictions, detection_ms))
        method_inputs = [("risk_aware_adaptive", importance_detections)]
        if research_methods and not oracle_importance:
            method_inputs.extend(
                [
                    ("human_roi", _fixed_roi_detections(pre_predictions)),
                    ("random_roi", _random_roi_detections(sample.name, sample.image.shape[:2], pre_predictions)),
                ]
            )
        if oracle_importance:
            method_inputs = [("oracle_human_roi", sample.ground_truth)]
        pairing_results = {
            quality: uniform_results[quality] if quality in uniform_results else compress_uniform(sample.image, quality)
            for quality in pairing_qualities
        }
        pending: list[tuple[str, Any, Any, int, int]] = []
        for method, method_detections in method_inputs:
            for name, config in adaptive_configs:
                result = compress_adaptive(sample.image, method_detections, config)
                setting = f"{method}_{name}"
                paired_quality, paired = _match_uniform_quality(result.file_size, pairing_results)
                pending.append((setting, result, paired, paired_quality, paired.file_size - result.file_size))
        pending_predictions, pending_detection_ms = _detect_batch(
            detector, [image for _, result, paired, _, _ in pending for image in (result.image, paired.image)], batch_size=detection_batch_size
        )
        for item_index, (setting, result, paired, paired_quality, pairing_error) in enumerate(pending):
            predictions = pending_predictions[item_index * 2]
            paired_predictions = pending_predictions[item_index * 2 + 1]
            adaptive_candidate_ms = 0.0 if oracle_importance else pre_detection_ms
            records.append(_record(sample, "adaptive", setting, result, predictions, pending_detection_ms, paired_quality, adaptive_candidate_ms, result.file_size, pairing_error))
            records.append(_record(sample, "uniform_paired", f"paired_{setting}", paired, paired_predictions, pending_detection_ms, paired_quality, 0.0, result.file_size, pairing_error))
        if index < 8:
            sample_dir = output_dir / sample.name; sample_dir.mkdir(exist_ok=True)
            Image.fromarray(sample.image).save(sample_dir / "original.jpg")
            draw_detections(sample.image, pre_predictions, sample_dir / "pre_detections.jpg")
        _write_checkpoint(output_dir, records, index + 1, total)
        print(f"processed {index + 1}/{total} images; records={len(records)}", flush=True)
    if not records: raise RuntimeError("No labelled images were found")
    fields = list(records[0].keys())
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    summary = summarize_records(records)
    summary["dataset"] = {"images": str(images), "labels": str(labels), "metadata_dir": str(dataset.metadata_dir) if dataset.metadata_dir else None, "person_class": person_class, "oracle_importance": oracle_importance, "research_methods": research_methods, "adaptive_tile_size": adaptive_tile_size, "tile_workers": tile_workers, "include_hevc": include_hevc, "uniform_qualities": list(uniform_qualities), "pairing_qualities": list(pairing_qualities), "detection_batch_size": detection_batch_size, "importance_semantics": importance_semantics}
    if research_methods:
        summary["comparison_methods"] = ["uniform", "random_roi", "human_roi", "risk_aware_adaptive"]
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary




