"""Evaluate frozen allocation and operating thresholds on a held-out dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path



from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))











def _load_runtime():
    """Load optional model dependencies only after parsing and input checks."""
    global np, torch, _record, compress_adaptive, encode_uniform, compress_uniform, YoloPersonDataset, UltralyticsPersonDetector, TileRateAllocator, build_candidate_bank, EVAL_OFFSETS, select_near_budget, dataset_average_precision, CompressionConfig, choose_paths, load_sample, paired_report, sequence, serialize_detections, write_json, _micro
    import numpy as np
    import torch
    from task_aware_compression.benchmark import _record
    from task_aware_compression.compressor import compress_adaptive, encode_uniform, compress_uniform
    from task_aware_compression.dataset import YoloPersonDataset
    from task_aware_compression.detector import UltralyticsPersonDetector
    from task_aware_compression.learned_allocation import TileRateAllocator, build_candidate_bank, EVAL_OFFSETS, select_near_budget
    from task_aware_compression.metrics import dataset_average_precision
    from task_aware_compression.types import CompressionConfig
    from allocation_utils import choose_paths, load_sample, paired_report, sequence, serialize_detections, write_json
    from strict_evaluation import _micro


def calibrate_from_tune(tune_dir: Path, minimum_precision_delta=-.01):
    items = [json.loads(line) for line in (tune_dir / "detections.jsonl").open(encoding="utf-8")]
    from task_aware_compression.types import BoundingBox, Detection
    from task_aware_compression.metrics import evaluate_detections
    def totals(name, threshold):
        total = np.zeros(3)
        for item in items:
            if item["setting"] != name:
                continue
            pred = [Detection(BoundingBox(**x["box"]), x["confidence"]) for x in item["predictions"] if x["confidence"] >= threshold]
            truth = [Detection(BoundingBox(**x["box"])) for x in item["ground_truth"]]
            metrics = evaluate_detections(pred, truth, (1, 1))
            total += [metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]]
        return {"recall": total[0] / max(1, total[0] + total[2]), "precision": total[0] / max(1, total[0] + total[1])}
    baseline = totals("baseline", .12)
    candidates = []
    for threshold in np.arange(.12, .251, .005):
        metric = totals("learned", float(threshold))
        candidates.append({"threshold": float(threshold), **metric})
    feasible = [c for c in candidates if c["precision"] >= baseline["precision"] + minimum_precision_delta]
    if not feasible:
        raise ValueError("no precision-matched tune operating threshold")
    chosen = max(feasible, key=lambda item: (item["recall"], item["precision"]))
    return {"baseline": baseline, "chosen": chosen, "candidates": candidates, "source": str(tune_dir)}


LEGACY_V2_CHECKPOINT_SHA256 = "e28024b48c26d378f3378d40d24ef266f68a4f61ea37ae452e977b721a90bf45"


def resolve_checkpoint_compression_config(state_config: dict, checkpoint_sha256: str) -> dict:
    """Resolve only the artifact-verified legacy V2 checkpoint's missing mode.

    New checkpoints store the mode explicitly. Unknown legacy checkpoints must
    establish provenance rather than inherit a scientific default silently.
    """
    resolved = dict(state_config)
    if "importance_semantics" not in resolved:
        if checkpoint_sha256 != LEGACY_V2_CHECKPOINT_SHA256:
            raise ValueError("Checkpoint lacks importance_semantics and is not the provenance-verified legacy V2 artifact")
        resolved["importance_semantics"] = "truncated_6sigma"
    if resolved["importance_semantics"] not in {"historical_full_gaussian", "truncated_6sigma"}:
        raise ValueError(f"unknown importance semantics: {resolved['importance_semantics']}")
    return resolved


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != state["split"]["model_sha256"]:
        raise ValueError("detector does not match training checkpoint")
    calibration = calibrate_from_tune(args.tune)
    threshold = calibration["chosen"]["threshold"]
    config = CompressionConfig(**resolve_checkpoint_compression_config(state["compression_config"], hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()))
    allocator = TileRateAllocator(**state["model_config"])
    allocator.load_state_dict(state["state_dict"]); allocator.eval()
    dataset = YoloPersonDataset(args.images, args.labels)
    paths = dataset.paths() if args.all else choose_paths(dataset.paths(), args.count, state["split"]["seed"] + 1)
    eval_sequences = {sequence(p) for p in paths}
    if eval_sequences & (set(state["split"]["train_sequences"]) | set(state["split"]["tune_sequences"])):
        raise ValueError("evaluation sequences overlap allocator training/tuning")
    freeze = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "importance_semantics": config.importance_semantics, "receiver_confidence": threshold, "baseline_confidence": .12, "minimum_ratio": .98, "calibration": calibration, "names": [p.stem for p in paths], "eval_sequences": sorted(eval_sequences), "split": state["split"]}
    write_json(args.output / "frozen_design.json", freeze)
    detector = UltralyticsPersonDetector(str(args.model), confidence=.12, device="cpu", imgsz=1280)
    rows, scored = [], {}
    capture = {}
    def capture_hook(module, inputs, output):
        if isinstance(output, tuple):
            capture["decoded_pre_nms"] = output[0].detach().cpu().numpy().copy()
            capture["input_shape"] = np.array(inputs[0][0].shape[2:]) * 8
    handle = detector.model.model.model[-1].register_forward_hook(capture_hook)
    raw_dir = args.output / "raw_outputs"; raw_dir.mkdir()
    try:
        with (args.output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            for index, path in enumerate(paths, 1):
                sample = load_sample(dataset, path)
                detector.confidence = .12
                start = time.perf_counter(); candidates = detector.predict(sample.image); sender_ms = (time.perf_counter() - start) * 1000
                baseline = compress_adaptive(sample.image, candidates, config)
                bank = build_candidate_bank(sample.image, candidates, config, baseline.file_size, EVAL_OFFSETS)
                if bank.compress(bank.baseline_levels).encoded != baseline.encoded:
                    raise RuntimeError("codec identity mismatch")
                with torch.no_grad():
                    scores = allocator.scores(torch.as_tensor(bank.features), torch.as_tensor(bank.qualities, dtype=torch.float32)).numpy()
                start = time.perf_counter()
                levels = select_near_budget(scores, bank.costs, baseline.file_size - bank.fixed_bytes, minimum_ratio=.98, fallback=bank.baseline_levels)
                projection_ms = (time.perf_counter() - start) * 1000
                learned = bank.compress(levels); learned.encode_ms += projection_ms
                if not .98 <= learned.file_size / baseline.file_size <= 1:
                    raise RuntimeError("exact byte interval violated")
                methods = {"baseline": baseline, "learned": learned}
                if args.jpeg:
                    sizes = {q: len(encode_uniform(sample.image, q)) for q in range(1, 101)}
                    for name, result in list(methods.items()):
                        q = min(sizes, key=lambda q: abs(sizes[q] - result.file_size))
                        methods["jpeg_" + name] = compress_uniform(sample.image, q)
                detector.confidence = .001
                for name, result in methods.items():
                    capture.clear()
                    start = time.perf_counter(); predictions = detector.predict(result.image); receiver_ms = (time.perf_counter() - start) * 1000
                    np.savez_compressed(raw_dir / f"{sample.name}_{name}.npz", **capture, original_shape=np.array(sample.image.shape[:2]))
                    thresholds = {name: threshold if name == "learned" else .12}
                    if name == "learned":
                        thresholds["learned_fixed012"] = .12
                    for setting, conf in thresholds.items():
                        filtered = [det for det in predictions if det.confidence >= conf]
                        row = _record(sample, "method", setting, result, filtered, receiver_ms, candidate_detection_ms=sender_ms if name in {"baseline", "learned"} else 0)
                        row["receiver_threshold"] = conf
                        rows.append(row)
                        scored.setdefault(setting, []).append((predictions, sample.ground_truth))
                        stream.write(json.dumps({"sample": sample.name, "setting": setting, "receiver_threshold": conf, "predictions": serialize_detections(predictions), "ground_truth": serialize_detections(sample.ground_truth)}) + "\n")
                    if index <= 2:
                        folder = args.output / "samples" / sample.name; folder.mkdir(parents=True, exist_ok=True)
                        (folder / (name + (".jpg" if name.startswith("jpeg_") else ".tac"))).write_bytes(result.encoded)
                        Image.fromarray(result.image).save(folder / f"{name}.png")
                with (args.output / "records.partial.csv").open("w", newline="", encoding="utf-8") as csv_stream:
                    writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                print(f"validation {index}/{len(paths)} {sample.name}", flush=True)
    finally:
        handle.remove()
    (args.output / "records.partial.csv").replace(args.output / "records.csv")
    report = {"n": len(paths), "n_sequences": len(eval_sequences), "frozen_design": freeze, "methods": {}, "learned_minus_baseline": paired_report(rows, "learned", "baseline", state["split"]["seed"], 4000), "learned_fixed012_minus_baseline": paired_report(rows, "learned_fixed012", "baseline", state["split"]["seed"], 4000)}
    if args.jpeg:
        report["learned_minus_jpeg"] = paired_report(rows, "learned", "jpeg_learned", state["split"]["seed"], 4000)
    for name in scored:
        values = [r for r in rows if r["setting"] == name]
        report["methods"][name] = {"micro": _micro(values), "dataset_ap50": dataset_average_precision(scored[name]), "mean_bpp": float(np.mean([r["bits_per_pixel"] for r in values])), "mean_end_to_end_ms": float(np.mean([r["end_to_end_ms"] for r in values]))}
    write_json(args.output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("n", "n_sequences", "methods", "learned_minus_baseline")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--tune", type=Path, required=True)
    p.add_argument("--images", type=Path, default=ROOT / "data/VisDrone/images/val"); p.add_argument("--labels", type=Path, default=ROOT / "data/VisDrone/labels/val")
    p.add_argument("--model", type=Path, default=ROOT / "models/visdrone_yolov8n_1280_b18_s17_best.pt")
    p.add_argument("--count", type=int, default=80); p.add_argument("--all", action="store_true"); p.add_argument("--jpeg", action="store_true")
    p.add_argument("--threads", type=int, default=4); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    for label, path in (("allocator", args.checkpoint), ("detector", args.model)):
        if not path.is_file():
            p.error(f"Missing local {label} checkpoint. Models are withheld; see MODEL_PROVENANCE.md.")
    if not (args.tune / "detections.jsonl").is_file():
        p.error("Missing private tune detections.jsonl. Generate local tune outputs; these annotations are not distributed.")
    for directory in (args.images, args.labels):
        if not directory.is_dir():
            p.error(f"Missing dataset directory: {directory}. Follow DATASET.md first.")
    try:
        _load_runtime()
    except ImportError as error:
        p.error(f"Missing learned dependency ({error}). Install: python -m pip install -e '.[learned]'")
    run(args)


if __name__ == "__main__":
    main()
