"""Native-resolution allocation with class distillation and separate tune data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace








ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))














def _load_runtime():
    """Load optional model dependencies only after parsing and input checks."""
    global cv2, np, torch, F, DEFAULT_CFG_DICT, v8DetectionLoss, _decode_jpeg, compress_adaptive, compress_uniform, YoloPersonDataset, UltralyticsPersonDetector, FEATURE_NAMES, TRAIN_OFFSETS, EVAL_OFFSETS, TileRateAllocator, budgeted_softmax, balanced_softmax, build_candidate_bank, select_near_budget, CompressionConfig, dataset_average_precision, evaluate_detections, choose_paths, load_sample, paired_report, rgb_tensor, serialize_detections, sequence, write_json
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.loss import v8DetectionLoss
    from task_aware_compression.compressor import _decode_jpeg, compress_adaptive, compress_uniform
    from task_aware_compression.dataset import YoloPersonDataset
    from task_aware_compression.detector import UltralyticsPersonDetector
    from task_aware_compression.learned_allocation import (
        FEATURE_NAMES, TRAIN_OFFSETS, EVAL_OFFSETS, TileRateAllocator,
        budgeted_softmax, balanced_softmax, build_candidate_bank, select_near_budget,
    )
    from task_aware_compression.types import CompressionConfig
    from task_aware_compression.metrics import dataset_average_precision, evaluate_detections
    from allocation_utils import (
        choose_paths, load_sample, paired_report, rgb_tensor, serialize_detections, sequence, write_json,
    )


def make_splits(train_data, eval_data, seed, train_count, tune_count, eval_count):
    excluded = {sequence(path) for path in eval_data.paths()}
    pool = [path for path in train_data.paths() if sequence(path) not in excluded]
    tune = choose_paths(pool, tune_count, seed + 700)
    tune_sequences = {sequence(path) for path in tune}
    train = choose_paths([path for path in pool if sequence(path) not in tune_sequences], train_count, seed)
    validation = choose_paths(eval_data.paths(), eval_count, seed + 1)
    return train, tune, validation


def detector_canvas(image, size):
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    # Match Ultralytics' rectangular letterbox, including exact OpenCV resize.
    dw, dh = (size - resized_width) % 32, (size - resized_height) % 32
    left, top = int(round(dw / 2 - .1)), int(round(dh / 2 - .1))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, int(round(dh / 2 + .1)), left, int(round(dw / 2 + .1)), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, (resized_width, resized_height, left, top)


def tensor_targets(sample, geometry, shape, device):
    resized_width, resized_height, left, top = geometry
    height, width = sample.image.shape[:2]
    out_height, out_width = shape
    boxes = []
    for detection in sample.ground_truth:
        box = detection.box.clip(width, height)
        if box.area:
            boxes.append([
                (box.center[0] * resized_width / width + left) / out_width,
                (box.center[1] * resized_height / height + top) / out_height,
                box.width * resized_width / width / out_width,
                box.height * resized_height / height / out_height,
            ])
    return {"batch_idx": torch.zeros(len(boxes), device=device), "cls": torch.zeros(len(boxes), 1, device=device), "bboxes": torch.tensor(boxes, device=device, dtype=torch.float32).reshape(-1, 4)}


def distillation(raw, reference):
    logits = raw[1]["scores"]
    probability = logits.sigmoid()
    target = reference
    weights = (.1 + 20 * target).detach()
    return ((probability - target).square() * weights).sum() / (target.sum().clamp_min(1))


def background_response_loss(raw, boxes, canvas_shape):
    decoded = raw[0].detach()
    probability = raw[1]["scores"].sigmoid()[0, 0]
    center_x, center_y = decoded[0, 0], decoded[0, 1]
    mask = torch.ones_like(probability, dtype=torch.bool)
    height, width = canvas_shape
    for cx, cy, bw, bh in boxes:
        inside = (center_x >= (cx - bw * .75) * width - 12) & (center_x <= (cx + bw * .75) * width + 12)
        inside &= (center_y >= (cy - bh * .75) * height - 12) & (center_y <= (cy + bh * .75) * height + 12)
        mask &= ~inside
    return (probability[mask].square()).sum() / max(1, len(boxes))


def cache_training(args, detector, dataset, paths, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    teacher = detector.model.model.eval()
    features_all, audit = [], []
    for index, path in enumerate(paths, 1):
        sample = load_sample(dataset, path)
        detector.confidence = args.confidence
        candidates = detector.predict(sample.image)
        bank = build_candidate_bank(sample.image, candidates, config, offsets=TRAIN_OFFSETS)
        original, geometry = detector_canvas(sample.image, args.size)
        baseline_image = bank.compress(bank.baseline_levels).image
        # Candidates remain uint8 on disk. One frame is materialized at a time.
        tensors = {}
        height, width = sample.image.shape[:2]
        resized_width, resized_height, left, top = geometry
        rectangles = []
        for tile, (x, y, w, h) in enumerate(bank.rectangles):
            x1, x2 = round(x * resized_width / width) + left, round((x + w) * resized_width / width) + left
            y1, y2 = round(y * resized_height / height) + top, round((y + h) * resized_height / height) + top
            tensors[f"tile_{tile}"] = np.stack([
                cv2.resize(_decode_jpeg(payload, w, h), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                for payload in bank.payloads[tile]
            ])
            rectangles.append((x1, y1, x2 - x1, y2 - y1))
        # Identity forward uses exactly the whole-image receiver resize.
        baseline_canvas, _ = detector_canvas(baseline_image, args.size)
        tiled_identity = baseline_canvas.copy()
        for tile, (x, y, w, h) in enumerate(rectangles):
            tiled_identity[y:y + h, x:x + w] = tensors[f"tile_{tile}"][bank.baseline_levels[tile]]
        with torch.no_grad():
            reference = teacher(rgb_tensor(original, device)[None])[1]["scores"].sigmoid().cpu().numpy()
        targets = tensor_targets(sample, geometry, original.shape[:2], device)
        np.savez_compressed(
            directory / f"{sample.name}.npz", features=bank.features, qualities=bank.qualities,
            costs=bank.costs, fixed_bytes=bank.fixed_bytes, target_bytes=bank.target_bytes,
            baseline_levels=bank.baseline_levels, original=original, baseline=baseline_canvas,
            tile_identity=tiled_identity, rectangles=np.asarray(rectangles), reference=reference,
            targets=targets["bboxes"].cpu().numpy(), **tensors,
        )
        features_all.append(bank.features)
        audit.append({"sample": sample.name, "identity_resize_mean_abs_error": float(np.abs(tiled_identity.astype(float) - baseline_canvas).mean()), "tiles": len(rectangles)})
        print(f"cache {index}/{len(paths)} {sample.name}", flush=True)
    write_json(directory / "audit.json", {"frames": audit})
    return np.concatenate(features_all)


def reconstruct_cached(data, probabilities, levels, device):
    baseline = rgb_tensor(data["baseline"], device)[None]
    identity = rgb_tensor(data["tile_identity"], device)[None]
    canvas = baseline.clone()
    hard = F.one_hot(torch.as_tensor(levels, device=device), probabilities.shape[1]).float()
    weights = probabilities + (hard - probabilities).detach()
    # A fixed identity correction cancels the per-tile vs whole-image resize
    # difference at initialization without changing the derivative.
    for tile, (x, y, w, h) in enumerate(data["rectangles"]):
        values = torch.from_numpy(data[f"tile_{tile}"][..., ::-1].copy()).permute(0, 3, 1, 2).float().to(device) / 255
        mixed = (weights[tile, :, None, None, None] * values).sum(0)
        canvas[:, :, y:y + h, x:x + w] = mixed + baseline[:, :, y:y + h, x:x + w] - identity[:, :, y:y + h, x:x + w]
    return canvas.clamp(0, 1)


def quick_evaluate(args, detector, allocator, dataset, paths, config, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, scored = [], {"baseline": [], "learned": []}
    device = next(allocator.parameters()).device
    with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
        for index, path in enumerate(paths, 1):
            sample = load_sample(dataset, path)
            detector.confidence = args.confidence
            candidates = detector.predict(sample.image)
            baseline = compress_adaptive(sample.image, candidates, config)
            bank = build_candidate_bank(sample.image, candidates, config, baseline.file_size, EVAL_OFFSETS)
            if bank.compress(bank.baseline_levels).encoded != baseline.encoded:
                raise RuntimeError("identity codec mismatch")
            with torch.no_grad():
                scores = allocator.scores(torch.as_tensor(bank.features, device=device), torch.as_tensor(bank.qualities, dtype=torch.float32, device=device)).cpu().numpy()
            levels = select_near_budget(scores, bank.costs, bank.target_bytes - bank.fixed_bytes, minimum_ratio=args.minimum_ratio, fallback=bank.baseline_levels)
            learned = bank.compress(levels)
            if not args.minimum_ratio <= learned.file_size / baseline.file_size <= 1:
                raise RuntimeError("allocation byte interval violated")
            detector.confidence = .001
            for name, result in (("baseline", baseline), ("learned", learned)):
                predictions = detector.predict(result.image)
                scored[name].append((predictions, sample.ground_truth))
                metrics = evaluate_detections([det for det in predictions if det.confidence >= args.confidence], sample.ground_truth, sample.image.shape[:2])
                rows.append({"sample": sample.name, "setting": name, "file_size_bytes": result.file_size, "bits_per_pixel": result.bits_per_pixel, **metrics})
                stream.write(json.dumps({"sample": sample.name, "setting": name, "predictions": serialize_detections(predictions), "ground_truth": serialize_detections(sample.ground_truth)}) + "\n")
            print(f"eval {index}/{len(paths)} {sample.name}", flush=True)
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    from strict_evaluation import _micro
    report = {"n": len(paths), "n_sequences": len({sequence(p) for p in paths}), "methods": {name: {"micro": _micro([row for row in rows if row["setting"] == name]), "dataset_ap50": dataset_average_precision(scored[name])} for name in scored}, "learned_minus_baseline": paired_report(rows, "learned", "baseline", args.seed, 2000)}
    write_json(output / "report.json", report)
    return report


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    train_data = YoloPersonDataset(args.train_images, args.train_labels)
    eval_data = YoloPersonDataset(args.eval_images, args.eval_labels)
    train, tune, validation = make_splits(train_data, eval_data, args.seed, args.train_count, args.tune_count, args.eval_count)
    split = {"train_names": [p.stem for p in train], "tune_names": [p.stem for p in tune], "validation_names": [p.stem for p in validation], "train_sequences": sorted({sequence(p) for p in train}), "tune_sequences": sorted({sequence(p) for p in tune}), "validation_sequences": sorted({sequence(p) for p in validation}), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(), "seed": args.seed}
    write_json(args.output / "split.json", split)
    write_json(args.output / "config.json", {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()})
    config = CompressionConfig(tile_size=256, background_quality=8, person_quality=82, importance_semantics="truncated_6sigma")
    detector = UltralyticsPersonDetector(str(args.model), confidence=args.confidence, device=args.device, imgsz=args.size)
    cache_dir = args.cache if args.cache else args.output / "cache"
    if args.cache:
        feature_chunks = []
        for path in train:
            with np.load(cache_dir / f"{path.stem}.npz") as cached:
                feature_chunks.append(cached["features"])
        features = np.concatenate(feature_chunks)
    else:
        features = cache_training(args, detector, train_data, train, config, cache_dir)
    allocator = TileRateAllocator(hidden=32).to(device)
    allocator.feature_mean.copy_(torch.as_tensor(features.mean(0), device=device))
    allocator.feature_scale.copy_(torch.as_tensor(np.maximum(features.std(0), .05), device=device))
    teacher = detector.model.model.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # The predictor created cached anchors under inference_mode. Regenerate
    # them in ordinary autograd mode before differentiating through the input.
    teacher.model[-1].shape = None
    original_args = teacher.args
    teacher.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, **original_args})
    criterion = v8DetectionLoss(teacher)
    if args.initial:
        initial = torch.load(args.initial, map_location=device, weights_only=True)
        if initial["split"] != split:
            raise ValueError("initial checkpoint uses different experiment splits")
        allocator.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.Adam(allocator.parameters(), lr=args.lr)
    logs = []
    rng = np.random.default_rng(args.seed)
    try:
        for epoch in range(args.epochs):
            values = []
            for index in rng.permutation(len(train)):
                with np.load(cache_dir / f"{train[index].stem}.npz") as data:
                    features = torch.as_tensor(data["features"], device=device)
                    qualities = torch.as_tensor(data["qualities"], dtype=torch.float32, device=device)
                    costs = torch.as_tensor(data["costs"], dtype=torch.float32, device=device)
                    budget = int(data["target_bytes"] - data["fixed_bytes"])
                    scores = allocator.scores(features, qualities)
                    probabilities = balanced_softmax(scores, costs, budget, args.temperature) if args.balanced else budgeted_softmax(scores, costs, budget, args.temperature)
                    levels = select_near_budget(scores.detach().cpu().numpy(), data["costs"], budget, minimum_ratio=args.minimum_ratio, fallback=data["baseline_levels"])
                    reconstruction = reconstruct_cached(data, probabilities, levels, device)
                    targets = torch.as_tensor(data["targets"], device=device)
                    batch = {"batch_idx": torch.zeros(len(targets), device=device), "cls": torch.zeros(len(targets), 1, device=device), "bboxes": targets}
                    raw = teacher(reconstruction)
                    detection_loss, _ = criterion(raw, batch)
                    task = detection_loss.sum()
                    kd = distillation(raw, torch.as_tensor(data["reference"], device=device))
                    background = background_response_loss(raw, targets, reconstruction.shape[-2:])
                    residual = (allocator(features) / allocator.max_shift).square().mean()
                    rate = (probabilities * costs).sum() / budget
                    loss = task + args.distill_weight * kd + args.background_weight * background + args.identity_weight * residual + args.rate_weight * (1 - rate).square()
                    optimizer.zero_grad(); loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(allocator.parameters(), 5)
                    if not torch.isfinite(grad) or not torch.isfinite(loss):
                        raise RuntimeError("nonfinite loss or gradient")
                    optimizer.step()
                    values.append({"loss": float(loss.detach()), "task": float(task.detach()), "distillation": float(kd.detach()), "background": float(background.detach()), "rate": float(rate.detach()), "mean_abs_shift": float(allocator(features).detach().abs().mean())})
            row = {"epoch": epoch + 1, **{key: float(np.mean([v[key] for v in values])) for key in values[0]}}
            logs.append(row); print(json.dumps(row), flush=True)
            torch.save({"state_dict": allocator.state_dict(), "model_config": {"hidden": 32, "max_shift": 12.0}, "compression_config": asdict(config), "split": split, "logs": logs, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, args.output / "rate_allocator.pt")
            write_json(args.output / "training.json", {"logs": logs})
    finally:
        teacher.args = original_args
    allocator.eval()
    report = quick_evaluate(args, detector, allocator, train_data, tune, config, args.output / "tune")
    print(json.dumps(report, indent=2), flush=True)
    if args.validate:
        report = quick_evaluate(args, detector, allocator, eval_data, validation, config, args.output / "validation")
        print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-images", type=Path, default=ROOT / "data/VisDrone/VisDrone2019-DET-train/images")
    p.add_argument("--train-labels", type=Path, default=ROOT / "data/VisDrone/VisDrone2019-DET-train/labels")
    p.add_argument("--eval-images", type=Path, default=ROOT / "data/VisDrone/images/val")
    p.add_argument("--eval-labels", type=Path, default=ROOT / "data/VisDrone/labels/val")
    p.add_argument("--model", type=Path, default=ROOT / "models/visdrone_yolov8n_1280_b18_s17_best.pt")
    p.add_argument("--train-count", type=int, default=32); p.add_argument("--tune-count", type=int, default=24); p.add_argument("--eval-count", type=int, default=80)
    p.add_argument("--size", type=int, default=1280); p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=.001); p.add_argument("--temperature", type=float, default=.5)
    p.add_argument("--distill-weight", type=float, default=5.0); p.add_argument("--identity-weight", type=float, default=.02); p.add_argument("--rate-weight", type=float, default=1)
    p.add_argument("--minimum-ratio", type=float, default=.98); p.add_argument("--confidence", type=float, default=.12)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=731)
    p.add_argument("--validate", action="store_true"); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--balanced", action="store_true"); p.add_argument("--cache", type=Path)
    p.add_argument("--initial", type=Path); p.add_argument("--background-weight", type=float, default=0)
    args = p.parse_args()
    if not args.model.is_file():
        p.error("Missing local detector checkpoint. No weights are distributed; see MODEL_PROVENANCE.md.")
    for directory in (args.train_images, args.train_labels, args.eval_images, args.eval_labels):
        if not directory.is_dir():
            p.error(f"Missing dataset directory: {directory}. Follow DATASET.md first.")
    if args.initial is not None and not args.initial.is_file():
        p.error("The optional initial allocator checkpoint must be a local file.")
    if args.cache is not None and not args.cache.is_dir():
        p.error("The optional training cache must be an existing local directory.")
    try:
        _load_runtime()
    except ImportError as error:
        p.error(f"Missing learned dependency ({error}). Install: python -m pip install -e '.[learned]'")
    run(args)


if __name__ == "__main__":
    main()
