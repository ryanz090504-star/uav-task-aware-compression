"""Sequence uncertainty for pooled AP, Precision and Recall from saved outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_aware_compression.types import BoundingBox, Detection
from task_aware_compression.metrics import evaluate_detections


def matched_score_curve(predictions, truth):
    used = set(); scores, flags = [], []
    for prediction in sorted(predictions, key=lambda det: det.confidence, reverse=True):
        matches = [(prediction.box.iou(det.box), index) for index, det in enumerate(truth) if index not in used]
        overlap, index = max(matches) if matches else (0, -1)
        matched = overlap >= .5
        if matched:
            used.add(index)
        scores.append(prediction.confidence); flags.append(matched)
    return np.asarray(scores), np.asarray(flags, dtype=np.float64)


def bootstrap(experiment, left, right, repeats, seed):
    items = [json.loads(line) for line in (experiment / "detections.jsonl").open(encoding="utf-8")]
    names = sorted({item["sample"] for item in items if item["setting"] == left})
    sequences = sorted({name.split("_")[0] for name in names})
    seq_index = {name: index for index, name in enumerate(sequences)}
    curve_data, counts = {}, {}
    for method in (left, right):
        score_lists, flag_lists, seq_lists = [], [], []
        gt_counts, micro = np.zeros(len(sequences)), np.zeros((len(sequences), 3))
        for item in items:
            if item["setting"] != method:
                continue
            predictions = [Detection(BoundingBox(**det["box"]), det["confidence"]) for det in item["predictions"]]
            truth = [Detection(BoundingBox(**det["box"])) for det in item["ground_truth"]]
            index = seq_index[item["sample"].split("_")[0]]
            scores, flags = matched_score_curve(predictions, truth)
            score_lists.append(scores); flag_lists.append(flags); seq_lists.append(np.full(len(scores), index))
            gt_counts[index] += len(truth)
            threshold = item.get("receiver_threshold", .12)
            metrics = evaluate_detections([det for det in predictions if det.confidence >= threshold], truth, (1, 1))
            micro[index] += [metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]]
        scores, flags, ids = np.concatenate(score_lists), np.concatenate(flag_lists), np.concatenate(seq_lists)
        order = np.argsort(-scores, kind="stable")
        curve_data[method] = (flags[order], ids[order], gt_counts)
        counts[method] = micro
    def metrics(method, weights):
        flags, ids, gt = curve_data[method]
        multiplier = weights[ids]
        tp = np.cumsum(flags * multiplier)
        rank = np.cumsum(multiplier)
        precision = np.divide(tp, rank, out=np.zeros_like(tp), where=rank > 0)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        total_truth = weights @ gt
        ap = float((flags * multiplier * envelope).sum() / max(1, total_truth))
        total_tp, fp, fn = weights @ counts[method]
        return np.array([ap, total_tp / max(1, total_tp + fp), total_tp / max(1, total_tp + fn)])
    observed = metrics(left, np.ones(len(sequences))) - metrics(right, np.ones(len(sequences)))
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, 3))
    for index in range(repeats):
        weights = np.bincount(rng.integers(0, len(sequences), len(sequences)), minlength=len(sequences))
        draws[index] = metrics(left, weights) - metrics(right, weights)
    report = {"left": left, "right": right, "n": len(names), "n_sequences": len(sequences), "repeats": repeats, "seed": seed, "metrics": {name: {"delta": float(observed[i]), "ci95": np.percentile(draws[:, i], [2.5, 97.5]).tolist()} for i, name in enumerate(("dataset_ap50", "micro_precision", "micro_recall"))}}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=Path); p.add_argument("--left", default="learned"); p.add_argument("--right", default="baseline")
    p.add_argument("--repeats", type=int, default=4000); p.add_argument("--seed", type=int, default=731)
    args = p.parse_args()
    if not (args.experiment / "detections.jsonl").is_file():
        p.error("Missing local detections.jsonl. Generate private evaluation outputs using docs/REPRODUCE.md; GT-containing detections are not distributed.")
    report = bootstrap(args.experiment, args.left, args.right, args.repeats, args.seed)
    target = args.experiment / f"bootstrap_{args.left}_minus_{args.right}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
