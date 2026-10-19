"""Reanalyse paired compression records with image-cluster uncertainty.

The legacy ``mAP50`` column is a per-image AP@0.5. It is intentionally
reported under that name here; pooled dataset AP needs raw detections.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _micro(rows: list[dict]) -> dict[str, float]:
    tp = sum(_f(row, "true_positive") for row in rows)
    fp = sum(_f(row, "false_positive") for row in rows)
    fn = sum(_f(row, "false_negative") for row in rows)
    return {
        "targets": int(tp + fn),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "mean_image_ap50": float(np.mean([_f(row, "mAP50") for row in rows])),
    }


def _bootstrap_delta(pairs: list[tuple[dict, dict]], metric: str, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if metric == "micro_recall":
        a_tp = np.array([_f(a, "true_positive") for a, _ in pairs])
        b_tp = np.array([_f(b, "true_positive") for _, b in pairs])
        gt = np.array([_f(a, "true_positive") + _f(a, "false_negative") for a, _ in pairs])
        delta = (a_tp.sum() - b_tp.sum()) / gt.sum()
        draws = []
        for _ in range(repeats):
            indices = rng.integers(0, len(pairs), len(pairs))
            draws.append((a_tp[indices].sum() - b_tp[indices].sum()) / gt[indices].sum())
    else:
        differences = np.array([_f(a, metric) - _f(b, metric) for a, b in pairs])
        delta = float(differences.mean())
        draws = [float(differences[rng.integers(0, len(pairs), len(pairs))].mean()) for _ in range(repeats)]
    return {"delta": float(delta), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def _cluster_bootstrap_delta(pairs: list[tuple[dict, dict]], metric: str, repeats: int, seed: int) -> dict:
    """Resample VisDrone sequence IDs, retaining all frames from each draw."""
    rng = np.random.default_rng(seed)
    group_indices: dict[str, list[int]] = {}
    for index, (adaptive, _) in enumerate(pairs):
        group_indices.setdefault(str(adaptive["sample"]).split("_")[0], []).append(index)
    clusters = list(group_indices.values())
    if len(clusters) < 2:
        return {"n_sequences": len(clusters), "ci95": None}
    a_tp = np.array([_f(a, "true_positive") for a, _ in pairs])
    b_tp = np.array([_f(b, "true_positive") for _, b in pairs])
    gt = np.array([_f(a, "true_positive") + _f(a, "false_negative") for a, _ in pairs])
    differences = np.array([_f(a, metric) - _f(b, metric) for a, b in pairs]) if metric != "micro_recall" else None
    draws = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        if metric == "micro_recall": draws.append((a_tp[indices].sum() - b_tp[indices].sum()) / gt[indices].sum())
        else: draws.append(float(differences[indices].mean()))
    return {"n_sequences": len(clusters), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def analyse(records: list[dict], repeats: int = 2000, seed: int = 17) -> dict:
    originals = [row for row in records if row["mode"] == "original"]
    lookup = {(row["sample"], row["setting"]): row for row in records if row["mode"] == "uniform_paired"}
    names = sorted({row["setting"] for row in records if row["mode"] == "adaptive"})
    result = {
        "metric_note": "micro_recall pools TP and GT across images; mean_image_ap50 is not dataset mAP; bootstrap resamples images",
        "original": _micro(originals) if originals else None,
        "settings": {},
    }
    for name in names:
        adaptive = [row for row in records if row["mode"] == "adaptive" and row["setting"] == name]
        pairs = [(row, lookup[(row["sample"], "paired_" + name)]) for row in adaptive]
        errors = np.array([abs(_f(a, "file_size_bytes") - _f(b, "file_size_bytes")) / _f(a, "file_size_bytes") for a, b in pairs])
        signed_errors = np.array([(_f(b, "file_size_bytes") - _f(a, "file_size_bytes")) / _f(a, "file_size_bytes") for a, b in pairs])
        item = {
            "n": len(pairs),
            "adaptive": _micro([a for a, _ in pairs]),
            "jpeg": _micro([b for _, b in pairs]),
            "size_error": {"median": float(np.median(errors)), "p95": float(np.percentile(errors, 95)), "max": float(errors.max()), "mean_signed_jpeg_minus_adaptive": float(np.mean(signed_errors)), "jpeg_smaller_fraction": float(np.mean(signed_errors < 0)), "within_2pct": int(sum(errors <= .02)), "within_5pct": int(sum(errors <= .05))},
            "all_pairs": {
                "micro_recall": _bootstrap_delta(pairs, "micro_recall", repeats, seed),
                "mean_image_ap50": _bootstrap_delta(pairs, "mAP50", repeats, seed),
                "mean_image_precision": _bootstrap_delta(pairs, "precision", repeats, seed),
                "micro_recall_sequence_bootstrap": _cluster_bootstrap_delta(pairs, "micro_recall", repeats, seed),
            },
        }
        for threshold in (.02, .05):
            filtered = [pair for pair, error in zip(pairs, errors) if error <= threshold]
            item[f"within_{int(threshold * 100)}pct"] = {
                "n": len(filtered),
                "micro_recall": _bootstrap_delta(filtered, "micro_recall", repeats, seed) if filtered else None,
                "micro_recall_sequence_bootstrap": _cluster_bootstrap_delta(filtered, "micro_recall", repeats, seed) if filtered else None,
                "mean_image_ap50": _bootstrap_delta(filtered, "mAP50", repeats, seed) if filtered else None,
            }
        result["settings"][name] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if not args.records.is_file():
        parser.error(f"Missing records CSV: {args.records}. Generate local baseline outputs as described in docs/REPRODUCE.md; raw records are not distributed.")
    with args.records.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    result = analyse(records, args.bootstrap)
    target = args.output or args.records.with_name("strict_evaluation.json")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(target)
    for name, item in result["settings"].items():
        d = item["all_pairs"]["micro_recall"]
        print(f"{name}: n={item['n']} micro recall delta={d['delta']:.4f} CI={d['ci95']} size p95={item['size_error']['p95']:.3f}")


if __name__ == "__main__":
    main()
