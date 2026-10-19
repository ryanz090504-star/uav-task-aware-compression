from __future__ import annotations
from typing import Iterable
from .types import Detection

def _match_detections(predictions: list[Detection], ground_truth: list[Detection], iou_threshold: float) -> tuple[int, int, int, list[float]]:
    ordered = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    used: set[int] = set(); true_positive = 0; false_positive = 0; matched_ious: list[float] = []
    for prediction in ordered:
        best_index, best_iou = -1, 0.0
        for index, truth in enumerate(ground_truth):
            if index in used: continue
            overlap = prediction.box.iou(truth.box)
            if overlap > best_iou: best_index, best_iou = index, overlap
        if best_iou >= iou_threshold:
            used.add(best_index); true_positive += 1; matched_ious.append(best_iou)
        else: false_positive += 1
    return true_positive, false_positive, len(ground_truth) - len(used), matched_ious

def average_precision(predictions: list[Detection], ground_truth: list[Detection], iou_threshold: float = 0.5) -> float:
    if not ground_truth: return 1.0 if not predictions else 0.0
    ordered = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    used: set[int] = set(); precisions: list[float] = []; recalls: list[float] = []; true_positive = 0
    for rank, prediction in enumerate(ordered, start=1):
        best_index, best_iou = -1, 0.0
        for index, truth in enumerate(ground_truth):
            if index in used: continue
            overlap = prediction.box.iou(truth.box)
            if overlap > best_iou: best_index, best_iou = index, overlap
        if best_iou >= iou_threshold:
            used.add(best_index); true_positive += 1
        precisions.append(true_positive / rank); recalls.append(true_positive / len(ground_truth))
    if not recalls: return 0.0
    recall_points = [0.0] + recalls + [1.0]; precision_points = [0.0] + precisions + [0.0]
    for index in range(len(precision_points) - 2, -1, -1): precision_points[index] = max(precision_points[index], precision_points[index + 1])
    area = sum((recall_points[i] - recall_points[i - 1]) * precision_points[i] for i in range(1, len(recall_points)))
    return float(max(0.0, min(1.0, area)))

def evaluate_detections(predictions: Iterable[Detection], ground_truth: Iterable[Detection], image_shape: tuple[int, int], iou_threshold: float = 0.5, small_area_ratio: float = 0.018) -> dict[str, float]:
    predictions, ground_truth = list(predictions), list(ground_truth)
    tp, fp, fn, matched_ious = _match_detections(predictions, ground_truth, iou_threshold)
    truth_count = len(ground_truth)
    recall = tp / truth_count if truth_count else 1.0
    precision = tp / (tp + fp) if tp + fp else (1.0 if not truth_count else 0.0)
    image_area = max(1, image_shape[0] * image_shape[1])
    small_truth = [item for item in ground_truth if item.box.area / image_area <= small_area_ratio]
    small_predictions = [item for item in predictions if item.box.area / image_area <= small_area_ratio * 1.8]
    small_tp, _, _, _ = _match_detections(small_predictions, small_truth, iou_threshold)
    small_recall = small_tp / len(small_truth) if small_truth else 1.0
    small_ap = average_precision(small_predictions, small_truth, iou_threshold)
    return {"recall": float(recall), "precision": float(precision), "mAP50": average_precision(predictions, ground_truth, iou_threshold), "ap_small": float(small_ap), "small_recall": float(small_recall), "miss_rate": float(1.0 - recall), "true_positive": float(tp), "false_positive": float(fp), "false_negative": float(fn), "mean_matched_iou": float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0}


def dataset_average_precision(samples: Iterable[tuple[list[Detection], list[Detection]]], iou_threshold: float = 0.5) -> float:
    """Pool scored detections across images before AP integration.

    Predictions must be collected at a sufficiently low confidence threshold;
    this function cannot recover detections already filtered by a model.
    """
    samples = list(samples)
    total_truth = sum(len(truth) for _, truth in samples)
    if total_truth == 0:
        return 1.0 if not any(predictions for predictions, _ in samples) else 0.0
    scored = [(prediction.confidence, image_index, prediction.box) for image_index, (predictions, _) in enumerate(samples) for prediction in predictions]
    scored.sort(key=lambda item: item[0], reverse=True)
    used = [set() for _ in samples]
    tp = 0
    precisions = [0.0]
    recalls = [0.0]
    for rank, (_, image_index, box) in enumerate(scored, start=1):
        truth = samples[image_index][1]
        matches = [(box.iou(item.box), index) for index, item in enumerate(truth) if index not in used[image_index]]
        if matches:
            overlap, best = max(matches)
            if overlap >= iou_threshold:
                used[image_index].add(best)
                tp += 1
        precisions.append(tp / rank)
        recalls.append(tp / total_truth)
    precisions.append(0.0)
    recalls.append(1.0)
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return float(sum((recalls[index] - recalls[index - 1]) * precisions[index] for index in range(1, len(recalls))))
