"""Learn a residual tile-quality policy while retaining the TAC/JPEG codec.

PyTorch is optional for the core package and required only by this module.
The policy sees RGB statistics and sender detections, never receiver labels.
"""
from __future__ import annotations

import json
import math
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .compressor import (
    CompressionResult, _MAGIC, _as_rgb, _encode_jpeg,
    _quality_for_importance, decode_adaptive,
)
from .importance import build_importance_map, tile_importance_map
from .types import CompressionConfig, Detection


FEATURE_NAMES = (
    "importance_mean", "importance_p95", "importance_max", "protected_fraction",
    "red_mean", "green_mean", "blue_mean", "gray_std", "edge_x", "edge_y",
    "candidate_density", "candidate_confidence", "smallest_candidate_fraction",
    "baseline_quality", "tile_area_fraction", "target_bpp",
)
TRAIN_OFFSETS = (-12, -6, 0, 6, 12)
EVAL_OFFSETS = tuple(range(-12, 13))


class TileRateAllocator(nn.Module):
    def __init__(self, hidden: int = 32, max_shift: float = 12.0):
        super().__init__()
        if hidden < 1 or not math.isfinite(max_shift) or max_shift <= 0:
            raise ValueError("hidden and max_shift must be positive")
        self.hidden = hidden
        self.max_shift = max_shift
        self.register_buffer("feature_mean", torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer("feature_scale", torch.ones(len(FEATURE_NAMES)))
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale.clamp_min(0.05)
        return self.max_shift * torch.tanh(self.net(normalized).squeeze(-1))

    def scores(self, features: torch.Tensor, qualities: torch.Tensor) -> torch.Tensor:
        center = features[:, FEATURE_NAMES.index("baseline_quality")] * 95 + self(features)
        return -((qualities - center[:, None]) / (self.max_shift / 2)).square()


def select_under_budget(scores: np.ndarray, costs: np.ndarray, budget: int) -> np.ndarray:
    """Project policy preferences onto an exact, separable encoded-byte budget."""
    scores = np.asarray(scores, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.int64)
    if scores.ndim != 2 or scores.shape != costs.shape or not scores.size:
        raise ValueError("scores and costs must have the same nonempty NxK shape")
    if not np.isfinite(scores).all() or np.any(costs <= 0):
        raise ValueError("scores must be finite and costs must be positive")
    if int(costs.min(axis=1).sum()) > budget:
        raise ValueError("byte budget is below the smallest available TAC encoding")
    rows = np.arange(len(costs))
    levels = scores.argmax(axis=1)
    if int(costs[rows, levels].sum()) <= budget:
        return levels

    normalized = costs / max(1, budget)
    low, high = 0.0, 1.0
    while True:
        feasible = (scores - high * normalized).argmax(axis=1)
        if int(costs[rows, feasible].sum()) <= budget:
            break
        high *= 2
        if not math.isfinite(high):
            raise RuntimeError("could not find a feasible allocation multiplier")
    for _ in range(48):
        middle = (low + high) / 2
        candidate = (scores - middle * normalized).argmax(axis=1)
        if int(costs[rows, candidate].sum()) <= budget:
            high, feasible = middle, candidate
        else:
            low = middle
    levels = feasible.copy()
    spent = int(costs[rows, levels].sum())
    # Recover useful upgrades skipped by the discrete dual solution.
    while True:
        extra = costs - costs[rows, levels, None]
        gain = scores - scores[rows, levels, None]
        allowed = (gain > 1e-12) & (spent + extra <= budget)
        if not allowed.any():
            break
        priority = np.where(allowed, gain / np.maximum(1, extra), -np.inf)
        row, level = np.unravel_index(priority.argmax(), priority.shape)
        spent += int(extra[row, level])
        levels[row] = level
    return levels


def select_near_budget(
    scores: np.ndarray, costs: np.ndarray, budget: int, *, minimum_ratio: float = .98,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Keep a hard ceiling and fill a lower bound using least costly score losses.

    A supplied feasible baseline is used if discrete candidate gaps prevent
    reaching the lower bound. Only codec bytes and policy scores are inspected.
    """
    if not 0 < minimum_ratio <= 1:
        raise ValueError("minimum_ratio must be in (0, 1]")
    scores = np.asarray(scores, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.int64)
    levels = select_under_budget(scores, costs, budget)
    rows = np.arange(len(costs))
    spent = int(costs[rows, levels].sum())
    lower = math.ceil(minimum_ratio * budget)
    while spent < lower:
        extra = costs - costs[rows, levels, None]
        loss = scores[rows, levels, None] - scores
        valid = (extra > 0) & (extra <= budget - spent)
        if not valid.any():
            break
        # Small steps avoid spending the whole remainder on one outlying tile.
        priorities = np.where(valid, loss / np.maximum(extra, 1), np.inf)
        row, level = np.unravel_index(priorities.argmin(), priorities.shape)
        spent += int(extra[row, level])
        levels[row] = level
    if spent < lower and fallback is not None:
        fallback = np.asarray(fallback, dtype=np.int64)
        if fallback.shape != levels.shape or np.any(fallback < 0) or np.any(fallback >= scores.shape[1]):
            raise ValueError("invalid fallback allocation")
        fallback_spent = int(costs[rows, fallback].sum())
        if not lower <= fallback_spent <= budget:
            raise ValueError("fallback allocation does not meet the byte interval")
        levels = fallback.copy()
    return levels


def budgeted_softmax(
    scores: torch.Tensor, costs: torch.Tensor, budget: int, temperature: float = 0.5,
) -> torch.Tensor:
    """Differentiable relaxation using measured JPEG and header byte costs."""
    if temperature <= 0 or budget <= 0:
        raise ValueError("temperature and budget must be positive")
    if scores.ndim != 2 or scores.shape != costs.shape:
        raise ValueError("scores and costs must have the same NxK shape")
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(costs).all()) or bool((costs <= 0).any()):
        raise ValueError("scores and costs must be finite; costs must be positive")
    if float(costs.min(dim=1).values.sum().detach()) > budget:
        raise ValueError("infeasible relaxed byte budget")
    normalized = costs / budget
    probabilities = torch.softmax(scores / temperature, dim=1)
    expected = (probabilities * normalized).sum()
    if float(expected.detach()) <= 1:
        return probabilities
    with torch.no_grad():
        low, high = 0.0, 1.0
        while float((torch.softmax((scores - high * normalized) / temperature, 1) * normalized).sum()) > 1:
            high *= 2
            if high > 1e12:
                raise ValueError("relaxed budget has no finite-temperature interior")
        for _ in range(40):
            middle = (low + high) / 2
            trial = torch.softmax((scores - middle * normalized) / temperature, 1)
            if float((trial * normalized).sum()) > 1:
                low = middle
            else:
                high = middle
    probabilities = torch.softmax((scores - high * normalized) / temperature, 1)
    expected = (probabilities * normalized).sum()
    means = (probabilities * normalized).sum(dim=1, keepdim=True)
    variance = (probabilities * (normalized - means).square()).sum()
    # An implicit root derivative keeps gradients tangent to the active budget.
    multiplier = high + (expected - 1) / (variance / temperature).detach().clamp_min(1e-12)
    return torch.softmax((scores - multiplier * normalized) / temperature, 1)


def balanced_softmax(scores, costs, budget: int, temperature: float = .5, ratio: float = .99):
    """Project relaxed probabilities onto a two-sided rate equality."""
    if temperature <= 0 or budget <= 0 or not 0 < ratio <= 1:
        raise ValueError("invalid temperature, budget or target ratio")
    normalized = costs / budget
    if float(normalized.min(1).values.sum().detach()) >= ratio or float(normalized.max(1).values.sum().detach()) <= ratio:
        return budgeted_softmax(scores, costs, budget, temperature)
    with torch.no_grad():
        low, high = -1., 1.
        def rate_at(multiplier):
            return float((torch.softmax((scores - multiplier * normalized) / temperature, 1) * normalized).sum())
        while rate_at(low) < ratio:
            low *= 2
        while rate_at(high) > ratio:
            high *= 2
        for _ in range(45):
            mid = (low + high) / 2
            if rate_at(mid) > ratio:
                low = mid
            else:
                high = mid
    probabilities = torch.softmax((scores - high * normalized) / temperature, 1)
    expected = (probabilities * normalized).sum()
    means = (probabilities * normalized).sum(1, keepdim=True)
    variance = (probabilities * (normalized - means).square()).sum()
    multiplier = high + (expected - ratio) / (variance / temperature).detach().clamp_min(1e-12)
    return torch.softmax((scores - multiplier * normalized) / temperature, 1)


@dataclass
class TileCandidateBank:
    image: np.ndarray
    importance: np.ndarray
    config: CompressionConfig
    rectangles: list[tuple[int, int, int, int]]
    qualities: np.ndarray
    costs: np.ndarray
    payloads: list[list[bytes]]
    baseline_levels: np.ndarray
    features: np.ndarray
    fixed_bytes: int
    target_bytes: int
    build_ms: float

    def header(self, records: list[dict[str, int]]) -> dict[str, Any]:
        height, width = self.image.shape[:2]
        return {
            "version": 2, "width": width, "height": height,
            "tile_size": self.config.tile_size, "tile_workers": self.config.tile_workers,
            "tiles": records,
        }

    def compress(self, levels: np.ndarray, mode: str = "learned_allocation") -> CompressionResult:
        start = time.perf_counter()
        levels = np.asarray(levels, dtype=np.int64)
        if levels.shape != (len(self.rectangles),) or np.any(levels < 0) or np.any(levels >= self.qualities.shape[1]):
            raise ValueError("invalid tile candidate indices")
        records, selected_payloads = [], []
        for index, (x, y, width, height) in enumerate(self.rectangles):
            level = levels[index]
            payload = self.payloads[index][level]
            records.append({"x": x, "y": y, "w": width, "h": height, "q": int(self.qualities[index, level]), "n": len(payload)})
            selected_payloads.append(payload)
        header = self.header(records)
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded = _MAGIC + struct.pack(">I", len(blob)) + blob + b"".join(selected_payloads)
        expected_size = self.fixed_bytes + int(self.costs[np.arange(len(levels)), levels].sum())
        if len(encoded) != expected_size:
            raise RuntimeError("TAC byte accounting disagrees with serialized length")
        encode_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        decoded, _ = decode_adaptive(encoded)
        decode_ms = (time.perf_counter() - start) * 1000
        pixels = self.image.shape[0] * self.image.shape[1]
        metadata = {**header, "target_bytes": self.target_bytes, "budget_feasible": len(encoded) <= self.target_bytes}
        return CompressionResult(
            decoded, encoded, mode, len(encoded), self.image.nbytes, len(encoded) * 8 / pixels,
            self.image.nbytes / len(encoded), self.build_ms + encode_ms, decode_ms,
            [record["q"] for record in records], self.importance, metadata,
        )


def build_candidate_bank(
    image: np.ndarray, detections: list[Detection], config: CompressionConfig,
    target_bytes: int | None = None, offsets: tuple[int, ...] = TRAIN_OFFSETS,
) -> TileCandidateBank:
    if config.tile_offset_x or config.tile_offset_y:
        raise ValueError("learned allocation currently requires an unshifted tile grid")
    if not offsets or 0 not in offsets:
        raise ValueError("candidate offsets must include the baseline (zero)")
    if target_bytes is not None and target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    start = time.perf_counter()
    image = _as_rgb(image)
    height, width = image.shape[:2]
    importance = build_importance_map(image, detections, config)
    means = tile_importance_map(importance, config.tile_size, config.tile_pooling)
    rectangles, payloads, qualities, costs, features, baseline_levels = [], [], [], [], [], []
    for row in range(means.shape[0]):
        for col in range(means.shape[1]):
            x, y = col * config.tile_size, row * config.tile_size
            tile = image[y:min(height, y + config.tile_size), x:min(width, x + config.tile_size)]
            h, w = tile.shape[:2]
            region = importance[y:y + h, x:x + w]
            base_q = _quality_for_importance(float(means[row, col]), config)
            qs = [int(np.clip(base_q + offset, 1, 95)) for offset in offsets]
            cache = {q: _encode_jpeg(tile, q) for q in set(qs)}
            row_payloads = [cache[q] for q in qs]
            row_costs = []
            for q, payload in zip(qs, row_payloads):
                record = {"x": x, "y": y, "w": w, "h": h, "q": q, "n": len(payload)}
                row_costs.append(len(payload) + len(json.dumps(record, separators=(",", ":")).encode("utf-8")))
            rgb = tile.astype(np.float32) / 255
            gray = rgb.mean(axis=2)
            candidates = [det for det in detections if det.box.x1 < x + w and det.box.x2 > x and det.box.y1 < y + h and det.box.y2 > y]
            features.append([
                float(region.mean()), float(np.percentile(region, 95)), float(region.max()), float((region >= .5).mean()),
                *rgb.mean(axis=(0, 1)).tolist(), float(gray.std()),
                float(np.abs(np.diff(gray, axis=1)).mean()) if w > 1 else 0.0,
                float(np.abs(np.diff(gray, axis=0)).mean()) if h > 1 else 0.0,
                math.log1p(len(candidates)) / math.log(65),
                float(np.mean([det.confidence for det in candidates])) if candidates else 0.0,
                min((det.box.area / (height * width) for det in candidates), default=0.0),
                base_q / 95, w * h / config.tile_size ** 2, 0.0,
            ])
            rectangles.append((x, y, w, h))
            qualities.append(qs)
            costs.append(row_costs)
            payloads.append(row_payloads)
            baseline_levels.append(qs.index(base_q))
    bank = TileCandidateBank(
        image, importance, config, rectangles, np.asarray(qualities, dtype=np.int64),
        np.asarray(costs, dtype=np.int64), payloads, np.asarray(baseline_levels, dtype=np.int64),
        np.asarray(features, dtype=np.float32), 0, 0, 0.0,
    )
    bank.fixed_bytes = 8 + len(json.dumps(bank.header([]), separators=(",", ":")).encode("utf-8")) + len(rectangles) - 1
    baseline_size = bank.fixed_bytes + int(bank.costs[np.arange(len(rectangles)), bank.baseline_levels].sum())
    bank.target_bytes = baseline_size if target_bytes is None else target_bytes
    bank.features[:, -1] = bank.target_bytes * 8 / (height * width) / 4
    bank.build_ms = (time.perf_counter() - start) * 1000
    return bank


def compress_learned(
    image: np.ndarray, detections: list[Detection], allocator: TileRateAllocator,
    config: CompressionConfig, target_bytes: int | None = None,
    offsets: tuple[int, ...] = EVAL_OFFSETS,
) -> CompressionResult:
    bank = build_candidate_bank(image, detections, config, target_bytes, offsets)
    start = time.perf_counter()
    device = next(allocator.parameters()).device
    with torch.no_grad():
        features = torch.as_tensor(bank.features, device=device)
        qualities = torch.as_tensor(bank.qualities, dtype=torch.float32, device=device)
        scores = allocator.scores(features, qualities).cpu().numpy()
    levels = select_under_budget(scores, bank.costs, bank.target_bytes - bank.fixed_bytes)
    allocation_ms = (time.perf_counter() - start) * 1000
    result = bank.compress(levels)
    result.encode_ms += allocation_ms
    result.metadata.update({"allocation_ms": allocation_ms, "candidate_bank_ms": bank.build_ms})
    if result.file_size > bank.target_bytes:
        raise RuntimeError("learned allocation exceeded its byte budget")
    return result
