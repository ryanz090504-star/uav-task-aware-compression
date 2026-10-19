from __future__ import annotations
import io
import json
import struct
import time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from .importance import build_importance_map, tile_importance_map
from .types import CompressionConfig, Detection

_MAGIC = b"TAC1"

@dataclass
class CompressionResult:
    image: np.ndarray
    encoded: bytes
    mode: str
    file_size: int
    raw_size: int
    bits_per_pixel: float
    compression_ratio: float
    encode_ms: float
    decode_ms: float
    tile_qualities: list[int]
    importance: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8: image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2: image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4: image = image[..., :3]
    if image.ndim != 3 or image.shape[2] != 3: raise ValueError("image must be an HxWx3 or HxWx4 array")
    return np.ascontiguousarray(image)

def _encode_jpeg(tile: np.ndarray, quality: int) -> bytes:
    output = io.BytesIO()
    Image.fromarray(tile, mode="RGB").save(output, format="JPEG", quality=int(quality), optimize=True, subsampling=0)
    return output.getvalue()

def _decode_jpeg(payload: bytes, width: int, height: int) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        result = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if result.shape[:2] != (height, width):
        result = np.asarray(Image.fromarray(result).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)
    return result

def _quality_for_importance(value: float, config: CompressionConfig) -> int:
    value = max(0.0, min(1.0, float(value)))
    # Smooth mapping keeps moderate-importance context cheaper than the person tile.
    if config.quality_curve == "linear": eased = value
    elif config.quality_curve == "sqrt": eased = np.sqrt(value)
    elif config.quality_curve == "square": eased = value * value
    else: eased = value * value * (3.0 - 2.0 * value)
    return int(round(config.background_quality + eased * (config.person_quality - config.background_quality)))

def encode_uniform(image: np.ndarray, quality: int = 50) -> bytes:
    return _encode_jpeg(_as_rgb(image), quality)

def decode_uniform(encoded: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(encoded)) as image: return np.asarray(image.convert("RGB"), dtype=np.uint8)

def encode_adaptive(image: np.ndarray, detections: list[Detection], config: CompressionConfig | None = None) -> tuple[bytes, np.ndarray, list[int], dict[str, Any]]:
    config = config or CompressionConfig()
    image = _as_rgb(image)
    importance = build_importance_map(image, detections, config)
    height, width = image.shape[:2]
    tile_jobs: list[tuple[int, int, int, int, int, np.ndarray]] = []
    if config.tile_offset_x or config.tile_offset_y:
        xs = [0] + list(range(config.tile_offset_x or config.tile_size, width, config.tile_size)) + [width]
        ys = [0] + list(range(config.tile_offset_y or config.tile_size, height, config.tile_size)) + [height]
        rectangles = [(xs[col], ys[row], xs[col + 1], ys[row + 1]) for row in range(len(ys) - 1) for col in range(len(xs) - 1)]
        def score_rectangle(x1: int, y1: int, x2: int, y2: int) -> float:
            tile = importance[y1:y2, x1:x2]
            mean = float(tile.mean())
            if config.tile_pooling == "mean": return mean
            if config.tile_pooling == "max": return float(tile.max())
            peak = float(np.percentile(tile, 95))
            if config.tile_pooling == "p95": return peak
            weight = 0.25 if config.tile_pooling == "blend25" else 0.5
            return (1 - weight) * mean + weight * peak
        scores = [score_rectangle(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles]
    else:
        tile_map = tile_importance_map(importance, config.tile_size, config.tile_pooling)
        rectangles = [(col * config.tile_size, row * config.tile_size, min(width, (col + 1) * config.tile_size), min(height, (row + 1) * config.tile_size)) for row in range(tile_map.shape[0]) for col in range(tile_map.shape[1])]
        scores = list(tile_map.flat)
    for (x1, y1, x2, y2), score in zip(rectangles, scores):
        quality = _quality_for_importance(score, config)
        tile_jobs.append((x1, y1, x2, y2, quality, image[y1:y2, x1:x2]))
    if config.tile_workers == 1:
        tiles = [_encode_jpeg(tile, quality) for _, _, _, _, quality, tile in tile_jobs]
    else:
        with ThreadPoolExecutor(max_workers=config.tile_workers) as executor:
            tiles = list(executor.map(lambda job: _encode_jpeg(job[5], job[4]), tile_jobs))
    qualities = [quality for _, _, _, _, quality, _ in tile_jobs]
    tile_records = [
        {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "q": quality, "n": len(payload)}
        for (x1, y1, x2, y2, quality, _), payload in zip(tile_jobs, tiles)
    ]
    header = {"version": 2, "width": width, "height": height, "tile_size": config.tile_size, "tile_workers": config.tile_workers, "tiles": tile_records}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded = _MAGIC + struct.pack(">I", len(header_bytes)) + header_bytes + b"".join(tiles)
    return encoded, importance, qualities, header

def decode_adaptive(encoded: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    if encoded[:4] != _MAGIC: raise ValueError("not a TAC adaptive image")
    if len(encoded) < 8: raise ValueError("truncated TAC header")
    header_length = struct.unpack(">I", encoded[4:8])[0]
    header_end = 8 + header_length
    header = json.loads(encoded[8:header_end].decode("utf-8"))
    image = np.zeros((header["height"], header["width"], 3), dtype=np.uint8)
    offset = header_end
    jobs: list[tuple[dict[str, int], bytes]] = []
    for tile in header["tiles"]:
        size = int(tile["n"]); payload = encoded[offset:offset + size]; offset += size
        if len(payload) != size:
            raise ValueError("truncated TAC tile payload")
        jobs.append((tile, payload))
    workers = max(1, int(header.get("tile_workers", 1)))
    if workers == 1:
        decoded_tiles = [_decode_jpeg(payload, tile["w"], tile["h"]) for tile, payload in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            decoded_tiles = list(executor.map(lambda job: _decode_jpeg(job[1], job[0]["w"], job[0]["h"]), jobs))
    for (tile, _), tile_image in zip(jobs, decoded_tiles):
        image[tile["y"]:tile["y"] + tile["h"], tile["x"]:tile["x"] + tile["w"]] = tile_image
    if offset != len(encoded): raise ValueError("TAC payload has trailing bytes")
    return image, header

def compress_uniform(image: np.ndarray, quality: int = 50) -> CompressionResult:
    image = _as_rgb(image); raw_size = image.nbytes
    start = time.perf_counter(); encoded = encode_uniform(image, quality); encode_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter(); decoded = decode_uniform(encoded); decode_ms = (time.perf_counter() - start) * 1000
    pixels = image.shape[0] * image.shape[1]
    return CompressionResult(decoded, encoded, "uniform", len(encoded), raw_size, len(encoded) * 8 / pixels, raw_size / max(1, len(encoded)), encode_ms, decode_ms, [quality], None, {"quality": quality})

def compress_adaptive(image: np.ndarray, detections: list[Detection], config: CompressionConfig | None = None) -> CompressionResult:
    image = _as_rgb(image); raw_size = image.nbytes
    start = time.perf_counter(); encoded, importance, qualities, metadata = encode_adaptive(image, detections, config); encode_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter(); decoded, _ = decode_adaptive(encoded); decode_ms = (time.perf_counter() - start) * 1000
    pixels = image.shape[0] * image.shape[1]
    return CompressionResult(decoded, encoded, "adaptive", len(encoded), raw_size, len(encoded) * 8 / pixels, raw_size / max(1, len(encoded)), encode_ms, decode_ms, qualities, importance, metadata)


def compress_budgeted(
    image: np.ndarray,
    detections: list[Detection],
    target_bytes: int,
    config: CompressionConfig | None = None,
    *,
    peak_weight: float = 0.45,
    quality_ladder: tuple[int, ...] = (5, 8, 15, 25, 40, 60, 82),
) -> CompressionResult:
    """Spend a fixed byte budget on high-importance tile quality upgrades.

    All tile candidates are encoded before selection. The target is supplied by
    the caller and must be determined without ground-truth labels.
    """
    config = config or CompressionConfig(tile_size=256)
    if target_bytes <= 0: raise ValueError("target_bytes must be positive")
    if not 0 <= peak_weight <= 1: raise ValueError("peak_weight must be in [0, 1]")
    ladder = tuple(sorted(set(int(q) for q in quality_ladder)))
    if not ladder or ladder[0] < 1 or ladder[-1] > 95: raise ValueError("invalid quality ladder")
    image = _as_rgb(image)
    raw_size = image.nbytes
    height, width = image.shape[:2]
    start = time.perf_counter()
    importance = build_importance_map(image, detections, config)
    means = tile_importance_map(importance, config.tile_size, "mean")
    peaks = tile_importance_map(importance, config.tile_size, "p95") if peak_weight else means
    jobs = []
    for row in range(means.shape[0]):
        for col in range(means.shape[1]):
            x, y = col * config.tile_size, row * config.tile_size
            tile = image[y:min(height, y + config.tile_size), x:min(width, x + config.tile_size)]
            candidates = [_encode_jpeg(tile, q) for q in ladder]
            score = 0.08 + float(means[row, col]) + peak_weight * max(0.0, float(peaks[row, col] - means[row, col]))
            jobs.append((x, y, tile.shape[1], tile.shape[0], score, candidates))
    levels = [0] * len(jobs)
    payload_bytes = sum(len(job[5][0]) for job in jobs)

    def header_for(levels_: list[int]) -> dict[str, Any]:
        records = [
            {"x": job[0], "y": job[1], "w": job[2], "h": job[3], "q": ladder[level], "n": len(job[5][level])}
            for job, level in zip(jobs, levels_)
        ]
        return {"version": 2, "width": width, "height": height, "tile_size": config.tile_size, "tile_workers": config.tile_workers, "tiles": records}

    def header_bytes(levels_: list[int]) -> bytes:
        return json.dumps(header_for(levels_), separators=(",", ":")).encode("utf-8")

    # A few spare bytes absorb changes in the serialized tile lengths.
    payload_limit = target_bytes - 8 - len(header_bytes(levels)) - 64
    while True:
        best = None
        for index, job in enumerate(jobs):
            level = levels[index]
            if level + 1 >= len(ladder): continue
            extra = len(job[5][level + 1]) - len(job[5][level])
            if payload_bytes + extra > payload_limit: continue
            gain = job[4] * (np.sqrt(ladder[level + 1]) - np.sqrt(ladder[level]))
            priority = float(gain / max(1, extra))
            if best is None or priority > best[0]: best = (priority, index, extra)
        if best is None: break
        _, index, extra = best
        levels[index] += 1
        payload_bytes += extra
    header = header_for(levels)
    header_blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded = _MAGIC + struct.pack(">I", len(header_blob)) + header_blob + b"".join(job[5][level] for job, level in zip(jobs, levels))
    encode_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    decoded, _ = decode_adaptive(encoded)
    decode_ms = (time.perf_counter() - start) * 1000
    pixels = height * width
    return CompressionResult(decoded, encoded, "budgeted", len(encoded), raw_size, len(encoded) * 8 / pixels, raw_size / max(1, len(encoded)), encode_ms, decode_ms, [ladder[level] for level in levels], importance, header)


def compress_floor_constrained(
    image: np.ndarray,
    detections: list[Detection],
    target_bytes: int,
    background_quality: int,
    config: CompressionConfig | None = None,
) -> CompressionResult:
    """Raise the background quality floor and solve foreground Q for budget.

    The search uses only encoded byte counts. The returned result retains the
    exact TAC bitstream and records the selected foreground quality.
    """
    config = config or CompressionConfig(tile_size=256)
    if target_bytes <= 0: raise ValueError("target_bytes must be positive")
    if not 1 <= background_quality <= 95: raise ValueError("background_quality must be in [1, 95]")
    cache: dict[int, CompressionResult] = {}
    def at(foreground_quality: int) -> CompressionResult:
        if foreground_quality not in cache:
            candidate = replace(config, background_quality=background_quality, person_quality=foreground_quality)
            cache[foreground_quality] = compress_adaptive(image, detections, candidate)
        return cache[foreground_quality]
    low, high = background_quality, 95
    while low < high:
        mid = (low + high) // 2
        if at(mid).file_size < target_bytes: low = mid + 1
        else: high = mid
    for quality in range(max(background_quality, low - 2), min(95, low + 2) + 1):
        at(quality)
    quality, result = min(cache.items(), key=lambda item: abs(item[1].file_size - target_bytes))
    result.mode = "floor_constrained"
    result.metadata = {**(result.metadata or {}), "selected_foreground_quality": quality, "target_bytes": target_bytes}
    return result


def deblock_tile_seams(image: np.ndarray, tile_size: int, blend: float = 0.5) -> np.ndarray:
    """Interpolate only the two pixel columns/rows touching each tile seam."""
    if tile_size < 2: raise ValueError("tile_size must be at least 2")
    if not 0 <= blend <= 1: raise ValueError("blend must be in [0, 1]")
    result = image.astype(np.float32).copy()
    height, width = result.shape[:2]
    for x in range(tile_size, width, tile_size):
        if x < 2 or x + 1 >= width: continue
        left, right = result[:, x - 2].copy(), result[:, x + 1].copy()
        result[:, x - 1] = (1 - blend) * result[:, x - 1] + blend * (2 * left + right) / 3
        result[:, x] = (1 - blend) * result[:, x] + blend * (left + 2 * right) / 3
    for y in range(tile_size, height, tile_size):
        if y < 2 or y + 1 >= height: continue
        top, bottom = result[y - 2].copy(), result[y + 1].copy()
        result[y - 1] = (1 - blend) * result[y - 1] + blend * (2 * top + bottom) / 3
        result[y] = (1 - blend) * result[y] + blend * (top + 2 * bottom) / 3
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)

def save_encoded(result: CompressionResult, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(result.encoded)

def save_preview(result: CompressionResult, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(result.image).save(path)

def save_importance_map(importance: np.ndarray, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(importance * 255, 0, 255).astype(np.uint8), mode="L").save(path)
