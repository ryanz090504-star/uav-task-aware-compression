from __future__ import annotations
import math
import numpy as np
from .types import CompressionConfig, Detection

def _box_risk(det: Detection, image_shape: tuple[int, int], reference: float, small_weight: float = 0.58, uncertainty_weight: float = 0.42) -> float:
    height, width = image_shape
    area_ratio = det.box.area / max(1.0, float(width * height))
    small_object_risk = min(1.0, math.sqrt(reference / max(area_ratio, 1e-8)))
    uncertainty = 1.0 - max(0.0, min(1.0, det.confidence))
    return max(0.0, min(1.0, small_weight * small_object_risk + uncertainty_weight * uncertainty))

def build_importance_map(image: np.ndarray, detections: list[Detection], config: CompressionConfig | None = None) -> np.ndarray:
    """Build importance using the explicitly selected Gaussian support semantics.

    Full-image and six-sigma support are scientifically distinct: the texture
    gate tests exact positive support, so discarding small tails changes it.
    """
    config = config or CompressionConfig()
    if image.ndim != 3 or image.shape[2] < 3: raise ValueError("image must have shape (height, width, channels)")
    height, width = image.shape[:2]
    importance = np.zeros((height, width), dtype=np.float32)
    for det in detections:
        box = det.box.clip(width, height)
        if box.area <= 0: continue
        risk = _box_risk(det, (height, width), config.small_object_reference, config.small_object_weight, config.uncertainty_weight)
        sigma_x = max(2.0, box.width * (0.28 + 0.22 * risk) + config.context_radius)
        sigma_y = max(2.0, box.height * (0.28 + 0.22 * risk) + config.context_radius)
        cx, cy = box.center
        if config.importance_semantics == "historical_full_gaussian":
            x1, x2, y1, y2 = 0, width, 0, height
        elif config.importance_semantics == "truncated_6sigma":
            x1 = max(0, int(math.floor(cx - 6 * sigma_x)))
            x2 = min(width, int(math.ceil(cx + 6 * sigma_x)) + 1)
            y1 = max(0, int(math.floor(cy - 6 * sigma_y)))
            y2 = min(height, int(math.ceil(cy + 6 * sigma_y)) + 1)
        else:
            raise ValueError(f"unknown importance semantics: {config.importance_semantics}")
        yy, xx = np.mgrid[y1:y2, x1:x2]
        field = np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
        hard_region = ((xx >= box.x1 - config.context_radius) & (xx < box.x2 + config.context_radius) & (yy >= box.y1 - config.context_radius) & (yy < box.y2 + config.context_radius))
        strength = 0.72 + 0.28 * risk
        field = np.maximum(field * strength, hard_region.astype(np.float32) * strength)
        field = field.astype(np.float32)
        current = importance[y1:y2, x1:x2]
        if config.importance_aggregation == "max":
            importance[y1:y2, x1:x2] = np.maximum(current, field)
        elif config.importance_aggregation == "sum":
            importance[y1:y2, x1:x2] = np.clip(current + field, 0.0, 1.0)
        else:
            importance[y1:y2, x1:x2] = 1.0 - (1.0 - current) * (1.0 - field)
    gray = image[..., :3].astype(np.float32).mean(axis=2)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    texture = np.sqrt(gx * gx + gy * gy)
    scale = float(np.percentile(texture, 90)) if texture.size else 1.0
    if scale > 1e-6:
        texture = np.clip(texture / scale, 0.0, 1.0)
        importance = np.clip(importance + config.texture_weight * texture * (importance > 0), 0.0, 1.0)
    return importance

def tile_importance_map(importance: np.ndarray, tile_size: int, pooling: str = "mean") -> np.ndarray:
    height, width = importance.shape
    rows, cols = (height + tile_size - 1) // tile_size, (width + tile_size - 1) // tile_size
    result = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        y1, y2 = row * tile_size, min(height, (row + 1) * tile_size)
        for col in range(cols):
            x1, x2 = col * tile_size, min(width, (col + 1) * tile_size)
            tile = importance[y1:y2, x1:x2]
            if pooling == "mean": result[row, col] = float(tile.mean())
            elif pooling == "p95": result[row, col] = float(np.percentile(tile, 95))
            elif pooling == "max": result[row, col] = float(tile.max())
            elif pooling == "blend25": result[row, col] = float(0.75 * tile.mean() + 0.25 * np.percentile(tile, 95))
            elif pooling == "blend50": result[row, col] = float(0.5 * tile.mean() + 0.5 * np.percentile(tile, 95))
            else: raise ValueError("unknown tile pooling")
    return result
