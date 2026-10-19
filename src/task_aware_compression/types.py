from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

IMPORTANCE_SEMANTICS = ("historical_full_gaussian", "truncated_6sigma")

@dataclass(frozen=True)
class BoundingBox:
    """An image-space box using xyxy pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    @property
    def width(self) -> float: return max(0.0, self.x2 - self.x1)
    @property
    def height(self) -> float: return max(0.0, self.y2 - self.y1)
    @property
    def area(self) -> float: return self.width * self.height
    @property
    def center(self) -> tuple[float, float]: return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    def clip(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(max(0.0, min(float(width), self.x1)), max(0.0, min(float(height), self.y1)), max(0.0, min(float(width), self.x2)), max(0.0, min(float(height), self.y2)))
    def iou(self, other: "BoundingBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

@dataclass
class Detection:
    box: BoundingBox
    confidence: float = 1.0
    label: str = "person"
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CompressionConfig:
    tile_size: int = 128
    tile_workers: int = 1
    background_quality: int = 18
    person_quality: int = 82
    context_radius: int = 18
    small_object_reference: float = 0.018
    texture_weight: float = 0.12
    tile_pooling: str = "mean"
    tile_offset_x: int = 0
    tile_offset_y: int = 0
    small_object_weight: float = 0.58
    uncertainty_weight: float = 0.42
    quality_curve: str = "smoothstep"
    importance_aggregation: str = "max"
    # Preserve the later allocator behavior for legacy API/checkpoint configs.
    # The v0.1 Risk-Aware workflow selects historical_full_gaussian explicitly.
    importance_semantics: str = "truncated_6sigma"
    def __post_init__(self) -> None:
        if self.importance_semantics not in IMPORTANCE_SEMANTICS:
            raise ValueError(f"unknown importance semantics: {self.importance_semantics}")
        if self.tile_size < 8: raise ValueError("tile_size must be at least 8")
        if self.tile_workers < 1: raise ValueError("tile_workers must be positive")
        if not 1 <= self.background_quality <= 95: raise ValueError("background_quality must be in [1, 95]")
        if not 1 <= self.person_quality <= 95: raise ValueError("person_quality must be in [1, 95]")
        if self.background_quality > self.person_quality: raise ValueError("background_quality cannot exceed person_quality")
        if self.context_radius < 0: raise ValueError("context_radius cannot be negative")
        if self.tile_pooling not in {"mean", "p95", "max", "blend25", "blend50"}: raise ValueError("unknown tile pooling")
        if not 0 <= self.tile_offset_x < self.tile_size: raise ValueError("tile_offset_x must be within the tile")
        if not 0 <= self.tile_offset_y < self.tile_size: raise ValueError("tile_offset_y must be within the tile")
        if self.small_object_weight < 0 or self.uncertainty_weight < 0 or abs(self.small_object_weight + self.uncertainty_weight - 1.0) > 1e-6: raise ValueError("risk weights must be non-negative and sum to one")
        if self.quality_curve not in {"linear", "smoothstep", "sqrt", "square"}: raise ValueError("unknown quality curve")
        if self.importance_aggregation not in {"max", "sum", "prob_union"}: raise ValueError("unknown importance aggregation")

