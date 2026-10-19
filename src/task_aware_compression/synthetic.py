from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .types import BoundingBox, Detection

@dataclass
class SyntheticSample:
    image: np.ndarray
    ground_truth: list[Detection]
    name: str

def _draw_person(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float, color: tuple[int, int, int]) -> BoundingBox:
    head_r = max(2, int(3.2 * scale)); body_w = max(3, int(7 * scale)); body_h = max(5, int(14 * scale)); leg_h = max(5, int(10 * scale))
    head_center = (x, y + head_r); draw.ellipse((x - head_r, y, x + head_r, y + 2 * head_r), fill=color)
    body_top, body_bottom = y + 2 * head_r, y + 2 * head_r + body_h
    draw.rounded_rectangle((x - body_w // 2, body_top, x + body_w // 2, body_bottom), radius=max(1, int(scale)), fill=color)
    draw.line((x - body_w // 3, body_bottom, x - body_w // 2, body_bottom + leg_h), fill=color, width=max(1, int(scale * 1.5)))
    draw.line((x + body_w // 3, body_bottom, x + body_w // 2, body_bottom + leg_h), fill=color, width=max(1, int(scale * 1.5)))
    draw.line((x - body_w // 2, body_top + body_h // 3, x - body_w, body_top + body_h // 2), fill=color, width=max(1, int(scale)))
    draw.line((x + body_w // 2, body_top + body_h // 3, x + body_w, body_top + body_h // 2), fill=color, width=max(1, int(scale)))
    return BoundingBox(x - body_w, y, x + body_w, body_bottom + leg_h)

def make_synthetic_dataset(count: int = 6, width: int = 640, height: int = 384, seed: int = 7) -> list[SyntheticSample]:
    rng = random.Random(seed); samples: list[SyntheticSample] = []
    for sample_index in range(count):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        yy = np.arange(height, dtype=np.float32)[:, None]
        horizon = height * (0.42 + 0.04 * rng.random())
        array[..., 0] = np.clip(130 - yy * 0.06 + rng.random() * 8, 45, 180)
        array[..., 1] = np.clip(165 - yy * 0.12 + rng.random() * 8, 65, 190)
        array[..., 2] = np.clip(210 - yy * 0.05 + rng.random() * 10, 100, 230)
        image = Image.fromarray(array, mode="RGB"); draw = ImageDraw.Draw(image)
        draw.rectangle((0, int(horizon), width, height), fill=(62, 89, 60))
        for _ in range(110):
            x, y = rng.randrange(width), rng.randrange(int(horizon), height)
            shade = rng.randrange(50, 125); radius = rng.randrange(1, 5)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade // 2, shade, shade // 2))
        for _ in range(10):
            x, y = rng.randrange(width), rng.randrange(int(horizon), height)
            draw.line((x, y, x + rng.randrange(-35, 36), y + rng.randrange(-4, 5)), fill=(35, 55, 38), width=rng.randrange(2, 6))
        truth: list[Detection] = []
        object_count = 3 + rng.randrange(3)
        for person_index in range(object_count):
            scale = rng.choice([0.48, 0.65, 0.82, 1.05, 1.35])
            x = rng.randrange(int(24 * scale), width - int(24 * scale)); y = rng.randrange(int(horizon) + 4, height - int(34 * scale))
            box = _draw_person(draw, x, y, scale, (190 + rng.randrange(0, 40), 42 + rng.randrange(0, 25), 35))
            truth.append(Detection(box, 1.0, metadata={"synthetic_scale": scale, "id": person_index}))
        if sample_index % 2 == 0: image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
        samples.append(SyntheticSample(np.asarray(image, dtype=np.uint8), truth, f"sample_{sample_index:03d}"))
    return samples
