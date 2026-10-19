from __future__ import annotations
from pathlib import Path
from typing import Any, Protocol
import numpy as np
from PIL import Image, ImageDraw
from .types import BoundingBox, Detection

class Detector(Protocol):
    def predict(self, image: np.ndarray) -> list[Detection]: ...

def _components(mask: np.ndarray, min_pixels: int) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape; visited = np.zeros_like(mask, dtype=bool); components = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]: continue
            stack = [(y, x)]; visited[y, x] = True; count = 0; x1 = x2 = x; y1 = y2 = y
            while stack:
                cy, cx = stack.pop(); count += 1; x1, x2 = min(x1, cx), max(x2, cx); y1, y2 = min(y1, cy), max(y2, cy)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]: visited[ny, nx] = True; stack.append((ny, nx))
            if count >= min_pixels: components.append((x1, y1, x2 + 1, y2 + 1, count))
    return components

class ColorBlobPersonDetector:
    """Dependency-free detector for the included synthetic rescue benchmark.

    For real imagery, use ``UltralyticsPersonDetector`` when the optional
    ultralytics dependency is installed.
    """
    def __init__(self, threshold: float = 0.35, min_area: int = 8):
        self.threshold, self.min_area = threshold, min_area
    def predict(self, image: np.ndarray) -> list[Detection]:
        rgb = image[..., :3].astype(np.float32) / 255.0
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mask = (red > self.threshold) & (red > green * 1.12) & (red > blue * 1.05) & ((red - green) > 0.08)
        predictions: list[Detection] = []
        image_area = image.shape[0] * image.shape[1]
        for x1, y1, x2, y2, count in _components(mask, self.min_area):
            width, height = x2 - x1, y2 - y1
            if width < 2 or height < 3 or width > image.shape[1] * 0.35 or height > image.shape[0] * 0.35: continue
            aspect = height / max(1, width)
            if aspect < 0.65: continue
            confidence = min(0.99, 0.50 + 0.30 * min(1.0, aspect / 3.0) + 0.20 * min(1.0, count / max(1, width * height)))
            pad_x, pad_y = max(1, int(width * 0.14)), max(1, int(height * 0.10))
            predictions.append(Detection(BoundingBox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), confidence))
        return sorted(predictions, key=lambda item: item.confidence, reverse=True)

class UltralyticsPersonDetector:
    def __init__(self, model: str = "yolov8n.pt", confidence: float = 0.12, device: str | None = None, imgsz: int = 1280):
        if not Path(model).is_file():
            raise RuntimeError(f"Missing local detector checkpoint: {model}. No weights are distributed; supply a lawfully obtained local file. See MODEL_PROVENANCE.md.")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Ultralytics is not installed; install the 'yolo' extra or use --detector heuristic") from error
        self.model = YOLO(model); self.confidence = confidence; self.device = device; self.imgsz = imgsz
    def _convert_result(self, result: Any) -> list[Detection]:
        predictions: list[Detection] = []
        boxes = result.boxes
        for index in range(len(boxes)):
            label = int(boxes.cls[index].item()) if boxes.cls is not None else -1
            if label != 0: continue
            coords = boxes.xyxy[index].tolist(); confidence = float(boxes.conf[index].item())
            predictions.append(Detection(BoundingBox(*coords), confidence))
        return predictions
    def predict(self, image: np.ndarray) -> list[Detection]:
        kwargs = {"conf": self.confidence, "verbose": False, "imgsz": self.imgsz}
        if self.device: kwargs["device"] = self.device
        return self._convert_result(self.model.predict(source=image, **kwargs)[0])
    def predict_batch(self, images: list[np.ndarray], batch_size: int = 4) -> list[list[Detection]]:
        if not images: return []
        kwargs = {"conf": self.confidence, "verbose": False, "imgsz": self.imgsz, "batch": batch_size}
        if self.device: kwargs["device"] = self.device
        results = self.model.predict(source=images, **kwargs)
        return [self._convert_result(result) for result in results]

def build_detector(kind: str = "auto", model: str = "yolov8n.pt", confidence: float = 0.12, device: str | None = None) -> Detector:
    if kind == "heuristic": return ColorBlobPersonDetector()
    if kind == "yolo": return UltralyticsPersonDetector(model, confidence, device)
    if kind == "auto":
        try: return UltralyticsPersonDetector(model, confidence, device)
        except RuntimeError: return ColorBlobPersonDetector()
    raise ValueError(f"unknown detector: {kind}")

def draw_detections(image: np.ndarray, detections: list[Detection], output: str | Path) -> None:
    canvas = Image.fromarray(image[..., :3].astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        box = detection.box
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=(40, 255, 80), width=2)
        draw.text((box.x1 + 2, max(0, box.y1 - 13)), f"person {detection.confidence:.2f}", fill=(40, 255, 80))
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path)
