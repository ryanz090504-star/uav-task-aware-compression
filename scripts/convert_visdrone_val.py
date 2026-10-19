"""Convert VisDrone validation labels and retain occlusion/truncation sidecars."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path("data/VisDrone")
SOURCE = ROOT / "VisDrone2019-DET-val"
IMAGES = ROOT / "images" / "val"
LABELS = ROOT / "labels" / "val"
METADATA = ROOT / "metadata" / "val"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for directory in (SOURCE / "images", SOURCE / "annotations"):
        if not directory.is_dir():
            parser.error(f"Missing dataset directory: {directory}. Acquire official data and follow DATASET.md first.")
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    for image in sorted((SOURCE / "images").glob("*.jpg")):
        shutil.copy2(image, IMAGES / image.name)
    for annotation in sorted((SOURCE / "annotations").glob("*.txt")):
        image_path = IMAGES / f"{annotation.stem}.jpg"
        if not image_path.exists():
            continue
        width, height = Image.open(image_path).size
        labels: list[str] = []
        persons: list[dict[str, int]] = []
        for row in annotation.read_text(encoding="utf-8").strip().splitlines():
            values = row.split(",")
            if len(values) < 8 or values[4] == "0":
                continue
            x, y, box_width, box_height = map(int, values[:4])
            category = int(values[5])
            if category not in (1, 2):
                continue
            labels.append(f"0 {(x + box_width / 2) / width:.6f} {(y + box_height / 2) / height:.6f} {box_width / width:.6f} {box_height / height:.6f}\n")
            persons.append({"category": category, "truncation": int(values[6]), "occlusion": int(values[7])})
        (LABELS / f"{annotation.stem}.txt").write_text("".join(labels), encoding="utf-8")
        (METADATA / f"{annotation.stem}.json").write_text(json.dumps({"image": image_path.name, "persons": persons}, ensure_ascii=False), encoding="utf-8")
    print("images", len(list(IMAGES.glob("*.jpg"))), "labels", len(list(LABELS.glob("*.txt"))), "metadata", len(list(METADATA.glob("*.json"))))


if __name__ == "__main__":
    main()
