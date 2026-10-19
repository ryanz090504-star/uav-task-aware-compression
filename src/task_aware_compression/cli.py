from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .benchmark import compress_file, load_detections, run_demo, run_yolo_dataset
from .detector import build_detector
from .types import CompressionConfig, IMPORTANCE_SEMANTICS
from .workflow import run_workflow


def _qualities(value: str) -> tuple[int, ...]:
    try:
        qualities = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("qualities must be a comma-separated list of integers") from error
    if not qualities or min(qualities) < 1 or max(qualities) > 100:
        raise argparse.ArgumentTypeError("qualities must be in [1, 100]")
    return qualities

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-aware adaptive compression for UAV human detection")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the reproducible synthetic benchmark")
    demo.add_argument("--output", default="outputs/demo"); demo.add_argument("--samples", type=int, default=6); demo.add_argument("--seed", type=int, default=7); demo.add_argument("--detector", choices=["heuristic", "yolo", "auto"], default="heuristic"); demo.add_argument("--model", default="yolov8n.pt")
    benchmark = sub.add_parser("benchmark", help="evaluate a YOLO-labelled image folder")
    benchmark.add_argument("--images", required=True); benchmark.add_argument("--labels", required=True); benchmark.add_argument("--metadata"); benchmark.add_argument("--output", default="outputs/benchmark"); benchmark.add_argument("--detector", choices=["heuristic", "yolo", "auto"], default="yolo"); benchmark.add_argument("--model", default="yolov8n.pt"); benchmark.add_argument("--confidence", type=float, default=0.12); benchmark.add_argument("--device"); benchmark.add_argument("--person-class", type=int, default=0); benchmark.add_argument("--limit", type=int); benchmark.add_argument("--oracle", action="store_true"); benchmark.add_argument("--uniform-qualities", type=_qualities, default=(90, 60, 35, 15)); benchmark.add_argument("--pairing-qualities", type=_qualities, default=tuple(range(1, 101))); benchmark.add_argument("--detection-batch-size", type=int, default=1); benchmark.add_argument("--tile-workers", type=int, default=1)
    workflow = sub.add_parser("workflow", help="run real-data benchmark and generate research visualizations")
    workflow.add_argument("--images", required=True); workflow.add_argument("--labels", required=True); workflow.add_argument("--metadata"); workflow.add_argument("--output", default="outputs/workflow"); workflow.add_argument("--detector", choices=["heuristic", "yolo", "auto"], default="yolo"); workflow.add_argument("--model", default="yolov8n.pt"); workflow.add_argument("--confidence", type=float, default=0.12); workflow.add_argument("--device"); workflow.add_argument("--person-class", type=int, default=0); workflow.add_argument("--limit", type=int); workflow.add_argument("--oracle", action="store_true"); workflow.add_argument("--visual-samples", type=int, default=4); workflow.add_argument("--tile-size", type=int, default=64); workflow.add_argument("--no-hevc", action="store_true"); workflow.add_argument("--uniform-qualities", type=_qualities, default=(90, 60, 35, 15)); workflow.add_argument("--pairing-qualities", type=_qualities, default=tuple(range(1, 101))); workflow.add_argument("--detection-batch-size", type=int, default=1); workflow.add_argument("--tile-workers", type=int, default=1)
    compress = sub.add_parser("compress", help="compress one image")
    compress.add_argument("--input", required=True); compress.add_argument("--output", required=True); compress.add_argument("--mode", choices=["uniform", "adaptive"], default="adaptive"); compress.add_argument("--quality", type=int, default=50); compress.add_argument("--detections"); compress.add_argument("--detector", choices=["heuristic", "yolo", "auto"], default="auto"); compress.add_argument("--model", default="yolov8n.pt"); compress.add_argument("--background-quality", type=int, default=5); compress.add_argument("--person-quality", type=int, default=75); compress.add_argument("--tile-size", type=int, default=128); compress.add_argument("--tile-workers", type=int, default=1)
    for command in (benchmark, workflow, compress):
        command.add_argument("--importance-semantics", choices=IMPORTANCE_SEMANTICS, default="truncated_6sigma" if command is compress else "historical_full_gaussian", help="Gaussian support semantics; full-image mode underlies the frozen v0.1 Risk-Aware headline")
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command in {"benchmark", "workflow"}:
        for directory in (args.images, args.labels):
            if not Path(directory).is_dir():
                parser.error(f"Missing dataset directory: {directory}. Follow DATASET.md; no dataset is bundled.")
    if args.command == "compress" and not Path(args.input).is_file():
        parser.error(f"Missing input image: {args.input}. Supply a local image or run the synthetic demo.")
    def detector():
        try:
            return build_detector(args.detector, args.model, getattr(args, "confidence", 0.12), getattr(args, "device", None))
        except RuntimeError as error:
            parser.error(str(error))
    if args.command == "demo":
        summary = run_demo(args.output, args.samples, args.seed, detector())
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    if args.command == "benchmark":
        selected_detector = detector()
        summary = run_yolo_dataset(args.images, args.labels, args.output, selected_detector, args.limit, args.person_class, args.oracle, metadata_dir=args.metadata, uniform_qualities=args.uniform_qualities, pairing_qualities=args.pairing_qualities, detection_batch_size=args.detection_batch_size, tile_workers=args.tile_workers, importance_semantics=args.importance_semantics)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    if args.command == "workflow":
        selected_detector = detector()
        summary = run_workflow(args.images, args.labels, args.output, selected_detector, args.limit, args.person_class, args.oracle, args.visual_samples, args.tile_size, include_hevc=not args.no_hevc, metadata_dir=args.metadata, uniform_qualities=args.uniform_qualities, pairing_qualities=args.pairing_qualities, detection_batch_size=args.detection_batch_size, tile_workers=args.tile_workers, importance_semantics=args.importance_semantics)
        print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    if args.command == "compress":
        detections = load_detections(args.detections) if args.detections else None
        if args.mode == "adaptive" and detections is None:
            selected_detector = detector()
            with Image.open(args.input) as source: image = np.asarray(source.convert("RGB"), dtype=np.uint8)
            detections = selected_detector.predict(image)
        config = CompressionConfig(tile_size=args.tile_size, tile_workers=args.tile_workers, background_quality=args.background_quality, person_quality=args.person_quality, importance_semantics=args.importance_semantics)
        print(json.dumps(compress_file(args.input, args.output, args.mode, detections, args.quality, config), ensure_ascii=False, indent=2)); return 0
    return 2

if __name__ == "__main__": raise SystemExit(main())


