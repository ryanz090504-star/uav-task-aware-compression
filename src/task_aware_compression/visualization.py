from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .compressor import compress_adaptive, compress_uniform
from .importance import tile_importance_map
from .types import CompressionConfig, Detection


def _matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as error:
        raise RuntimeError("Visualization requires matplotlib; install requirements-cpu.txt") from error


def _save_figure(figure: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    _matplotlib().close(figure)


def save_importance_visualization(
    image: np.ndarray,
    importance: np.ndarray,
    detections: list[Detection],
    path: str | Path,
    title: str = "Human importance map",
) -> None:
    """Save candidate boxes, heatmap overlay, and importance distribution."""
    plt = _matplotlib()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.6),
        gridspec_kw={"width_ratios": (1.05, 1.05, 0.9)},
        layout="constrained",
    )
    axes[0].imshow(image)
    axes[0].set_title("Candidate detections")
    axes[1].imshow(image)
    heatmap = axes[1].imshow(importance, cmap="inferno", alpha=0.64, vmin=0, vmax=1)
    axes[1].set_title(title)
    colorbar = figure.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.025)
    colorbar.ax.set_title("score", fontsize=9, pad=8)
    axes[2].hist(importance.ravel(), bins=30, range=(0, 1), color="#d95f02", alpha=0.9)
    axes[2].set_title("Pixel importance distribution")
    axes[2].set_xlabel("importance")
    axes[2].set_ylabel("pixels")
    for detection in detections:
        box = detection.box
        for axis, color in ((axes[0], "#35e06f"), (axes[1], "white")):
            axis.add_patch(plt.Rectangle((box.x1, box.y1), box.width, box.height, fill=False, edgecolor=color, linewidth=1.5))
    for axis in axes[:2]:
        axis.set_axis_off()
    figure.suptitle(title, fontsize=14, fontweight="bold")
    _save_figure(figure, path)


def save_tile_quality_visualization(
    image: np.ndarray,
    importance: np.ndarray,
    config: CompressionConfig,
    tile_qualities: list[int],
    path: str | Path,
) -> None:
    """Save the importance-to-quality allocation for every coding tile."""
    plt = _matplotlib()
    tile_map = tile_importance_map(importance, config.tile_size)
    rows, cols = tile_map.shape
    quality_map = np.asarray(tile_qualities, dtype=np.float32).reshape(rows, cols)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(image)
    axes[0].set_title("Importance by tile")
    axes[1].imshow(quality_map, cmap="viridis", vmin=config.background_quality, vmax=config.person_quality)
    axes[1].set_title("JPEG quality by tile")
    axes[0].set_xticks(np.arange(cols) * config.tile_size + config.tile_size / 2)
    axes[0].set_yticks(np.arange(rows) * config.tile_size + config.tile_size / 2)
    axes[1].set_xticks(np.arange(cols))
    axes[1].set_yticks(np.arange(rows))
    for axis in axes:
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.grid(color="white", linewidth=0.45, alpha=0.7)
    annotation_threshold = max(0.15, float(np.quantile(tile_map, 0.9)))
    for row in range(rows):
        for col in range(cols):
            if tile_map[row, col] >= annotation_threshold:
                axes[0].text(col * config.tile_size + config.tile_size / 2, row * config.tile_size + config.tile_size / 2, f"{tile_map[row, col]:.2f}", ha="center", va="center", fontsize=6, color="white", bbox={"facecolor": "black", "alpha": 0.35, "pad": 1})
            if tile_map[row, col] >= annotation_threshold:
                axes[1].text(col, row, str(int(quality_map[row, col])), ha="center", va="center", fontsize=7, color="white")
    figure.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04, label="JPEG quality")
    figure.suptitle(f"Tile allocation ({config.tile_size}x{config.tile_size})", fontsize=14, fontweight="bold")
    _save_figure(figure, path)


def save_compression_strength_visualization(
    image: np.ndarray,
    detections: list[Detection],
    path: str | Path,
    tile_size: int = 64,
    importance_semantics: str = "truncated_6sigma",
) -> list[dict[str, Any]]:
    """Compare four uniform strengths and one adaptive setting."""
    plt = _matplotlib()
    strengths: list[tuple[str, np.ndarray, float, int]] = [("Original", image, 0.0, image.nbytes)]
    for quality in (90, 60, 35, 15):
        result = compress_uniform(image, quality)
        strengths.append((f"Uniform Q{quality}", result.image, result.bits_per_pixel, result.file_size))
    adaptive = compress_adaptive(image, detections, CompressionConfig(tile_size=tile_size, background_quality=5, person_quality=75, importance_semantics=importance_semantics))
    strengths.append(("Adaptive strong\nBG Q5 / person Q75", adaptive.image, adaptive.bits_per_pixel, adaptive.file_size))
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for axis, (label, preview, bpp, file_size) in zip(axes.ravel(), strengths):
        axis.imshow(preview)
        suffix = "original" if file_size == image.nbytes and bpp == 0 else f"{file_size / 1024:.1f} KB | {bpp:.3f} BPP"
        axis.set_title(f"{label}\n{suffix}")
        axis.set_axis_off()
    axes.ravel()[-1].axis("off")
    figure.suptitle("Compression strength comparison", fontsize=15, fontweight="bold")
    _save_figure(figure, path)
    return [{"setting": label.replace("\n", " "), "file_size_bytes": size, "bits_per_pixel": bpp} for label, _, bpp, size in strengths]


def save_metric_visualizations(records_path: str | Path, output_dir: str | Path) -> None:
    """Create performance-vs-compression and efficiency plots from CSV records."""
    plt = _matplotlib()
    output_dir = Path(output_dir)
    with Path(records_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    def label_for(row: dict[str, str]) -> str | None:
        setting = row["setting"]
        if row["mode"] == "original": return "Original"
        if row["mode"] == "uniform": return "Uniform JPEG"
        if row["mode"] == "hevc": return "HEVC Intra"
        if setting.startswith("risk_aware_adaptive_"): return "Risk-Aware Adaptive"
        if setting.startswith("human_roi_"): return "Human ROI"
        if setting.startswith("random_roi_"): return "Random ROI"
        if setting.startswith("oracle_human_roi_"): return "Oracle"
        return None

    grouped_settings: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        label = label_for(row)
        if label is not None:
            grouped_settings.setdefault((label, row["setting"]), []).append(row)
    series: dict[str, list[dict[str, float]]] = {}
    fields = ("bits_per_pixel", "recall", "mAP50", "ap_small", "file_size_bytes", "end_to_end_ms", "fps")
    for (label, _), values in grouped_settings.items():
        series.setdefault(label, []).append({field: sum(float(row[field]) for row in values) / len(values) for field in fields})
    for values in series.values():
        values.sort(key=lambda row: row["bits_per_pixel"])
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for label, values in series.items():
        sizes = [row["bits_per_pixel"] for row in values]
        axes[0].plot(sizes, [row["recall"] for row in values], "o-", label=label)
        axes[1].plot(sizes, [row["mAP50"] for row in values], "o-", label=label)
        axes[2].plot(sizes, [row["ap_small"] for row in values], "o-", label=label)
    for axis, ylabel, title in zip(axes, ("Recall", "mAP50", "AP small"), ("Human recall", "mAP@0.5", "AP Small")):
        axis.set_xlabel("Bits per pixel")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Detection performance versus compression strength", fontsize=15, fontweight="bold")
    _save_figure(figure, output_dir / "compression_cliff_metrics.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, values in series.items():
        sizes = [row["file_size_bytes"] / 1024 for row in values]
        axes[0].plot(sizes, [row["end_to_end_ms"] for row in values], "o-", label=label)
        axes[1].plot(sizes, [row["fps"] for row in values], "o-", label=label)
    axes[0].set(xlabel="File size (KB)", ylabel="End-to-end latency (ms)", title="Storage versus latency")
    axes[1].set(xlabel="File size (KB)", ylabel="FPS", title="Throughput versus storage")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle("Compression efficiency", fontsize=15, fontweight="bold")
    _save_figure(figure, output_dir / "compression_efficiency.png")

    preferred = [
        ("Matched uniform", "paired_risk_aware_adaptive_adaptive_bg8_fg82"),
        ("Random ROI", "random_roi_adaptive_bg8_fg82"),
        ("Human ROI", "human_roi_adaptive_bg8_fg82"),
        ("Risk-Aware", "risk_aware_adaptive_adaptive_bg8_fg82"),
    ]
    comparison: list[tuple[str, dict[str, float]]] = []
    for label, fragment in preferred:
        matches = [row for row in rows if row["setting"].startswith(fragment)]
        if matches:
            comparison.append((label, {field: sum(float(row[field]) for row in matches) / len(matches) for field in ("recall", "mAP50", "ap_small", "bits_per_pixel")}))
    if comparison:
        figure, axes = plt.subplots(1, 4, figsize=(15, 4.4))
        colors = ["#7f8c8d", "#e07a5f", "#3d9970", "#176b87"]
        labels = [item[0] for item in comparison]
        for axis, field, title in zip(axes, ("recall", "mAP50", "ap_small", "bits_per_pixel"), ("Human Recall", "mAP@0.5", "AP Small", "Bits per pixel")):
            values = [item[1][field] for item in comparison]
            bars = axis.bar(labels, values, color=colors[:len(values)])
            axis.set_title(title)
            axis.tick_params(axis="x", rotation=24, labelsize=8)
            axis.grid(axis="y", alpha=0.2)
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        figure.suptitle("Same-budget method comparison", fontsize=15, fontweight="bold")
        _save_figure(figure, output_dir / "method_comparison.png")


def save_small_person_zoom(
    image: np.ndarray,
    ground_truth: list[Detection],
    adaptive_image: np.ndarray,
    path: str | Path,
) -> None:
    """Show how strong uniform and adaptive coding affect the smallest people."""
    if not ground_truth:
        return
    plt = _matplotlib()
    uniform = compress_uniform(image, 15).image
    smallest = sorted(ground_truth, key=lambda item: item.box.area)[:3]
    figure, axes = plt.subplots(len(smallest), 3, figsize=(9, 3 * len(smallest)), squeeze=False)
    variants = [("Original", image), ("Uniform Q15", uniform), ("Adaptive", adaptive_image)]
    height, width = image.shape[:2]
    for row, detection in enumerate(smallest):
        box = detection.box
        pad = max(18, int(max(box.width, box.height) * 3))
        x1, y1 = max(0, int(box.x1) - pad), max(0, int(box.y1) - pad)
        x2, y2 = min(width, int(box.x2) + pad), min(height, int(box.y2) + pad)
        for col, (label, variant) in enumerate(variants):
            axes[row, col].imshow(variant[y1:y2, x1:x2])
            axes[row, col].set_title(f"{label} | person {box.width:.0f}x{box.height:.0f}px")
            axes[row, col].set_axis_off()
    figure.suptitle("Small-person preservation under strong compression", fontsize=14, fontweight="bold")
    _save_figure(figure, path)


def save_workflow_diagram(path: str | Path) -> None:
    """Render the end-to-end task-aware compression workflow."""
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(15, 3.2))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 2)
    axis.axis("off")
    steps = [("1. UAV image", "VisDrone / real frame"), ("2. Candidate detection", "low threshold\nprioritize recall"), ("3. Importance map", "confidence + size\ncontext + texture"), ("4. Adaptive coding", "tile quality\nstrong background"), ("5. Detection", "same YOLO model\non decoded image"), ("6. Evaluation", "Recall / mAP / BPP\nlatency / FPS")]
    for index, (heading, detail) in enumerate(steps):
        x = 0.15 + index * 1.65
        axis.text(x + 0.66, 1.2, heading, ha="center", va="center", fontsize=10, fontweight="bold", color="white", bbox={"boxstyle": "round,pad=0.55", "facecolor": "#176b87", "edgecolor": "#0b3954", "linewidth": 1.5})
        axis.text(x + 0.66, 0.52, detail, ha="center", va="center", fontsize=8, color="#17324d")
        if index < len(steps) - 1:
            axis.annotate("", xy=(x + 1.48, 1.2), xytext=(x + 1.34, 1.2), arrowprops={"arrowstyle": "->", "color": "#e07a5f", "linewidth": 2})
    figure.suptitle("Task-aware UAV compression workflow", fontsize=15, fontweight="bold")
    _save_figure(figure, path)


def write_workflow_manifest(output_dir: str | Path, summary: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    manifest = {"workflow": ["load real UAV image and YOLO labels", "run low-threshold candidate person detection", "build continuous Human Importance Map", "allocate tile JPEG qualities from importance", "decode and run the same detector", "compare Original, Uniform JPEG, HEVC Intra, Random ROI, Human ROI, Risk-Aware Adaptive, and optional Oracle methods", "report Recall, mAP, AP small, miss rate, file size, BPP, latency, and FPS"], "outputs": ["workflow_diagram.png", "compression_cliff_metrics.png", "compression_efficiency.png", "method_comparison.png", "visual_samples/*/importance_visualization.png", "visual_samples/*/tile_quality_visualization.png", "visual_samples/*/compression_strength.png", "visual_samples/*/small_person_zoom.png"], "summary": summary}
    (output_dir / "workflow_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
