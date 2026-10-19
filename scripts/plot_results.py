"""Plot frozen aggregate results; no models, images or detections are read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to((root / 'results').resolve()):
        parser.error('write figures outside frozen results/, for example outputs/results.png')
    try:
        import matplotlib
    except ImportError:
        parser.error("Matplotlib is required for plotting. Install with: python -m pip install -e '.[plots]'")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    baseline = json.loads((root / 'results/baseline/paired_recall_summary.json').read_text())
    learned = json.loads((root / 'results/learned_v2/bootstrap_learned_minus_baseline.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = [
        (axes[0], ['Mean image Recall'], [baseline], 'Risk-Aware vs size-matched JPEG', 'delta_recall'),
        (axes[1], ['Dataset AP@0.5', 'Micro Precision', 'Micro Recall'],
         [learned['metrics'][key] for key in ('dataset_ap50', 'micro_precision', 'micro_recall')],
         'Learned V2 vs baseline (tuned)', 'delta'),
    ]
    for ax, labels, metrics, title, delta_key in groups:
        values = [item[delta_key] * 100 for item in metrics]
        lower = [value - item['ci95'][0] * 100 for value, item in zip(values, metrics)]
        upper = [item['ci95'][1] * 100 - value for value, item in zip(values, metrics)]
        ax.errorbar(values, range(len(labels)), xerr=[lower, upper], fmt='o', capsize=4)
        ax.set_yticks(range(len(labels)), labels)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Difference (percentage points), 95% interval')
        ax.margins(y=.4)
    fig.suptitle('Recorded validation results; validation reused during research')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
