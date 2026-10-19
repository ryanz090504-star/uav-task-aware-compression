# Reproduction entry points

Run from the repository root in an activated fresh virtual environment. Synthetic execution, installation, imports, help and the complete retained tests passed on Windows/Python 3.12; see [the validation report](CLEAN_ENVIRONMENT_TEST.md). Real-data release verification subsequently completed: historical Risk-Aware has a documented one-image dependency boundary, and Learned V2 reproduces exactly with historical-compatible components. See REAL_DATA_RELEASE_VERIFICATION.md. Both checkpoints are withheld; no model acquisition links are provided. Independent inference and allocator retraining require suitable lawful local artifacts and detector provenance. Public aggregates permit inspection, not reconstruction of missing detections.

## Synthetic path and tests

```sh
python -m pip install -e .
python -m task_aware_compression.cli demo --detector heuristic --samples 6 --output outputs/demo
python -m unittest discover -s tests -v
```

The heuristic path avoids YOLO/model acquisition. To run all retained tests without optional skips, install the learned extra after reviewing its dependencies, then repeat unittest discovery. On Windows, PyTorch may report a missing Visual C++ Redistributable even when imports succeed; see the validation report before treating a DLL error as a project-code defect. CPU requirements and lock files record the historical research environment, not the newer versions resolved during this validation.

## Local dataset preparation

Follow [DATASET.md](../DATASET.md) to acquire official DET train/val archives and prepare ignored local data. Conversion is not an experiment, but is not executed by release preparation. Do not commit its images, labels or annotation sidecars.

## Risk-Aware baseline — prerequisites withheld

With the exact privately held detector at the path identified in models/PROVENANCE.json, install the yolo and plots extras. This workflow includes baseline controls and optional visual outputs in an ignored directory:

```sh
python -m pip install -e '.[yolo,plots]'
python -m task_aware_compression.cli workflow --images data/VisDrone/images/val --labels data/VisDrone/labels/val --model models/visdrone_yolov8n_1280_b18_s17_best.pt --detector yolo --confidence 0.12 --device cpu --limit 548 --tile-size 256 --no-hevc --visual-samples 0 --detection-batch-size 8 --tile-workers 8 --importance-semantics historical_full_gaussian --output outputs/baseline_local
python scripts/strict_evaluation.py outputs/baseline_local/benchmark/records.csv --output outputs/baseline_local/strict_evaluation.json
```

Dense Q1-Q100 pairing is the CLI default; preserve it. This does not reproduce results with an arbitrary upstream detector. It generates private dataset-derived artifacts that remain excluded. Historical exactness also depends on matching environment and JPEG implementation. Use a fresh output directory.

## Learned V2 — prerequisites withheld

The retained training/config.json is a record, not an automatic --config input. These explicit arguments reflect its recorded balanced-run settings; inspect frozen manifests before use:

```sh
python -m pip install -e '.[learned,plots]'
python scripts/train_rate_allocator_v2.py --model models/visdrone_yolov8n_1280_b18_s17_best.pt --train-count 48 --tune-count 32 --eval-count 80 --size 1280 --epochs 8 --lr 0.0005 --temperature 0.5 --distill-weight 1.0 --identity-weight 0.02 --rate-weight 1 --minimum-ratio 0.98 --confidence 0.12 --threads 4 --device cpu --seed 731 --balanced --output outputs/learned_v2_local
python scripts/evaluate_rate_allocator_v2.py --checkpoint outputs/learned_v2_local/rate_allocator.pt --tune outputs/learned_v2_local/tune --model models/visdrone_yolov8n_1280_b18_s17_best.pt --all --threads 4 --output outputs/learned_v2_eval_local
python scripts/bootstrap_saved_detections.py outputs/learned_v2_eval_local --left learned --right baseline
```

Those retraining commands create a new experiment: the evaluator chooses its operating point from local tune detections. For the original frozen release verification, use the fixed-calibration recipe below instead; do not run a new threshold search. Retraining is not guaranteed to reproduce the original checkpoint; the original allocator is withheld. Source evaluation may generate raw arrays, previews and GT-containing detections locally; these are ignored, not release assets. Do not overwrite frozen results/ or publish generated raw material.

## Frozen V2 release verification without retuning

The completed release check reproduced all 548 images / 76 sequences exactly, including 1,096 raw-output files and both bootstrap files. Its compatible runtime was Python 3.12.14, torch 2.13.0+cpu, torchvision 0.28.0+cpu, Ultralytics 8.4.129, Pillow 12.3.0, NumPy 2.5.2 and OpenCV 5.0.0.93. The archive's Python snapshot says 3.12.13, but the exact original V2 launch inventory is unconfirmed. See REAL_DATA_RELEASE_VERIFICATION.md for recovery provenance, all prerequisite hashes, exact comparisons and limitations. requirements-cpu-lock.txt records the recoverable package snapshot; the newer general clean-install environment is not a substitute for strict inference equality.

Supply the exact withheld checkpoints and lawful local data externally. Set ALLOCATOR_CHECKPOINT, DETECTOR_CHECKPOINT, VALIDATION_IMAGES, VALIDATION_LABELS, TUNE_OUTPUT and V2_VERIFICATION_OUTPUT to their local paths. The output must be a fresh directory outside this repository. Save the following Python adapter outside the repo and execute it from the public root with the compatible interpreter. This adapter loads the archived calibration metadata; it performs no tune or validation threshold search. The generic evaluator CLI does perform tune selection, so use this adapter for an explicitly frozen operating point.

```python
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

root = Path.cwd().resolve()
sys.path.insert(0, str(root / "scripts"))
import evaluate_rate_allocator_v2 as evaluator

evaluator._load_runtime()
design = json.loads((root / "results/learned_v2/frozen_design.json").read_text())
assert design["baseline_confidence"] == 0.12
assert design["receiver_confidence"] == 0.14500000000000002
assert design["calibration"]["chosen"]["threshold"] == design["receiver_confidence"]
checkpoint = Path(os.environ["ALLOCATOR_CHECKPOINT"]).resolve()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == design["checkpoint_sha256"]
output = Path(os.environ["V2_VERIFICATION_OUTPUT"]).resolve()
assert output != root and root not in output.parents

# Orchestration only: return the stored calibration, without threshold search.
def frozen_calibration(tune_dir, minimum_precision_delta=-0.01):
    return copy.deepcopy(design["calibration"])

evaluator.calibrate_from_tune = frozen_calibration
evaluator.run(SimpleNamespace(
    checkpoint=checkpoint,
    tune=Path(os.environ["TUNE_OUTPUT"]),
    model=Path(os.environ["DETECTOR_CHECKPOINT"]),
    images=Path(os.environ["VALIDATION_IMAGES"]),
    labels=Path(os.environ["VALIDATION_LABELS"]),
    all=True, count=548, jpeg=False, threads=4, output=output,
))
```

The public evaluator checks the detector hash against the allocator split and rejects sequence overlap. For the known legacy checkpoint it explicitly resolves truncated_6sigma; do not select historical_full_gaussian for V2. The frozen thresholds are baseline 0.12 and learned 0.14500000000000002; the fixed-0.12 learned comparison is secondary. AP uses confidence-floor 0.001 post-NMS detections, not operating-threshold-truncated predictions. If any result differs, inspect the dependency inventory and affected records; do not tune thresholds or change algorithms.

After the one full run, sequence-bootstrap commands remain portable; replace the generic external path below with the same V2_VERIFICATION_OUTPUT directory:

```sh
python scripts/bootstrap_saved_detections.py EXTERNAL_V2_OUTPUT --left learned --right baseline --repeats 4000 --seed 731
python scripts/bootstrap_saved_detections.py EXTERNAL_V2_OUTPUT --left learned_fixed012 --right baseline --repeats 4000 --seed 731
```

All per-image outputs, samples, detections/GT and raw heads remain external and excluded from Git. CLAIMS.md and retained results/ must not be overwritten. The complete retained unit suite currently has 31 passing tests after semantics regression coverage; no source/test change was needed for V2 verification.

## Aggregate plotting only

```sh
python -m pip install -e '.[plots]'
python scripts/plot_results.py --output outputs/release_results.png
```

This uses only the included aggregate summaries, with separate panels for distinct comparisons. It performs no model inference. Model provenance, dataset rights, environment availability and clean tests remain prerequisites for broader validation. No full experiment was run while finalizing this scope.

## Sanitized allocator candidate

The model-state export and loader are described in [MODEL_PROVENANCE.md](../MODEL_PROVENANCE.md). They retain exact allocator tensors/normalization with new serialization hashes. No full research rerun is needed to remove private metadata. A data-free comparison verifies identical forward outputs and quality scores; a temporary compatible inference wrapper can be materialized outside Git for the unchanged evaluator. Its derived hash must not be represented as the original frozen hash. Distribution remains subject to final review, and the exact detector is still withheld.
