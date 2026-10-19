# UAV Human-Detection Task-Aware Compression

Research code for retaining person-detection information when UAV imagery must be compressed. Risk-Aware allocates JPEG quality around detector proposals and their context. Learned Allocation V2 predicts tile-quality residuals while keeping the detector/codec fixed and projecting allocations onto actual encoded-byte budgets.

## Status

This is the final local v0.1.0 research release candidate, prepared for AGPL-3.0 subject to ownership/integration review. **The code and evaluation pipeline are public; the exact detector checkpoint is not currently redistributed**, pending dataset/model redistribution review. Independent real-data reproduction needs lawful local data and the matching detector. A sanitized allocator candidate is prepared separately from Git. [Model policy](MODEL_PROVENANCE.md), [release notes](RELEASE_NOTES_v0.1.0.md).

## Quick start

Use Python 3.10 or newer in an activated fresh environment, from the repository root:

```sh
python -m pip install .
python -m task_aware_compression.cli demo --detector heuristic --samples 6 --output outputs/demo
python -m unittest discover -s tests -v
```

The demo needs no VisDrone files, downloaded model or private checkpoint. PyTorch-dependent tests are optional with a base install. To run the complete 35-test suite, install CPU PyTorch first:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

All 35 tests pass locally with PyTorch installed. [CI](.github/workflows/tests.yml) runs installation, the full unit suite and a synthetic smoke test without VisDrone or a detector. Generated outputs are ignored. For training/evaluation extras and exact frozen configurations, see [reproduction](docs/REPRODUCE.md).

## Main results

[CLAIMS.md](CLAIMS.md) defines the frozen metrics and interpretation.

| Comparison | Public headline |
|---|---|
| Risk-Aware vs per-image size-matched JPEG, 548 validation images | **+2.62 pp mean per-image Recall** |
| Learned V2 vs its baseline, tuned operating point | **+1.78 pp AP@0.5 and +3.39 pp Precision at similar Recall and slightly lower BPP** |

Learned V2's numerical validation reproduces exactly. It is a separate result and does not universally replace Risk-Aware. Its tuned Recall interval includes zero. The separate historical +3.15 pp target-level Recall aggregation is not the Risk-Aware headline. Exact values, method versions and reproducibility qualifications are in [results](docs/RESULTS.md), [evaluation](docs/EVALUATION.md) and [verification](docs/REAL_DATA_RELEASE_VERIFICATION.md).

## Repository structure

```text
src/task_aware_compression/  codec, importance maps, allocator, metrics and CLI
scripts/                   dataset conversion, V2, metrics, plotting and safe export
tests/                    dataset-free core, method-version and allocator tests
configs/                   portable dataset and frozen baseline configuration
results/                   compact aggregate evidence, frozen configs and split IDs
models/                    checkpoint hashes and provenance, no weights
docs/                      method, evaluation, reproduction and release audits
.github/workflows/         data-free CI
```

## Reproduction entry points

- Synthetic execution: the quick start above.
- Dataset preparation: [DATASET.md](DATASET.md) and the retained conversion scripts after official acquisition.
- Risk-Aware: CLI workflow with explicit historical_full_gaussian and dense Q1-Q100 JPEG matching.
- Learned V2: training/evaluation scripts with truncated_6sigma; use the frozen-calibration recipe for exact checking without retuning.
- Metrics: strict_evaluation.py and bootstrap_saved_detections.py on private regenerated outputs.
- Allocator export/load: export_allocator_release.py; [artifact format and provenance](MODEL_PROVENANCE.md).
- Aggregate plotting: plot_results.py with the optional plots extra.

[Reproduction instructions](docs/REPRODUCE.md) specify external prerequisites. Default arguments are not a substitute for the frozen experiment recipe.

## Data and limitations

Acquire VisDrone separately. Images, labels, annotations, crops and GT-containing detections are excluded. Categories 1 and 2 are merged into one human/person task. Validation was reused during research; it is not an untouched final test set. JPEG size matching is approximate. CPU timing is not evidence of real-time UAV deployment. **The system has not been deployed in real search-and-rescue operations.** [Full limitations](docs/LIMITATIONS.md).

## Citation and license

Use [CITATION.cff](CITATION.cff) for the project title/version and collective contributor attribution; no paper, individual author list or DOI is invented. Cite VisDrone and external work as listed in [third-party notices](THIRD_PARTY_NOTICES.md). [LICENSE](LICENSE) is the standard GNU AGPL v3 text, subject to final first-party ownership/integration review. It does not grant rights over external data or models. [Licensing audit](LICENSE_AUDIT.md).
