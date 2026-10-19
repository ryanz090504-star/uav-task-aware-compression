# Minimal v0.1.0 portability audit

The repository is English-only and contains no concrete personal machine paths in public configuration. The standard LICENSE contains its upstream copyright/contact text; these are not contributor secrets. Data/model paths in configs are repository-relative, and frozen manifests retain historical relative locations as provenance.

No file over 10 MB, model checkpoint, binary figure, compiled artifact, image, annotation, raw array, per-image result CSV or detection JSONL is part of this candidate. Numerical results are aggregates. Frozen split manifests include permitted dataset identifiers, not boxes/labels. The former supplementary-asset manifest has been removed: GT-containing detections are not v0.1.0 release assets.

Required V2 helpers were extracted unchanged into scripts/allocation_utils.py. The two V2 imports now target that module; training/evaluation algorithms are unchanged. The old pilot trainer, incomplete detector trainer, defective report generator, pilot plotter and unused preprocessing modules/tests are excluded. scripts/plot_results.py reads only retained aggregate summaries; it does not run inference.

Python >=3.10, NumPy and Pillow support the synthetic/core path. Optional learned execution uses PyTorch, Ultralytics and OpenCV; plotting uses Matplotlib. The historical CPU requirement files remain environment records, not a verified cross-platform lock. No runtime binaries are bundled. Optional HEVC requires external ffmpeg/libx265 and is disabled in the documented baseline command.

All retained Python files are checked for syntax and local import targets, and public runtime imports were exercised in fresh environments. Extracted helper functions have identical ASTs to their pre-cleanup implementations. Retained numerical artifacts are hash-checked against their pre-cleanup values. The new baseline paired summary is an aggregate of existing records, not a new experiment. LICENSE matches the downloaded standard GNU text hash recorded in RELEASE_MANIFEST.md.

Fresh Windows/Python 3.12 installation, all 22 retained tests, synthetic CLI, aggregate plotting and a non-editable wheel smoke test passed; see docs/CLEAN_ENVIRONMENT_TEST.md. Other platforms remain untested. Full experiments are not run. Exact historical reproduction is blocked by withheld checkpoints and raw evidence; no model/data acquisition beyond official dataset instructions is promised. AGPL preparation remains subject to final ownership/integration-license review. See OPEN_ISSUES_BEFORE_RELEASE.md and docs/LIMITATIONS.md.

Validation added only CLI/input/dependency preflight and delayed optional imports. Scientific algorithms, tests, CLAIMS.md and retained numerical results are unchanged. Local outputs/build/editable metadata are ignored scratch artifacts and may contain machine paths; the release inventory excludes them.

## Final local candidate update (2026-10-18)

See [FINAL_RELEASE_AUDIT.md](docs/FINAL_RELEASE_AUDIT.md) for current checks. Generated outputs, build copies, caches and private validation logs are no longer inside the candidate. The sanitized allocator NPZ/JSON is an external, separately checksummed candidate; no detector or model binary is included in Git. The current full suite has 35 passing tests; historical test counts above refer to earlier stages. Concrete absolute/private paths and Chinese text are absent. Escaped relative paths in frozen result metadata are historical provenance identifiers, not executable machine-specific configs, and remain byte-identical.
