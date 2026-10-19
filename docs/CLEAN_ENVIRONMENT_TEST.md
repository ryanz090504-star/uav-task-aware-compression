# Clean-environment validation — v0.1.0

Date: 2026-10-17. Public checkout only. No internal archive, old project virtual environment, VisDrone data, withheld checkpoints, full research experiments, Git operations or GitHub connection were used. PyPI installation was authorized.

## Outcome

PASS: fresh editable installation, package/script imports, complete retained tests, deterministic synthetic demo, all CLI help, actionable missing-input errors, aggregate plotting and a non-editable wheel smoke test. Real-data prerequisite failure is not successful research reproduction.

## Environment and commands

Python: 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)].
Platform: Windows-11-10.0.26200-SP0; Windows 11 x86-64.
Primary pip: 26.2.1 (upgraded from ensurepip 25.0.1).

Virtual environments were created outside the repo using the available standalone CPython interpreter, not any research environment. No system-site packages or user PYTHONPATH were enabled. An exploratory environment was followed by an entirely fresh final environment after release-interface fixes. A third independent environment tested the built wheel. Wheel caching was permitted, but installed environments were not reused. This is Python package isolation on the existing Windows host, not a new Windows installation.

```text
python -m venv <outside-repo>/final
<final-python> -m pip install --upgrade pip
<final-python> -m pip install -e .
<final-python> -m pip install -e '.[learned,plots]'
<final-python> -m pip install -e '.[yolo,plots]'
<final-python> -m pip install -e '.[plots]'
<final-python> -m pip check
<final-python> -m pip list --format=json
<validation-python> -m pip wheel . --no-deps --wheel-dir outputs/clean-validation/wheels
python -m venv <outside-repo>/wheel
<wheel-python> -m pip install <built-wheel>
```

All install commands succeeded. pip check reported no broken requirements. Editable installation logs had no warnings/errors. The wheel environment retained its bootstrap pip 25.0.1 and printed the normal available-update notice. The initial base install contained only pip, NumPy, Pillow and the project before extras were installed.

## Unit tests

| Profile | Passed | Failed | Skipped | Unittest runtime |
|---|---:|---:|---:|---:|
| Base dependencies | 16 | 0 | 6 existing optional learned tests | 0.290 s |
| Learned/plots extras | 22 | 0 | 0 | 0.182 s |

Full discovery was python -m unittest discover -s tests -v. The full subprocess including interpreter/import overhead took 5.339 s. Tests were not changed or selectively disabled. PyTorch was installed so the final complete suite had no skips. Base-profile skips are not counted as full-suite success.

## Imports and entry points

- The package and all core modules imported from a directory outside the checkout without PYTHONPATH.
- After extras, all 14 public package modules and all eight script/helper modules imported. Both learned runtime import loaders executed, without a model.
- Root CLI, demo, compress, benchmark and workflow help passed. The installed uav-compress console help worked outside the checkout.
- All seven executable script help commands worked even in the base environment. allocation_utils.py is an import-only helper.
- Nine focused absent-input cases returned status 2 with actionable instructions: both converters, dataset workflow, explicit YOLO demo, missing input image, V2 train/evaluate, strict metrics and detection bootstrap.
- In the base-only wheel environment, missing Matplotlib and Ultralytics also produced clear installation instructions and status 2. For the latter check, an existing harmless text path exercised only the absent-package branch; no checkpoint was loaded.
- Converters now reject missing directories before writing empty outputs. Detector construction requires an existing local model instead of triggering weight acquisition.
- No source import relied on an excluded pilot/internal file. Public checkout-relative script discovery remains intentional.

## Synthetic execution and packaging

The exact README six-image heuristic demo completed with base dependencies and with extras. It produced 48 records, a summary JSON, experiment_notes.md, uniform JPEG/adaptive payloads, previews and importance maps. Its notes explicitly identify synthetic engineering data; these outputs do not revalidate VisDrone claims.

A second run with the same seed produced identical bytes for all 96 PNG/JPEG/TAC image/payload files. Timings varied as expected. Uniform and heuristic-adaptive single-image compress commands also passed on a generated synthetic original.

The aggregate plot command created outputs/release_results.png. A request to write inside frozen results/ was correctly rejected. No generated plot is included as release evidence.

The normal wheel built successfully with all 14 runtime modules, no excluded prototypes, no model/data artifact and a byte-identical LICENSE. It was installed in a separate fresh environment; from a neutral working directory, imports resolved to site-packages with no checkout path in sys.path. Console help and a one-image heuristic demo (eight records) passed. scripts/, results/ and docs/ are checkout assets, not hidden dependencies of the installed demo/core package. Historical CPU lockfiles were not used or claimed to recreate the old research environment.

## Every documented command

Checked 17 command occurrences (14 unique commands) in README.md, docs/REPRODUCE.md and DATASET.md. Successful missing-input handling below verifies parsing/diagnostics only, not data preparation or an experiment.

| Document | Command | Result |
|---|---|---|
| README.md | python -m pip install -e . | PASS, executed |
| README.md | python -m task_aware_compression.cli demo --detector heuristic --samples 6 --output outputs/demo | PASS, executed |
| README.md | python -m unittest discover -s tests -v | PASS, executed |
| docs/REPRODUCE.md | python -m pip install -e . | PASS, executed |
| docs/REPRODUCE.md | python -m task_aware_compression.cli demo --detector heuristic --samples 6 --output outputs/demo | PASS, executed |
| docs/REPRODUCE.md | python -m unittest discover -s tests -v | PASS, executed |
| docs/REPRODUCE.md | python -m pip install -e '.[yolo,plots]' | PASS, executed |
| docs/REPRODUCE.md | python -m task_aware_compression.cli workflow --images data/VisDrone/images/val --labels data/VisDrone/labels/val --model models/visdrone_yolov8n_1280_b18_s17_best.pt --detector yolo --confidence 0.12 --device cpu --limit 548 --tile-size 256 --no-hevc --visual-samples 0 --detection-batch-size 8 --tile-workers 8 --output outputs/baseline_local | PASS, prerequisite-only failure; full run blocked |
| docs/REPRODUCE.md | python scripts/strict_evaluation.py outputs/baseline_local/benchmark/records.csv --output outputs/baseline_local/strict_evaluation.json | PASS, prerequisite-only failure; full run blocked |
| docs/REPRODUCE.md | python -m pip install -e '.[learned,plots]' | PASS, executed |
| docs/REPRODUCE.md | python scripts/train_rate_allocator_v2.py --model models/visdrone_yolov8n_1280_b18_s17_best.pt --train-count 48 --tune-count 32 --eval-count 80 --size 1280 --epochs 8 --lr 0.0005 --temperature 0.5 --distill-weight 1.0 --identity-weight 0.02 --rate-weight 1 --minimum-ratio 0.98 --confidence 0.12 --threads 4 --device cpu --seed 731 --balanced --output outputs/learned_v2_local | PASS, prerequisite-only failure; full run blocked |
| docs/REPRODUCE.md | python scripts/evaluate_rate_allocator_v2.py --checkpoint outputs/learned_v2_local/rate_allocator.pt --tune outputs/learned_v2_local/tune --model models/visdrone_yolov8n_1280_b18_s17_best.pt --all --threads 4 --output outputs/learned_v2_eval_local | PASS, prerequisite-only failure; full run blocked |
| docs/REPRODUCE.md | python scripts/bootstrap_saved_detections.py outputs/learned_v2_eval_local --left learned --right baseline | PASS, prerequisite-only failure; full run blocked |
| docs/REPRODUCE.md | python -m pip install -e '.[plots]' | PASS, executed |
| docs/REPRODUCE.md | python scripts/plot_results.py --output outputs/release_results.png | PASS, executed |
| DATASET.md | python scripts/convert_visdrone_train.py | PASS, prerequisite-only failure; full run blocked |
| DATASET.md | python scripts/convert_visdrone_val.py | PASS, prerequisite-only failure; full run blocked |

No command has a machine-specific path. Single-quoted extras work in PowerShell/POSIX shells; cmd.exe needs double quotes. Linux/macOS execution was not tested. Seven documented real-data/preparation/metric command occurrences failed clearly at missing local prerequisites. No VisDrone data was read or converted.

## Failures and warnings

- Initial venv bootstrapping encountered sandbox TEMP permission errors in ensurepip. A dedicated writable temporary directory beside the external validation environments resolved this host issue; no package change was required.
- Interface review identified absent converter help/preflight, learned imports running before argument parsing, and unhelpful missing-file failures. The fixes below preceded a complete rebuild/retest in a new environment.
- PyTorch printed a missing Microsoft Visual C++ Redistributable warning and possible DLL-load-failure notice. Imports and all tests still succeeded. No native installer was downloaded or run. Other Windows hosts may need the runtime expected by their wheel.
- Ultralytics printed a first-run settings message. Settings/Matplotlib caches were redirected into ignored validation outputs; no weights were loaded.
- No scientific algorithms, retained tests, CLAIMS.md or research numerical artifacts were changed. Hashes of all claims/tests/results match the pre-fix snapshot.

## Fixes made

| File | Release-only change |
|---|---|
| src/task_aware_compression/cli.py | Validate dataset/image paths; translate dependency/model failures into CLI diagnostics. |
| src/task_aware_compression/detector.py | Require a local checkpoint before YOLO construction; no automatic acquisition. |
| scripts/convert_visdrone_train.py; convert_visdrone_val.py | Add argparse help and input-directory preflight. Conversion algorithms unchanged. |
| scripts/train_rate_allocator_v2.py; evaluate_rate_allocator_v2.py | Delay optional imports until after parsing/preflight. Scientific computational function ASTs unchanged. |
| scripts/strict_evaluation.py; bootstrap_saved_detections.py | Explain missing private input files before computation. |
| scripts/plot_results.py | Explain the missing plots dependency. |

README.md, docs/REPRODUCE.md and DATASET.md clarify verified versus blocked commands. Validation status and release inventory are updated in OPEN_ISSUES_BEFORE_RELEASE.md, PORTABILITY_AUDIT.md and RELEASE_MANIFEST.md. This report is new. pyproject.toml, tests, CLAIMS.md, LICENSE and numerical research evidence are unchanged.

Scientific package source files cli.py and detector.py did change, but only in input/CLI wrappers; no compression, inference, allocation, matching, metric or bootstrap algorithm changed.

## Actual installed packages

Final primary-environment pip list; these newer resolved versions are distinct from historical research requirements.

| Package | Version |
|---|---|
| anyio | 4.15.1 |
| certifi | 2026.7.22 |
| charset-normalizer | 3.5.2 |
| contourpy | 1.4.0 |
| cycler | 0.12.1 |
| filelock | 4.2.1 |
| fonttools | 4.67.0 |
| fsspec | 2026.9.0 |
| h11 | 0.16.0 |
| httpcore | 1.0.9 |
| httpx | 0.28.1 |
| idna | 3.20 |
| Jinja2 | 3.1.6 |
| kiwisolver | 1.5.1 |
| MarkupSafe | 3.0.4 |
| matplotlib | 3.11.2 |
| mpmath | 1.3.0 |
| networkx | 3.7 |
| numpy | 2.5.4 |
| nvidia-ml-py | 13.615.71 |
| opencv-python | 5.0.0.93 |
| packaging | 26.3 |
| pillow | 13.0.0 |
| pip | 26.2.1 |
| polars | 2.0.0 |
| polars-runtime-32 | 2.0.0 |
| psutil | 7.2.2 |
| pyparsing | 3.3.3 |
| python-dateutil | 2.9.0.post0 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| setuptools | 84.0.0 |
| six | 1.17.0 |
| sympy | 1.14.0 |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| typing_extensions | 4.16.0 |
| uav-task-aware-compression | 0.1.0 |
| ultralytics | 8.4.181 |
| ultralytics-platform | 0.1.90 |
| ultralytics-thop | 2.2.2 |
| urllib3 | 2.8.0 |

## Remaining limitations

- Both checkpoints remain withheld; detector lineage/model rights and lawful local dataset acquisition are still prerequisites for real-data reproduction.
- Full AP/bootstrap needs private regenerated GT-containing outputs. Public summaries alone cannot reconstruct them. Passing tests is not a fresh validation of research claims.
- Final author/IP and integration-license review is pending.
- Only this Windows/CPython combination was exercised, not other Python versions, operating systems, GPU builds or native runtime configurations.
- Ignored outputs/clean-validation/ contains local logs, harnesses, caches and wheels that can contain private machine paths; none is a public release file. Demo outputs and generated editable/build metadata are also ignored. Environment directories are outside the repository.
