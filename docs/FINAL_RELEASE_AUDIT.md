# Final v0.1.0 local release audit

Date: 2026-10-18. Status: **local candidate prepared; technical checks PASS; ownership/distribution sign-off pending**. This audit does not publish a repository, grant new rights or introduce a research result.

## Frozen scientific scope

Risk-Aware: historical_full_gaussian, +2.62 pp mean per-image Recall on 548 images / 13,969 targets. Release reproduction has 548/548 encoded byte-count matches and 547/548 current-stack detection outcomes; the single Ultralytics IoU-boundary case is explained in [evaluation](EVALUATION.md). Learned V2 retains truncated_6sigma and reproduces its frozen numerical results exactly, including AP +1.782048 pp, Precision +3.390162 pp, tuned Recall +0.071587 pp, BPP -0.001852240 and 548/548 budget checks. [Full verification](REAL_DATA_RELEASE_VERIFICATION.md).

README uses rounded headlines, detailed exact values remain in [results](RESULTS.md), and CLAIMS.md is byte-identical to the frozen version. Scientific package source, existing tests, configs, result summaries and the unmodified LICENSE retain their pre-finalization hashes. New code consists of a model-state export/load utility and four data-free integrity regressions; no scientific algorithm changed. No new research experiment was run.

## Final checks

| Check | Result |
|---|---|
| Non-editable package build/install and installed-path import | PASS |
| Complete retained and new unit suite | **35 passed, 0 failed, 0 errors, 0 skipped** |
| README six-sample heuristic demo | PASS |
| CI two-sample smoke demo and console help | PASS |
| Eight public script help entry points | PASS |
| CI-equivalent tests/demo without Ultralytics, torchvision, OpenCV or Matplotlib imports | PASS |
| CI YAML and data-free scope | PASS; GitHub runner execution NOT RUN |
| CITATION.cff | PASS: official CFF 1.2.0 schema via cffconvert 2.0.0 |
| Internal Markdown links | PASS; checked locally, external links not fetched |
| Chinese public-facing text | None |
| Concrete absolute/private paths or contributor usernames | None |
| Secret/token/private-key patterns | None found |
| Dataset images/labels/GT detections, withheld weights, compiled binaries | None in candidate |
| Files above 10 MB | None |
| LICENSE and THIRD_PARTY_NOTICES.md | Present |
| All frozen claim values/source/result hashes | Unchanged |
| Safe allocator state/output identity and deterministic export | PASS |

Test runtime was 2.696 seconds, excluding interpreter startup. Validation ran on Windows/Python 3.12.14 using a fresh external package environment with dependencies reused from the previously validated external clean installation. The package was built non-editably from an external staging copy and imported from its installed site-packages, not the checkout or internal archive environment. All logs, environments and generated demo outputs remain external. This is a final package check, not a new network-only dependency-resolution test on every platform.

The CPU PyTorch runtime prints its known Visual C++ Redistributable notice; all imports/tests/demo checks succeed. No installer was run. Initial external environment/tool setup issues were resolved outside the repository; no product/scientific source change was needed. The official citation validator was installed into an external validation-tools directory from PyPI. No GitHub connection or Git operation occurred.

## CI

[tests.yml](../.github/workflows/tests.yml) uses Python 3.12 and CPU PyTorch, installs the package, runs unittest discovery and a heuristic demo, and checks CLI/export help. It requires no VisDrone files, detector, allocator asset, private token or secret. Local equivalent tests pass with optional vision-library imports deliberately unavailable. The remote workflow will only run after a separately authorized repository action; it was not contacted here.

## Sanitized allocator asset

The external candidate comprises data-only allocator-v2-v0.1.0.npz, allocator-v2-v0.1.0.json and its own SHA256SUMS. Eight finite float32 tensors include the feature normalization buffers; JSON supplies architecture, feature order, normalization convention, quality offsets/ranges, codec/projection parameters, thresholds and method version. No Windows/local path, username, dataset path, optimizer state or training log is exported.

- Original checkpoint SHA256: e28024b48c26d378f3378d40d24ef266f68a4f61ea37ae452e977b721a90bf45.
- New NPZ SHA256: 5089346d903808f21433aa991009680aa5b643f2239fb2892f25ce30aa105f91.
- New JSON SHA256: 24bc8d558f2142346d5d98c6609aa57de4dd8dc43e43f49ff128dd3ce4e2c8e6.
- Original/exported tensors and normalization: exactly equal.
- Forward outputs and quality scores: exactly equal for 329 deterministic synthetic feature rows across four batch sizes.
- Repeated serialization: identical bytes; ZIP entries use fixed ordering/timestamps.
- Temporary inference-wrapper materialization: weights-only loading, tensor identity and explicit V2 mode resolution pass without research evaluation.

The checksums intentionally differ from the source checkpoint. [Model provenance](../MODEL_PROVENANCE.md) describes the loader, original relationship and unchanged-evaluator adapter. Asset files remain outside Git pending final distribution review. Detector weights and the original metadata-rich allocator are absent from both the Git candidate and sanitized bundle.

## Cleanup and inventory

Removed from the public candidate and preserved externally: build/, outputs/ (including outputs/clean-validation/), tests/__pycache__/, src/task_aware_compression/__pycache__/, src/uav_task_aware_compression.egg-info/ and the obsolete models/SHA256SUMS source-identifier list. No private _prepare_release.py or _finalize_scope.py helper existed in the final source inventory. Obsolete draft headings were finalized; useful historical licensing/reproduction audits remain clearly dated.

The candidate has 72 files. RELEASE_MANIFEST.md lists each file's purpose. Root SHA256SUMS hashes every candidate file except itself; external allocator assets have separate checksums. It is not a list of unavailable source files. No cache, build output, generated media, checkpoint, third-party source tree or bundled runtime remains under the public root. Final exact size and checksum verification are recorded by the external audit and final delivery report.

## Remaining sign-off

Final author/IP and integration-license review, confirmation of collective citation attribution, and any allocator-asset redistribution clearance remain pending. Detector withholding is intentional policy; local-supply/hash instructions clearly distinguish public pipeline code from unavailable exact detector weights. Original V2 process inventory and untested platforms remain documented limitations. These are not unresolved numerical mismatches. [Open issues](../OPEN_ISSUES_BEFORE_RELEASE.md).
