# Real-data release verification

> Current release status (2026-10-18): Learned V2 full verification **PASS**, with exact metrics, predictions and raw heads. Risk-Aware is accepted for release with its separately documented current-stack IoU-boundary qualification. Earlier baseline-stage pending/stop statements below are dated history; the V2 follow-up at the end supersedes its NOT RUN status.

Date: 2026-10-17 local / 2026-10-18 UTC. Historical Risk-Aware replay: **FAIL — STOP**. Learned V2 has not been run.

## Method identity

The v0.1 Risk-Aware headline refers to **historical_full_gaussian**, not the later **truncated_6sigma** variant. The initial release check selected the later variant and measured +2.312588910 pp mean per-image Recall. Source drift was discovered during release reproduction: truncating tiny Gaussian tails changes the exact-positive-support texture gate. That measurement belongs to a distinct later variant and is not a failed replay of the historical executable definition.

The present full replay explicitly selects historical_full_gaussian. Both modes remain supported, with validated config names, CLI selection and regression tests. The workflow/benchmark default is historical_full_gaussian; bare CompressionConfig and legacy allocator behavior retain truncated_6sigma. Always specify a mode for a frozen experiment. The TAC header/schema and JPEG parameters were not changed, preserving byte accounting.

CLAIMS.md and every frozen numerical result remain unchanged. The archive is read-only. No Git/GitHub operation occurred.

## Environment and exact state

- Python: 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)].
- Platform: Windows-11-10.0.26200-SP0; CPU inference.
- Resolved scientific packages: pip 26.2.1, numpy 2.5.4, pillow 13.0.0, torch 2.14.1, torchvision 0.29.1, ultralytics 8.4.181, opencv-python 5.0.0.93.
- Canonical release snapshot immediately before replay SHA256: 1bfebb1da2186c85d36fbff039e5db43cbd691eecd2f30a3f339bbaf2cf8cf2e. The full per-file snapshot and package inventory remain external.
- Snapshot encoding: sorted relative POSIX path, TAB, byte size, TAB, file SHA256, LF; exclude ignored outputs/caches/build products/editable metadata.
- Protocol: configs/risk_aware_v0.1.0.json. This newly versioned protocol is separate from retained frozen result files.
- Original dependency drift and shared detector channel-convention limitations are documented in RISK_AWARE_REPRODUCTION_INVESTIGATION.md.

## Prerequisite hashes

| Prerequisite | SHA256 |
|---|---|
| detector: visdrone_yolov8n_1280_b18_s17_best.pt | 263cfd00bbd54283ad6982d79b04e7f9306d3d63c82a3264f500fb81be25426a |
| allocator: rate_allocator.pt | e28024b48c26d378f3378d40d24ef266f68a4f61ea37ae452e977b721a90bf45 |
| Validation images manifest (548 files) | 7f2031b7a0ca3afd4f46f6dc830db47152a0834922e446082b4b8e0d27fbeca4 |
| Validation labels manifest (548 files) | be52d59b9ab3f762f55ae7fb07b23e98137057261f17c3e6a39e9d3211ce4e6e |
| Original validation ZIP | abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9 |
| Frozen historical records.csv (withheld) | 8c424482c06baabc71f28e18c4bd35f1a687192dbe308d09271c4f457fa9d9e6 |
| Replayed records.csv (external, not distributed) | a12a4c0bb2b1ff1bd93e2495d1d25e4a703eb7f16a7ee5223d0560f055005ad9 |

Data/model identities were rechecked before replay. There are 548 images, 548 converted label files and 13,969 merged human/person targets. The earlier ZIP crosscheck found zero image-byte or label-conversion mismatches. Directory hashes identify sorted per-file manifests, not ZIP bytes. Optional occlusion/truncation metadata sidecars remain absent; optional strata are unavailable.

Private absolute prerequisite paths and expanded arguments are recorded externally. No data/checkpoints were copied into this public repo.

## Exact replay command

Python is the validated external clean interpreter. Aliases point to the original external inputs and a fresh external artifact directory.

~~~powershell
python -m task_aware_compression.cli workflow --images "$VALIDATION_IMAGES" --labels "$VALIDATION_LABELS" --model "$DETECTOR" --detector yolo --confidence 0.12 --device cpu --limit 548 --tile-size 256 --no-hevc --visual-samples 0 --detection-batch-size 8 --tile-workers 8 --importance-semantics historical_full_gaussian --output "$ARTIFACTS/baseline"
~~~

This one-process public command is the reproduction equivalent. The actual verification resumed four contiguous ranges of 137 IDs through resume_per_image_shards.py and unchanged public per-image evaluation statements. Nineteen complete saved evaluations were recovered; only 529 images were evaluated on resume. Per-shard final aggregation was omitted. All 9,316 records were merged in original 548-image order before final statistics. The aborted eight-image single-process attempt contributes no records. Exact manager/worker commands are retained in resume-command.private.json.

~~~powershell
python "$ARTIFACTS/resume_per_image_shards.py" "$ARTIFACTS"
python "$ARTIFACTS/resume_per_image_shards.py" "$ARTIFACTS" --worker 1
python "$ARTIFACTS/resume_per_image_shards.py" "$ARTIFACTS" --worker 2
python "$ARTIFACTS/resume_per_image_shards.py" "$ARTIFACTS" --worker 3
python "$ARTIFACTS/resume_per_image_shards.py" "$ARTIFACTS" --worker 4
~~~

The worker commands were launched by the manager, not run twice. The wrapper records a hash of the unchanged public per-image loop and adds only external metadata logging. It retains research controls, Q15/Q35/Q60/Q90 JPEGs, dense Q1..100 pairing, original pending batch order and imgsize1280. Main claim selection is risk_aware_adaptive_adaptive_bg8_fg82_t256 paired with paired_risk_aware_adaptive_adaptive_bg8_fg82_t256. Risk weights 0.58/0.42, context 18, texture 0.12, reference 0.018, mean pooling, max aggregation, smoothstep and zero offsets match the historical settings. Header overhead counts toward bytes. Mode identity is recorded in the experiment summary/config, without introducing new container bytes.

External commands also ran preflight.py, compare_baseline.py and compare_extended.py using the same clean interpreter. Their exact source and logs remain in the external replay directory. They join the frozen and regenerated per-image records directly; no frozen result was overwritten.

## Historical replay comparison

All 548 images completed, producing 9,316 records (17 settings per image). Target count: 13969. Detector hash: exact match.

- Mean per-image adaptive Recall: 62.234050659%.
- Mean per-image matched JPEG Recall: 59.611846993%.
- Primary mean paired Recall delta: **+2.622203666 pp**.
- Separate pooled target-level Recall delta: **+3.156990479 pp**. This is the historical +3.15 pp aggregation, not the +2.62 pp headline.

| Metric | Frozen | Reproduced | Difference | PASS/FAIL |
|---|---:|---:|---:|---|
| image_count | 548 | 548 | +0 | PASS |
| target_count | 13969 | 13969 | +0 | PASS |
| mean_adaptive_recall | 0.62229248508 | 0.622340506593 | +4.80215136381e-05 | FAIL |
| mean_jpeg_recall | 0.596118469929 | 0.596118469929 | +0 | PASS |
| mean_delta_pp | 2.61740151509 | 2.62220366645 | +0.00480215136381 | FAIL |
| micro_delta_pp | 3.14983177035 | 3.15699047892 | +0.00715870856897 | FAIL |
| mean_adaptive_bpp | 0.408289765659 | 0.408289765659 | +0 | PASS |
| mean_jpeg_bpp | 0.408081146431 | 0.408081146431 | +0 | PASS |
| byte_error_median | 0.0132182525675 | 0.0132182525675 | +0 | PASS |
| byte_error_p95 | 0.0495407251615 | 0.0495407251615 | +0 | PASS |
| byte_error_max | 0.0598755832037 | 0.0598755832037 | +0 | PASS |
| adaptive_true_positive | 8856 | 8857 | +1 | FAIL |
| adaptive_false_positive | 10782 | 10781 | -1 | FAIL |
| adaptive_false_negative | 5113 | 5112 | -1 | FAIL |
| uniform_paired_true_positive | 8416 | 8416 | +0 | PASS |
| uniform_paired_false_positive | 9722 | 9722 | +0 | PASS |
| uniform_paired_false_negative | 5553 | 5553 | +0 | PASS |
| adaptive_target_level_recall | 0.633975230868 | 0.634046817954 | +7.15870856897e-05 | FAIL |
| jpeg_target_level_recall | 0.602476913165 | 0.602476913165 | +0 | PASS |
| detector_SHA256 | 263cfd00bbd54283ad6982d79b04e7f9306d3d63c82a3264f500fb81be25426a | 263cfd00bbd54283ad6982d79b04e7f9306d3d63c82a3264f500fb81be25426a | identical | PASS |
| importance_mode | historical_full_gaussian | historical_full_gaussian | identical | PASS |

Recall values without _pp are fractions. BPP is bits/pixel; byte-error values are fractions of adaptive bytes. Counts, hashes, output bytes and matched JPEG qualities require exact equality. Derived aggregate scalars allow only 1e-10 absolute arithmetic roundoff; confidence intervals are not acceptance tolerances. The comparison uses exact frozen CSV values, not merely two-decimal headline rounding.

Per-image claim-field comparisons:

- adaptive_file_size_bytes: 0/548 changed.
- uniform_paired_file_size_bytes: 0/548 changed.
- adaptive_paired_uniform_quality: 0/548 changed.
- uniform_paired_paired_uniform_quality: 0/548 changed.
- adaptive_true_positive: 1/548 changed.
- uniform_paired_true_positive: 0/548 changed.
- adaptive_false_positive: 1/548 changed.
- uniform_paired_false_positive: 0/548 changed.
- adaptive_false_negative: 1/548 changed.
- uniform_paired_false_negative: 0/548 changed.
- adaptive_recall: 1/548 changed.
- uniform_paired_recall: 0/548 changed.

Order of all record keys is identical: True. Every compared claim field is identical per image: False. Floating detector coordinates/AP and timing are not asserted bitwise identical across dependency versions.

## Remaining discrepancy and stop decision

The historical mode reproduces all 548 adaptive byte counts, all 548 matched-JPEG byte counts and all 548 matched-JPEG quality selections exactly. Adaptive Recall/TP/FP/FN match on 547/548 images; matched-JPEG Recall/counts match on 548/548. On one image the reproduced adaptive TP is one higher, FP one lower and FN one lower. The external remaining-discrepancy.private.csv identifies the image and both records. The resulting differences are +0.004802151364 pp in mean per-image gain and +0.007158708569 pp in pooled target-level gain. Both primary gains round to +2.62 pp, but integer count differences cannot be accepted as floating-point aggregation roundoff. Exact historical certification therefore remains blocked; no wider tolerance was adopted after seeing the outcome.

Equal encoded byte counts do not prove identical decoded pixels or detector inputs. A detector numerical/matching boundary difference is possible, but its cause is unconfirmed. No further inference, dependency experiment or scientific modification was attempted after this discrepancy. A focused comparison of the single affected image's encoded pixels and detector/matching intermediates is the next proposed investigation, not work performed here.

Merged normal-approximation paired-mean CI (Recall fraction): [0.017276139052501777, 0.03516793427658781]; frozen: [0.01722861803986741, 0.03511941226194596]. These intervals also differ and are not acceptance tolerances.

## Tile-quality and bitstream evidence limits

The frozen CSV does not contain tile-quality lists; frozen TAC streams are not retained with the numerical records. Direct exact per-tile quality matches cannot be counted: NOT VERIFIABLE, rather than zero mismatches or 548 inferred matches. The resumed evaluation retained quality lists for 529 images; the earlier 19 completed evaluations did not retain them. Existing synthetic tile-quality regressions are a different check. Equal file_size_bytes establishes equal byte counts, not bit-for-bit stream identity. A read-only archive search covered all non-runtime TAC files, the final result JSONs and their two handoff copies, and project_bundle.tar.gz member names. No reliable historical machine-readable tile manifest or TAC stream was found. The final result directory retains raster quality visualizations, which cannot establish exact tile lists. The 32 TAC files found elsewhere belong to synthetic/demo or later learned-allocation experiments and were excluded from historical comparisons. Exact relative paths, JSON hashes and search inventory are retained externally in tile-artifact-search.private.json. The frozen CSV SHA256 above anchors the accepted numerical comparison.

## Byte matching

Mean adaptive BPP 0.408289765659; matched JPEG BPP 0.408081146431. Absolute per-image JPEG/adaptive size mismatch: median 1.321825257%, P95 4.954072516%, maximum 5.987558320%. This is nearest-size discrete JPEG pairing, not exact equal bytes.

The original later-variant run had different adaptive/matched BPP and different size residuals because its tile allocation differed. It must not be conflated with this historical replay.

## Learned V1/V2 provenance and gate

- Historical Risk-Aware semantics: confirmed by archived full-frame source, its associated frozen records, controlled replay and exact full-dataset byte accounting above. The full-dataset detection outcomes still differ on one image.
- Exact date/commit when truncation entered: **unconfirmed**. The full-frame source is preserved in the bundle/handoff; the current truncated file has preserved mtime 2026-09-28. A file timestamp does not establish its actual edit/commit date.
- V1 training: truncated semantics, supported by all 32 recorded training-image feature means matching its checkpoint within 2.98e-8; full-image means differ by up to 0.0202194. V1 evaluation baseline bytes match truncated behavior on 80/80.
- V2 training: truncated semantics, supported by exact cached training-feature mean/scale linkage to the frozen checkpoint over 48 images and semantic probes. Two probe feature errors are 3.73e-7 and 1.25e-6 for truncated behavior versus about 0.12 for full-image behavior; a third has residual one-quality-level drift in the current environment. These were baseline/provenance probes, not learned evaluation.
- V2 full-validation saved baseline bytes match truncated behavior on 548/548; full-image behavior matches only 74/548.
- V2 training now explicitly selects truncated_6sigma. New checkpoints save their mode. The known frozen legacy checkpoint hash resolves its missing field explicitly to truncated_6sigma; unknown legacy checkpoints lacking a mode are rejected. Explicit valid modes are honored.
- V2 AP/Precision/Recall and budget guarantees remain **NOT RUN / unverified** in the release environment. This baseline replay does not certify them.

## Tests, warnings and runtime

The retained suite plus nine new regressions passes: **31 passed, 0 failed, 0 skipped** (unittest reported 0.205s, excluding interpreter/import startup). Tests cover independent dense references for both modes, positive-support texture amplification, fixed historical/later tile qualities, unchanged container schema, CLI/API selection, invalid-mode rejection and legacy checkpoint resolution. No existing tests were weakened.

~~~powershell
python -m unittest discover -s tests -v
python -m task_aware_compression.cli workflow --help
python -m task_aware_compression.cli demo --samples 2 --seed 7 --output "$ARTIFACTS/synthetic-demo"
~~~

Help and demo smoke checks passed. pytest was not installed in the validated runtime, so the retained standard-library unittest runner was used. An initial test attempt hit a sandbox temporary-directory cleanup permission error; rerunning with TEMP/TMP directed to the external artifact tmp directory passed all 31 tests unchanged.

Initial four-shard start UTC: 2026-10-18T00:30:59.138494+00:00. Finish UTC: 2026-10-18T03:35:16.078122+00:00. Runtime: 3012.313s (50.21min). Process exit: 0.

Reported runtime covers the resumed 529-image evaluation, merge and aggregation only. Earlier interrupted work and idle time are excluded; the overall wall interval is not a serial inference latency measurement. The normal-approximation paired-mean interval was recomputed once on merged data. No bootstrap was run per shard or rerun in this verification.

PyTorch again warned about the Visual C++ Redistributable; inference completed. Offline YOLO mode and external settings/cache/temp directories were used. No installer, dataset, model or remote repository was downloaded.

## Artifacts and limitations

All regenerated images, per-image CSVs, GT-related artifacts, plots, logs, model/data manifests, provenance probes and environment snapshots are external. The public repo contains this sanitized report, protocol/tests/source versioning and documentation only. No large/raw dataset artifact was added.

Validation data were reused during research; this is not an untouched final test set. Categories 1 and 2 are merged into one human/person task. CPU timing is not evidence of real-time UAV deployment. No real search-and-rescue deployment has occurred. Dataset and checkpoint distribution restrictions remain in force.

STOP: remaining historical replay differences prevent certification. No further algorithm changes or V2 execution were performed. CLAIMS.md remains unchanged pending resolution.


## Learned V2 full release verification (2026-10-18)

**Overall: PASS.** The frozen 548-image / 76-sequence / 13,969-target Learned Allocation V2 evaluation reproduces exactly with historical-compatible components. All 1,644 scientific record rows and all 1,644 post-NMS prediction/GT records match. All 1,096 raw-output NPZ files have exactly equal arrays, including raw heads and shape metadata. The four retained sample TAC streams match byte-for-byte. Every headline scalar and both 4,000-sequence-resample bootstrap files match exactly; no nonzero tolerance was needed. Timing values were excluded from equality checks.

This is a separate method/result from the historical full-Gaussian Risk-Aware/JPEG comparison. V2 retains **truncated_6sigma**. CLAIMS.md, numerical evidence and scientific implementation were not edited. No allocator retraining, tune threshold selection or validation retuning occurred.

### V2 environment provenance and compatibility

The exact original V2 launch inventory is **unconfirmed**: its config/design/logs contain CPU and configured thread count, but no immutable complete package/runtime inventory for that launch. The V2 document and artifacts were inspected independently; the Risk-Aware stack was not assumed to be the V2 stack. Archived requirements-cpu-lock.txt and installed package metadata recover the component versions below; archived pyvenv.cfg records Python 3.12.13, without proving that patch version at the V2 launch.

| Component | Recovered verification environment |
|---|---|
| Python | 3.12.14; archived environment snapshot says 3.12.13 |
| Ultralytics | 8.4.129 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| Pillow | 12.3.0 |
| NumPy | 2.5.2 |
| OpenCV | distribution 5.0.0.93; module 5.0.0 |
| Platform/device | Windows-11-10.0.26200-SP0; CPU float32; CUDA unavailable |
| Threads | configured 4, matching frozen config; observed 8 after upstream select_device resets Torch threads |
| Determinism | actual deterministic-algorithm flag False; no flag adjustment |

Historical packages were copied read-only into a separate external environment based on the validated clean interpreter; the original internal interpreter was never executed. Uncopied dependencies fall back to the validated external environment. No package download or GitHub connection occurred. Before full evaluation, four V2-specific saved baseline/learned raw heads, post-NMS lists and decoded sample PNGs matched exactly. The subsequent 1,096/1,096 raw-head equality confirms compatibility for the complete replay, rather than uniquely proving the original launch inventory. Original Python patch/OS/process state remain provenance limitations; no V2 output drift was observed in the compatible stack. The newer general clean-install stack was not used for this V2 full replay and is not silently certified as numerically equivalent.

### Prerequisites, frozen design and exact state

The exact original local dataset and both withheld checkpoints were supplied externally. Detector and allocator hashes pass before and after the run. Public/archived frozen_design.json, training split, training log, tune report and result report hashes match. The public training config has sanitized paths; every non-path config value matches the archived checkpoint/config. The checkpoint compression config is resolved explicitly to truncated_6sigma using the known legacy artifact hash, preserving its original map behavior.

| Prerequisite | SHA256 |
|---|---|
| detector checkpoint: visdrone_yolov8n_1280_b18_s17_best.pt | 263cfd00bbd54283ad6982d79b04e7f9306d3d63c82a3264f500fb81be25426a |
| allocator checkpoint: rate_allocator.pt | e28024b48c26d378f3378d40d24ef266f68a4f61ea37ae452e977b721a90bf45 |
| Archived V2 frozen_design.json | 1d4bca1a1f4a745c4df4e718729259f68697ddb6c9bee8cec4383da719a99baf |
| Archived V2 training/config.json | 3a1f179eece94915dd18d241272d2e6398ae8834c7e8b23f7a0a5d936efd9cd1 |
| Archived V2 training/split.json | 0a211bda390b5c83812ba9c796582a0497eb8f6ab5fdd169e82e2106f0f37eaa |
| Archived V2 records.csv | fd237754a512928270b78b4a83e661f5939edcf62249f2c084ab0dbf803ba021 |
| Archived V2 report.json | ede878ad998c01a87f128a1d72537055008d89ab6732c1342e030d7f30d036fd |
| Archived V2 tune/detections.jsonl | 057a4fcb7c502b7e4d743276c486957fe40b94742ea59e4f8e9d0a4804a1ef8b |
| Validation image manifest, 548 files | 7f2031b7a0ca3afd4f46f6dc830db47152a0834922e446082b4b8e0d27fbeca4 |
| Validation label manifest, 548 files | be52d59b9ab3f762f55ae7fb07b23e98137057261f17c3e6a39e9d3211ce4e6e |
| Public candidate snapshot before V2 | 4216e85a4aa04a607711788a0d2271832b5df10d714faf553acbf687602fff8f |
| Replayed records.csv, external | beb6da4c7f49299fd1d6581473862acb3a09d855f41f6a7e8cc3607b795a9573 |

Data manifests use sorted relative path, TAB, byte size, TAB, file SHA256, LF. The checkpoint split equals the archived training split and the split embedded in frozen_design.json. All 48 train and 32 tune image/label identities were recorded separately, without training. Train/tune prefixes are disjoint from each other and from the 76 validation prefixes. Validation names and order equal the frozen 548-name design exactly. The retained training config's eval_count=80 describes its earlier local subset, not a replacement for this explicitly frozen full-validation list.

The frozen baseline receiver threshold is 0.12. The learned threshold is copied as the exact stored float **0.14500000000000002**, publicly rounded to 0.145. Low-confidence inference floor 0.001, NMS IoU 0.7, max_det 300, imgsz 1280, class-0 wrapper filtering and historical RGB/NumPy channel convention remain unchanged. AP pools the low-confidence post-NMS detections; operating thresholds apply to Precision/Recall. Dataset AP is not the mean of the CSV's per-image AP column.

### Commands and threshold freezing

All commands ran from the public root with the external compatible interpreter, offline YOLO settings and external cache/temp/output paths. VERIFY_ROOT is the external learned-v2 verification directory. Private expanded paths, source hashes and arguments are retained in command.private.json; public source remained unchanged.

~~~powershell
python "$VERIFY_ROOT/preflight_v2.py"
python "$VERIFY_ROOT/probe_v2_environment.py"
python "$VERIFY_ROOT/run_frozen_v2.py"
python "$VERIFY_ROOT/compare_v2.py"
python scripts/bootstrap_saved_detections.py "$VERIFY_ROOT/run" --left learned --right baseline --repeats 4000 --seed 731
python scripts/bootstrap_saved_detections.py "$VERIFY_ROOT/run" --left learned_fixed012 --right baseline --repeats 4000 --seed 731
python "$VERIFY_ROOT/compare_v2_bootstrap.py"
~~~

The run_frozen_v2.py driver calls the unchanged public evaluator.run with all=True, jpeg=False, threads=4, the exact external checkpoints/data and a fresh external output directory. Its sole orchestration adapter supplies the archived calibration object verbatim when calibrate_from_tune is requested. The threshold-search routine is not executed; the loader was called once and threshold-search calls were zero. The existing CLI would recompute tune selection, so it was deliberately not used directly for the fixed-point check. This does not change codec, allocator, projection, detection, matching, metrics or thresholds. A portable fixed-calibration adapter is documented in REPRODUCE.md.

### Frozen versus reproduced metrics

Recall and Precision entries below are fractions; deltas ending in _pp are percentage points; BPP is bits/pixel. The table uses exact frozen values rather than rounded expectations. All differences are zero, with zero acceptance tolerance. Integer counts, encoded byte counts, arrays, predictions and stored thresholds require exact equality.

| Metric | Frozen | Reproduced | Difference | PASS/FAIL |
|---|---:|---:|---:|---|
| validation_images | 548 | 548 | +0 | PASS |
| validation_sequences | 76 | 76 | +0 | PASS |
| baseline_targets | 13969 | 13969 | +0 | PASS |
| baseline_recall | 0.6332593600114539 | 0.6332593600114539 | +0 | PASS |
| baseline_precision | 0.4492407698948758 | 0.4492407698948758 | +0 | PASS |
| baseline_mean_image_ap50 | 0.5155655272365327 | 0.5155655272365327 | +0 | PASS |
| baseline_dataset_ap50 | 0.5755142143146682 | 0.5755142143146682 | +0 | PASS |
| baseline_mean_bpp | 0.4041902087975474 | 0.4041902087975474 | +0 | PASS |
| learned_targets | 13969 | 13969 | +0 | PASS |
| learned_recall | 0.6339752308683514 | 0.6339752308683514 | +0 | PASS |
| learned_precision | 0.4831423895253683 | 0.4831423895253683 | +0 | PASS |
| learned_mean_image_ap50 | 0.5279425992582794 | 0.5279425992582794 | +0 | PASS |
| learned_dataset_ap50 | 0.5933346934268822 | 0.5933346934268822 | +0 | PASS |
| learned_mean_bpp | 0.4023379685415227 | 0.4023379685415227 | +0 | PASS |
| learned_fixed012_targets | 13969 | 13969 | +0 | PASS |
| learned_fixed012_recall | 0.6523015248049252 | 0.6523015248049252 | +0 | PASS |
| learned_fixed012_precision | 0.4362522142959736 | 0.4362522142959736 | +0 | PASS |
| learned_fixed012_mean_image_ap50 | 0.5372009650811732 | 0.5372009650811732 | +0 | PASS |
| learned_fixed012_dataset_ap50 | 0.5933346934268822 | 0.5933346934268822 | +0 | PASS |
| learned_fixed012_mean_bpp | 0.4023379685415227 | 0.4023379685415227 | +0 | PASS |
| learned_delta_recall_pp | 0.07158708568975047 | 0.07158708568975047 | +0 | PASS |
| learned_delta_precision_pp | 3.390161963049243 | 3.390161963049243 | +0 | PASS |
| learned_delta_AP_pp | 1.782047911221396 | 1.782047911221396 | +0 | PASS |
| learned_fixed012_delta_recall_pp | 1.904216479347132 | 1.904216479347132 | +0 | PASS |
| learned_fixed012_delta_precision_pp | -1.298855559890227 | -1.298855559890227 | +0 | PASS |
| learned_fixed012_delta_AP_pp | 1.782047911221396 | 1.782047911221396 | +0 | PASS |
| learned_delta_BPP | -0.001852240256024762 | -0.001852240256024762 | +0 | PASS |
| size_error_median | 0.0007598429307370536 | 0.0007598429307370536 | +0 | PASS |
| size_error_p95 | 0.01905436022128678 | 0.01905436022128678 | +0 | PASS |
| size_error_max | 0.0199274724500933 | 0.0199274724500933 | +0 | PASS |
| within_2pct | 548 | 548 | +0 | PASS |
| within_5pct | 548 | 548 | +0 | PASS |
| baseline_confidence | 0.12 | 0.12 | +0 | PASS |
| receiver_confidence | 0.145 | 0.145 | +0 | PASS |
| minimum_ratio | 0.98 | 0.98 | +0 | PASS |

At the tuned point, the reproduced improvements are **+1.782047911221396 pp AP@0.5**, **+3.390161963049243 pp Precision**, **+0.07158708568975047 pp Recall**. These preserve the frozen public rounded claims +1.78 / +3.39 / +0.07 pp. Mean BPP falls from 0.4041902087975474 to 0.40233796854152265, a difference of -0.001852240256024762 BPP.

### Budget, identity and secondary result

All **548/548** learned streams are no larger than their corresponding baseline; all also satisfy the frozen 98%-100% interval. Learned/baseline byte ratio minimum 0.9800725275499067, maximum 1; mean per-image ratio 0.9945160108994076. Absolute relative byte difference median **0.07598429307370536%**, P95 **1.9054360221286776%**, maximum **1.9927472450093298%**. Each baseline-level candidate-bank stream equals the independently encoded baseline stream byte-for-byte, checked by the public evaluator on all images.

Baseline and learned encoded byte counts match their frozen per-image records on 548/548 images each. Four archived sample TAC files are available and match bit-for-bit; no claim is made that all original full-dataset streams were retained. Raw head/prediction equality provides a separate full-dataset check.

The secondary common-threshold learned comparison at confidence 0.12 also reproduces: Recall **65.23015248049252%**, Precision **43.625221429597355%**, AP **59.33346934268822%**. Relative to baseline this is **+1.9042164793471317 pp Recall** and **-1.2988555598902274 pp Precision**. It remains secondary, not the public tuned headline.

Both sequence-bootstrap JSONs reproduce exactly for 4,000 resamples, seed 731, 76 sequences. The tuned Recall interval still includes zero. Small differences in the last arithmetic digits between distinct report/bootstrapping formulations are already present in frozen artifacts; each was compared to the same frozen formulation, without introducing a tolerance or changing its definition.

### Runtime, artifacts and release interpretation

Full inference plus evaluator aggregation: **513.770 seconds (8.56 minutes)**; start UTC 2026-10-18T16:11:05.139711+00:00; finish UTC 2026-10-18T16:19:38.912843+00:00. This excludes environment recovery, probes, comparisons and the separately rerun bootstrap commands. Timings differ from historical hardware/process timing and are not real-time UAV evidence. The Visual C++ Redistributable warning appeared again; imports and inference succeeded without an installer.

Every regenerated image, TAC, NPZ/head, per-image CSV, GT-containing JSONL, private manifest, package inventory and diagnostic command remains outside the public candidate under the external learned-v2 verification directory. Only documentation and inventory hashes changed publicly. The original archive, scientific source, CLAIMS.md, configs and frozen results remain unchanged. No model/data artifact or download link was added.

Numerical V2 release verification is complete. The historical Risk-Aware claim remains separate and accepted for release with its documented 548/548 byte-count agreement, 547/548 current-stack detection identity and explained Ultralytics IoU-boundary difference. V2 success does not convert that older current-stack run to bitwise exactness.

Validation reuse, merged VisDrone person categories, withheld checkpoints, dataset redistribution exclusions, licensing/ownership review and lack of field deployment remain in force. No untouched-test, real-time or search-and-rescue deployment claim follows from this replay. Remaining release blockers are rights/provenance sign-off and clear external prerequisite instructions, not unresolved V2 numerical disagreement.
