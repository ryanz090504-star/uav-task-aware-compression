# Frozen public claims for v0.1.0

This document defines the public interpretation of retained evidence. It introduces no experiments and changes no saved results. Differences below are percentage points (pp), not relative percentage improvements.

## Risk-Aware headline

**Risk-Aware improves mean per-image Recall by +2.62 pp versus per-image file-size-matched uniform JPEG in the recorded 548-image VisDrone validation experiment.** The operating point is tile=256, background Q8, foreground Q82, mean importance pooling. The same frozen detector is used for both methods. This describes task-information retention under compression, not detector retraining or performance above original images.

The historical per-image estimate averages paired image-level Recall differences. Its reported normal-approximation 95% interval is [+1.72, +3.51] pp. The retained aggregate is results/baseline/paired_recall_summary.json, calculated from the existing saved records without inference before those per-image records were excluded. Its source-record SHA256 and exact formula are recorded. results/baseline/summary.json provides the original aggregate metrics. Full per-image recomputation requires privately regenerated records; no raw records or report generator are shipped.

**+3.15 pp target-level Recall is a different historical aggregation of these records.** It pools TP and FN before calculating Recall (about 63.40% Risk-Aware versus 60.25% JPEG). It is retained in strict_evaluation.json as historical analysis, not described as the latest final revalidation or the public headline. Its reported sequence-bootstrap interval is [+2.41, +3.82] pp. Do not substitute it for +2.62 pp or combine their uncertainty estimates.

## Learned Allocation V2: a separate result

On 548 validation images from 76 filename-prefix sequences, with 13,969 merged person annotations:

| Metric | Baseline, confidence 0.12 | Learned V2, tuned confidence 0.145 | Difference |
|---|---:|---:|---:|
| Dataset AP@0.5 | 57.55% | 59.33% | +1.78 pp |
| Target-level Precision | 44.92% | 48.31% | +3.39 pp |
| Target-level Recall | 63.33% | 63.40% | +0.07 pp |

Recorded sequence-bootstrap 95% intervals are AP [+1.08, +2.42] pp, Precision [+1.98, +4.62] pp, and Recall [-0.47, +0.65] pp. Recall includes zero. These rounded values come from bootstrap_learned_minus_baseline.json (4,000 sequence resamples, seed 731).

Confidence 0.145 was selected on 32 independent allocator tune sequences by maximizing Recall while keeping Precision no more than 1 pp below the tune baseline. Allocator training used 48 other sequences; validation contains 76 sequences. Selection is recorded in frozen_design.json. AP uses saved low-confidence post-NMS predictions (floor 0.001), rather than only predictions above the operating threshold. The receiver threshold controls Precision and Recall.

At common confidence 0.12, a separate recorded comparison gives Recall +1.90 pp and Precision -1.30 pp. This tradeoff is not interchangeable with the tuned operating point. See bootstrap_learned_fixed012_minus_baseline.json and report.json.

**Learned V2 does not universally replace the Risk-Aware baseline.** Calibrated Recall is effectively unchanged and allocator inference adds cost. The baseline-versus-JPEG headline and learned-versus-baseline results are separate comparisons. Historical baseline and V2 baseline values come from different recorded runs and must not be spliced into a new comparison.

## Metric definitions

- TP: a confidence-ordered prediction matched one-to-one to an unused ground-truth box at IoU >=0.5. Unmatched predictions are FP; unmatched ground truth is FN.
- Target-level (micro) Recall = sum(TP)/(sum(TP)+sum(FN)); micro Precision = sum(TP)/(sum(TP)+sum(FP)). Mean per-image Recall averages image-level ratios, weighting images equally regardless of person count. The implementation assigns Recall=1 to an image without ground truth.
- Dataset AP@0.5 pools scored detections across images, constructs a monotone precision envelope and integrates over recall at IoU 0.5. This single merged-class metric is not official multi-class VisDrone evaluation or COCO AP@[0.5:0.95].
- Historical CSV mAP50 stores per-image AP@0.5. Averaging that column does not yield dataset AP. Older confidence-truncated AP cannot recover detections already filtered before saving and is not interchangeable with V2 low-confidence AP.
- Small-person GT has box area <=1.8% of image area. The code additionally filters small predictions at 1.8 times that threshold. These custom metrics are not standard COCO AP Small; no separate small-object headline is frozen here.
- BPP = 8 * encoded file bytes / image pixel count. TAC bytes include container metadata and tile payloads. Compression ratio uses uncompressed RGB bytes divided by encoded file bytes. Neither measures transmission energy.
- Historical mean-Recall uncertainty uses a normal interval over paired image differences. Historical pooled-Recall analysis and V2 include sequence resampling. These uncertainty methods differ and do not capture all training-seed or dataset-selection uncertainty.
- Adaptive end-to-end latency includes sender proposals, encoding, decoding and receiver detection. Incremental latency excludes sender proposals. Timings are environment-specific observations, not real-time guarantees.

## Validity and matching limitations

Conversion merges VisDrone category 1 (pedestrian) and category 2 (people) into class 0 (person), excluding annotation rows with score field 0 and other categories. This is a custom single-class task, not the official ignored-region/multi-class evaluation protocol.

Validation images were reused during earlier development and exploration. Allocator train/tune/validation separation by filename prefix does not create an untouched final test set. Train prefixes shared with validation are excluded, but prefix separation does not prove complete scene independence. Detector training/selection provenance requires separate review.

The historical loader supplies RGB NumPy arrays to an Ultralytics interface with a BGR convention. Comparisons share that behavior. Correcting channel handling requires separately authorized evaluation of every compared method; saved results are not corrected-channel results.

JPEG controls are selected per image from Q1-Q100 by nearest encoded size. This is approximate matching, not exact byte equality. The historical main setting reports median size error about 1.32%, P95 4.95%, maximum 5.99%; report sensitivity subsets alongside stricter matching language. Median/P95 tolerances do not imply equality for every pair.

V2's learned-versus-baseline byte constraint is separate: all 548 recorded learned files fit within baseline bytes; median absolute size difference is about 0.076%, P95 1.91%, maximum 1.99%. This does not imply exact JPEG matching or universal rate/performance dominance. The identity control addresses format/decoder changes, not detector or dataset limitations.

No universal improvement, real-time operation, untouched-test generalization, official VisDrone leaderboard result, complete reproduction of a third-party paper, or deployment readiness is claimed.

## Evidence

Baseline evidence is under results/baseline/. V2 reports, frozen design, training configuration/splits and tune summaries are under results/learned_v2/. VisDrone per-image metric CSVs and GT-containing prediction files are not distributed in Git or as v0.1.0 assets. Their historical identifiers remain provenance only in RELEASE_MANIFEST.md. Neither checkpoint is distributed. Full independent inference/AP/bootstrap reproduction therefore requires locally acquired data and matching privately held checkpoints; the public candidate supports aggregate inspection and synthetic execution. Publication remains subject to final ownership/licensing review and clean-environment validation.
