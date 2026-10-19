# Frozen v0.1.0 results

[CLAIMS.md](../CLAIMS.md) remains the single source of truth. Differences are percentage points (pp), not relative improvements. Release checking replays existing methods; it adds no new research claim.

## Risk-Aware

The public headline is **+2.62 pp mean per-image Recall versus per-image size-matched JPEG**, with historical_full_gaussian, tile 256 / BG8 / FG82 / mean pooling, on 548 validation images / 13,969 merged person targets.

The frozen exact mean gain is +2.6174015150906684 pp; its normal-approximation interval is [+1.72, +3.51] pp. Historical target-level gain +3.1498317703486323 pp is a separate pooled aggregation, not the headline or the latest revalidation result.

Release replay verifies 548/548 adaptive encoded byte counts and 547/548 current-stack detection outcomes. The remaining image is an explained Ultralytics version-dependent IoU=0.5 boundary; its frozen outcome reproduces with older components. This qualification does not change the frozen result. See [evaluation](EVALUATION.md) and [investigation](RISK_AWARE_REPRODUCTION_INVESTIGATION.md).

## Learned Allocation V2

Semantics are truncated_6sigma; validation is 548 images / 76 sequences / 13,969 targets. Baseline confidence is 0.12 and learned tuned confidence is 0.145 (stored float 0.14500000000000002).

| Metric | Frozen/reproduced baseline | Frozen/reproduced learned | Exact delta |
|---|---:|---:|---:|
| Dataset AP@0.5 | 57.55142143146682% | 59.33346934268822% | +1.782047911221396 pp |
| Precision | 44.92407698948758% | 48.314238952536825% | +3.390161963049243 pp |
| Recall | 63.32593600114539% | 63.39752308683514% | +0.07158708568975047 pp |
| Mean BPP | 0.4041902087975474 | 0.40233796854152265 | -0.001852240256024762 |

**Numerical validation reproduces exactly**, including all 1,644 scientific/prediction records, 1,096 raw-head array files and both 4,000-sequence-resample bootstrap results. The confirmed compatible stack and original-launch inventory limitation are documented in [verification](REAL_DATA_RELEASE_VERIFICATION.md). No thresholds were retuned.

All 548 learned outputs meet the baseline byte budget and the 98%-100% interval. Relative absolute size difference median is 0.07598429307370536%, P95 1.9054360221286776%, maximum 1.9927472450093298%.

The secondary common-conf=0.12 point yields Recall +1.9042164793471317 pp and Precision -1.2988555598902274 pp. It is not the main headline. Tuned Recall's sequence-bootstrap interval includes zero; Learned V2 does not universally replace Risk-Aware.

## Evidence availability

Compact aggregate reports, frozen configs and split IDs are public. Dataset material, per-image CSVs, GT detections and the exact detector remain excluded. A sanitized allocator export is prepared externally, with identical model state/outputs and new hashes; distribution review remains separate. [Model provenance](../MODEL_PROVENANCE.md).

Validation reuse, custom category merging, approximate JPEG matching and historical channel convention limit interpretation. CPU timing is not real-time deployment evidence, and there has been no real search-and-rescue deployment.
