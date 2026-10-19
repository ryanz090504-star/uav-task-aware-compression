# Evaluation protocol

Use [CLAIMS.md](../CLAIMS.md) for frozen numbers and exact definitions. Matching is confidence-ordered and one-to-one at IoU >=0.5. Micro Recall/Precision pool TP/FP/FN; mean image Recall weights every image equally. The historical Risk-Aware headline uses the latter, while the historical +3.15 pp statistic uses pooled counts.

Dataset AP pools confidence-ranked detections across images and integrates the monotone precision envelope. Historical CSV mAP50 is per-image AP, not dataset mAP. Learned V2 saves low-confidence post-NMS predictions for AP; a separately tuned threshold controls its Precision/Recall operating point. Do not mix threshold-truncated and low-confidence AP or this custom evaluator with official VisDrone/COCO evaluation.

The task merges VisDrone categories 1 and 2, excludes score-zero rows and other categories, and does not implement the complete official ignored-region protocol. The custom small-person metric is not COCO AP Small.

JPEG controls use nearest file size from Q1-Q100. Report residual size errors rather than calling every pair equal in bytes. V2's learned-versus-baseline hard budget is a distinct comparison. Container headers count toward bytes.

Historical per-image uncertainty is a normal approximation; sequence bootstrap addresses within-sequence dependence for pooled results. Neither establishes independent training-seed robustness. Validation was reused during research and is not an untouched final test set. Historical RGB arrays follow the retained detector-interface channel convention; correcting that requires fresh comparable evaluation.

strict_evaluation.py consumes locally generated baseline records; bootstrap_saved_detections.py consumes locally generated detection JSONL including GT. Both can write outputs; use a new ignored directory or explicit output path. These raw inputs are deliberately absent from the release. Do not upload regenerated data/annotations/detections.

## Frozen method identities and release checks

Risk-Aware uses historical_full_gaussian for the +2.62 pp mean-image Recall headline. The later truncated_6sigma implementation is a distinct variant. The full current-stack historical replay has exact adaptive byte-count agreement on 548/548 images and exact detection outcomes on 547/548. On one image, Ultralytics 8.4.129 gives IoU 0.4999998971 and 8.4.181 gives 0.5000036525 for the same candidate/GT; a 0.000061-pixel box-edge shift changes one FP/FN pair into a TP. Repetition is deterministic within each stack, input pixels are identical, and NMS candidates/order are unchanged. Older components recover the frozen row. Both mean gains round to +2.62 pp. See the [single-image investigation](RISK_AWARE_REPRODUCTION_INVESTIGATION.md). Do not equate byte-count agreement to full bitstream identity when original streams were not retained.

Learned V2 retains truncated_6sigma and the frozen 0.12 baseline / 0.145 learned thresholds. Its complete 548-image numerical replay passes exactly: AP gain +1.782047911221396 pp, Precision +3.390161963049243 pp, Recall +0.07158708568975047 pp and BPP -0.001852240256024762. All byte-budget checks pass. [Exact metric table](RESULTS.md), [full verification](REAL_DATA_RELEASE_VERIFICATION.md).

No new threshold search is part of frozen verification. The CLI's ordinary tune calibration is replaced only by an external orchestration adapter that returns the archived calibration object. Codec, allocator, detector, projection and metric algorithms remain unchanged. AP uses low-confidence floor 0.001 outputs; thresholds filter Precision/Recall. Seed 731 and 4,000 sequence resamples reproduce both saved bootstrap reports exactly. Equality requires zero tolerance in the compatible stack; timings are excluded. Exact original launch inventory remains unconfirmed despite full output compatibility.
