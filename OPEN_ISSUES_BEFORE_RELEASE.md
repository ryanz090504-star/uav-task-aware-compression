# Final local v0.1.0 release status and remaining sign-off

Scientific scope is frozen. Numerical validation and local release checks pass. No GitHub repository/commit/publication has been created.

## Completed

- Risk-Aware historical_full_gaussian: +2.62 pp mean per-image Recall, 548 images / 13,969 targets. Byte-count agreement is 548/548; current-stack detection agreement is 547/548. The remaining Ultralytics-dependent IoU boundary is explained and accepted for release with the documented qualification.
- Learned V2 truncated_6sigma: frozen metrics, all raw heads/predictions and both sequence-bootstrap files reproduce exactly on 548 images / 76 sequences. All 548 byte-budget checks pass. No retuning occurred.
- Final non-editable package install, 35 unit tests without skips, synthetic demos, help commands and CI-equivalent checks pass. CI is configured for data-free installation/tests/demo; remote GitHub execution has not occurred.
- Sanitized allocator candidate is created externally. Eight inference tensors/normalization buffers and deterministic outputs are identical; new NPZ/JSON hashes and the tested loader are documented. The original metadata-rich checkpoint remains excluded.
- English/privacy/secret-pattern, link, size and artifact-inventory checks pass. CLAIMS.md values, scientific source and frozen evidence are unchanged.

## Remaining before publication or asset upload

- Obtain final author/IP ownership attestation and AGPL/Ultralytics integration review. Standard unmodified LICENSE is present; numerical validation does not establish rights. Confirm whether collective contributor attribution in CITATION.cff should be replaced by the actual authorized author list; no individual names or DOI were invented.
- Detector weights remain withheld by v0.1.0 policy pending final dataset/model redistribution and parent-lineage review. The exact hash and local-supply instructions remain public. A compatible replacement can exercise code but cannot establish the frozen numbers.
- Clear model/dataset rights before distributing the sanitized allocator candidate. Technical sanitization and identical outputs do not grant redistribution rights. Keep the original checkpoint, all data and GT outputs private/external.
- Independent real-data reproduction still needs lawful local data and the exact withheld detector. Public code, aggregates and the safe allocator loader do not remove this prerequisite. No model download link is supplied.
- Exact original V2 launch inventory is unconfirmed; the compatible component stack is verified by all saved raw outputs. Other platforms/native runtimes remain unvalidated. Validation reuse, custom category merging, approximate JPEG matching and historical channel convention remain limitations.

No additional scientific experiment or algorithm adjustment is required by the completed checks. CPU timing is not real-time UAV evidence; the system has not been deployed in real search-and-rescue operations. See docs/FINAL_RELEASE_AUDIT.md for the final local checklist.
