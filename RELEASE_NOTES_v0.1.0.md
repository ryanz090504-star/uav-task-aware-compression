# v0.1.0 release notes — local candidate

Scientific scope is frozen. The code and evaluation pipeline are public-release candidates; the exact detector is not redistributed. No GitHub repository or release has been created.

- Risk-Aware: +2.62 pp mean per-image Recall against per-image size-matched JPEG, 548 images / 13,969 targets, historical_full_gaussian.
- Learned V2: +1.78 pp dataset AP@0.5 and +3.39 pp Precision at similar Recall and slightly lower BPP, truncated_6sigma. Its full frozen numerical validation reproduces exactly on 548 images / 76 sequences.
- All 548 learned files satisfy their baseline byte budget. Exact values and definitions are in [results](docs/RESULTS.md) and [evaluation](docs/EVALUATION.md); [CLAIMS.md](CLAIMS.md) remains authoritative.

The deterministic synthetic demo and public unit tests require no VisDrone data or detector. Lightweight CI installs CPU PyTorch for the optional allocator tests. Remote CI has not been run; local checks are recorded in [the final audit](docs/FINAL_RELEASE_AUDIT.md).

A sanitized allocator candidate exists outside Git as numeric NPZ state and JSON metadata, with separate checksums and identical allocator outputs. It has a new serialization/hash and is not the original checkpoint. It remains subject to model/dataset redistribution review. The detector and original metadata-rich allocator remain withheld. See [model provenance](MODEL_PROVENANCE.md).

Validation was reused during research, the person-category merge is custom, JPEG matching is approximate, CPU timing is not real-time deployment evidence, and the system has not been deployed in real search-and-rescue. See [limitations](docs/LIMITATIONS.md). First-party AGPL-3.0 preparation remains subject to ownership/integration review; no new dataset or model rights are granted.

Repository file checksums are in SHA256SUMS. The external allocator bundle has its own SHA256SUMS. No dataset, GT artifact, detector, runtime bundle or machine-specific path is included in Git.
