# Final v0.1.0 release manifest

Scientific scope is frozen. This inventory includes every file under the cleaned public root; caches/build outputs and private helpers are absent. Source archive material remains untouched. Root SHA256SUMS is the single repository-file hash inventory; this manifest records purpose rather than duplicating hashes.

Detector/original allocator remain withheld. The sanitized NPZ/JSON candidate and its separate SHA256SUMS are external assets, not Git contents. Rights review remains separate. See MODEL_PROVENANCE.md and docs/FINAL_RELEASE_AUDIT.md.

| Public file | Inclusion purpose |
|---|---|
| .github/workflows/tests.yml | Lightweight public installation/unit/demo CI; no real data or checkpoint |
| .gitignore | Public release metadata, instructions, policy or dependency declaration |
| CHANGELOG.md | Public release metadata, instructions, policy or dependency declaration |
| CITATION.cff | CFF 1.2.0 software citation; confirmed public author Ryan Zhang; public release date deferred |
| CLAIMS.md | Single authoritative frozen claim definitions; byte-identical |
| configs/risk_aware_v0.1.0.json | Portable/frozen protocol configuration; unchanged |
| configs/visdrone_person.yaml | Portable/frozen protocol configuration; unchanged |
| CONTRIBUTING.md | Public release metadata, instructions, policy or dependency declaration |
| DATASET.md | Public release metadata, instructions, policy or dependency declaration |
| docs/CLEAN_ENVIRONMENT_TEST.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/EVALUATION.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/FINAL_RELEASE_AUDIT.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/LIMITATIONS.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/METHOD.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/REAL_DATA_RELEASE_VERIFICATION.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/REPRODUCE.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/RESULTS.md | Method, evaluation, reproduction or valuable dated audit documentation |
| docs/RISK_AWARE_REPRODUCTION_INVESTIGATION.md | Method, evaluation, reproduction or valuable dated audit documentation |
| EXCLUDED_MATERIAL.md | Public release metadata, instructions, policy or dependency declaration |
| LICENSE | Unmodified standard GNU AGPL v3 text; ownership/integration sign-off pending |
| LICENSE_AUDIT.md | Public release metadata, instructions, policy or dependency declaration |
| MODEL_PROVENANCE.md | Public release metadata, instructions, policy or dependency declaration |
| models/PROVENANCE.json | Withheld-original and sanitized-candidate provenance; no weights |
| OPEN_ISSUES_BEFORE_RELEASE.md | Public release metadata, instructions, policy or dependency declaration |
| PORTABILITY_AUDIT.md | Public release metadata, instructions, policy or dependency declaration |
| pyproject.toml | Public release metadata, instructions, policy or dependency declaration |
| README.md | Public release metadata, instructions, policy or dependency declaration |
| RELEASE_MANIFEST.md | Public release metadata, instructions, policy or dependency declaration |
| RELEASE_SIGNOFF.md | Confirmed author identity, explicit rights-holder authorization and withheld-material policy |
| RELEASE_NOTES_v0.1.0.md | Public release metadata, instructions, policy or dependency declaration |
| requirements-cpu-lock.txt | Public release metadata, instructions, policy or dependency declaration |
| requirements-cpu.txt | Public release metadata, instructions, policy or dependency declaration |
| results/baseline/paired_recall_summary.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/baseline/strict_evaluation.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/baseline/summary.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/baseline/workflow_manifest.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/bootstrap_learned_fixed012_minus_baseline.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/bootstrap_learned_minus_baseline.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/frozen_design.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/report.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/training/config.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/training/split.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/training/training.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| results/learned_v2/tune/report.json | Frozen compact aggregate evidence/configuration/split provenance; unchanged |
| scripts/allocation_utils.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/bootstrap_saved_detections.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/convert_visdrone_train.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/convert_visdrone_val.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/evaluate_rate_allocator_v2.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/export_allocator_release.py | Data-only sanitized allocator export/load and integrity verification |
| scripts/plot_results.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/strict_evaluation.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| scripts/train_rate_allocator_v2.py | Retained preparation, training/evaluation, metric or plot entry/helper |
| SHA256SUMS | Actual repository-file hashes; excludes only its own self-hash |
| src/task_aware_compression/__init__.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/benchmark.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/cli.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/compressor.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/dataset.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/detector.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/hevc.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/importance.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/learned_allocation.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/metrics.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/synthetic.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/types.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/visualization.py | Retained first-party scientific implementation; unchanged by finalization |
| src/task_aware_compression/workflow.py | Retained first-party scientific implementation; unchanged by finalization |
| tests/test_allocator_export.py | Dataset-free unit/regression source; existing tests unchanged |
| tests/test_core.py | Dataset-free unit/regression source; existing tests unchanged |
| tests/test_importance_semantics.py | Dataset-free unit/regression source; existing tests unchanged |
| tests/test_learned_allocation.py | Dataset-free unit/regression source; existing tests unchanged |
| THIRD_PARTY_NOTICES.md | Public release metadata, instructions, policy or dependency declaration |
