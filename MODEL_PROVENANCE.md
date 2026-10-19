# Model and checkpoint provenance

The source/evaluation pipeline is public-release code. The exact detector checkpoint is **not redistributed in v0.1.0**, pending final dataset/model redistribution review. Code publication does not make the original detector available. A sanitized Learned V2 allocator candidate is prepared externally; original metadata-rich checkpoints remain withheld.

## Frozen detector — withheld

- Filename: visdrone_yolov8n_1280_b18_s17_best.pt; bytes 6,327,402.
- SHA256: 263cfd00bbd54283ad6982d79b04e7f9306d3d63c82a3264f500fb81be25426a.
- Role: fixed sender proposals and receiver detection; native teacher/loss for allocator training.
- Recorded training: seed 17, image size 1280, 120 configured epochs, batch 18, continuation from a prior b18 detector. Historical training records select epoch 88; the stripped checkpoint epoch=-1 does not independently prove that choice. Recorded Ultralytics metadata is 8.4.129.
- Early lineage evidence includes an upstream YOLOv8n initialization and short continuation stages, but the full parent/hash and rights chain is not authenticated. Do not merge unrelated archive runs into a claimed lineage.
- Original checkpoint metadata contains private paths. Dataset/upstream/model redistribution rights and sanitized export are unresolved. No download link or detector copy is supplied.

Supply a lawfully held local detector using workflow --model or the V2 model argument. Verify SHA256 before claiming exact reproduction. A merely architecture-compatible detector can exercise the pipeline but cannot be assumed to reproduce these frozen results. The validated compatible V2 stack uses Ultralytics 8.4.129 / torch 2.13.0+cpu; see [verification](docs/REAL_DATA_RELEASE_VERIFICATION.md) for full versions and original-launch limitations. Frozen thresholds, image size, class merge and historical channel convention must be preserved.

## Original Learned V2 allocator — withheld

- Archive artifact: rate_allocator.pt, 20,463 bytes, balanced eight-epoch run.
- SHA256: e28024b48c26d378f3378d40d24ef266f68a4f61ea37ae452e977b721a90bf45.
- Origin: first-party train_rate_allocator_v2.py, seed 731, 48 training / 32 tune sequences, 1280 native-resolution detector, hidden=32, max_shift=12, 1,633 trainable parameters.
- The fixed detector hash, public split manifest and frozen design bind the experiment. This model uses truncated_6sigma, not historical_full_gaussian.
- The original includes local config paths/logs. It remains excluded; export does not grant additional dataset/model rights.

## Sanitized allocator release candidate

| File | Bytes | New SHA256 |
|---|---:|---|
| allocator-v2-v0.1.0.npz | 8560 | 5089346d903808f21433aa991009680aa5b643f2239fb2892f25ce30aa105f91 |
| allocator-v2-v0.1.0.json | 3492 | 24bc8d558f2142346d5d98c6609aa57de4dd8dc43e43f49ff128dd3ce4e2c8e6 |

These files and their SHA256SUMS are outside Git. They are prepared local assets, not an announced or uploaded distribution. Final rights/redistribution sign-off remains required. The new hashes intentionally differ from the original checkpoint.

The NPZ is a data-only ZIP of eight finite float32 arrays: feature_mean, feature_scale, and the weights/biases of three linear layers. No pickle/object array, detector, optimizer, logs or dataset/machine path is present. Fixed ZIP entry ordering/timestamps avoid carrying source-machine file metadata. JSON holds the architecture, feature names/order, normalization buffer references and 0.05 scale floor, integer quality offsets -12 through +12 clipped to Q1-Q95, codec parameters, 98%-100% allocation rule, frozen thresholds, method version and original/detector/split hashes. It contains no training config/path or usernames.

Method version: learned-allocation-v2.truncated_6sigma.v0.1.0. The tensors/normalization are exactly equal to the original; deterministic forward outputs and quality scores are bitwise equal on 329 synthetic feature rows in four batch sizes. Repeated export gives identical bytes. This is an inference-state export, not retraining or a new model. It does not retroactively change the source artifact hash or the frozen evaluation evidence.

## Reproduce export and load

Run from the repo root with PyTorch available and the exact original allocator locally held. Choose a fresh external output directory:

```sh
python scripts/export_allocator_release.py --checkpoint EXTERNAL_ORIGINAL_ALLOCATOR --output EXTERNAL_ALLOCATOR_ASSET
```

The command rejects a different source hash and an output under the repository. It copies only allowlisted inference data, verifies identical tensors/outputs and writes separate asset checksums. No data or detector is needed. For use with the unchanged evaluator, the following adapter creates a minimal, temporary inference checkpoint outside Git from the clean NPZ/JSON and public split manifest:

```python
import hashlib
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, "scripts")
from export_allocator_release import load_export
model, spec = load_export(Path("EXTERNAL_ALLOCATOR_ASSET"))
split_path = Path("results/learned_v2/training/split.json")
assert hashlib.sha256(split_path.read_bytes()).hexdigest() == spec["provenance"]["split_manifest_sha256"]
state = {
    "state_dict": model.state_dict(),
    "model_config": {key: spec["architecture"][key] for key in ("hidden", "max_shift")},
    "compression_config": spec["codec"],
    "split": json.loads(split_path.read_text()),
}
torch.save(state, Path("EXTERNAL_INFERENCE_CHECKPOINT.pt"))
```

That temporary wrapper has its own hash and is not the historical checkpoint. The evaluator accepts its explicit truncated_6sigma mode, checks the required detector hash and split overlap, and uses the same inference state. Follow [frozen-calibration instructions](docs/REPRODUCE.md); do not run a new threshold search. The original-checkpoint hash assertion in that recipe applies to the original artifact; when using a derived wrapper, verify the NPZ/JSON checksums, state equality and required detector/split hashes instead, and record the wrapper's new hash. No scientific algorithm is changed by this materialization.

The recorded full V2 verification used the original frozen checkpoint. This export has been checked for state/output identity, not subjected to a new full research experiment. All raw verification artifacts remain external. Ownership, detector lineage and model/dataset distribution questions are tracked in [open issues](OPEN_ISSUES_BEFORE_RELEASE.md) and [licensing audit](LICENSE_AUDIT.md).
