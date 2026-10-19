"""Export/load the data-only v0.1 allocator candidate; no detector or data required."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "e28024b48c26d378f3378d40d24ef266f68a4f61ea37ae452e977b721a90bf45"
METHOD_VERSION = "learned-allocation-v2.truncated_6sigma.v0.1.0"
TENSORS = "allocator-v2-v0.1.0.npz"
SPEC = "allocator-v2-v0.1.0.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_export(folder, state_dict, model_config, codec, provenance):
    import numpy as np
    from task_aware_compression.learned_allocation import FEATURE_NAMES, EVAL_OFFSETS

    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("export directory must be empty")
    folder.mkdir(parents=True, exist_ok=True)
    shapes = {}
    # Fixed entry ordering/timestamps; .npy contains numeric arrays, never pickle.
    with zipfile.ZipFile(folder / TENSORS, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, tensor in sorted(state_dict.items()):
            array = np.ascontiguousarray(tensor.detach().cpu().numpy())
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("only finite float32 model tensors are supported")
            stream = io.BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 0
            archive.writestr(entry, stream.getvalue())
            shapes[name] = {"shape": list(array.shape), "dtype": "float32"}
    hidden = model_config["hidden"]
    spec = {
        "format": "tac-allocator-npz-v1",
        "method_version": METHOD_VERSION,
        "architecture": {
            "type": "TileRateAllocator", **model_config,
            "input_features": len(FEATURE_NAMES),
            "layers": [f"Linear(16,{hidden})", "SiLU", f"Linear({hidden},{hidden})",
                       "SiLU", f"Linear({hidden},1)", "tanh times max_shift"],
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_normalization": {"mean_tensor": "feature_mean", "scale_tensor": "feature_scale", "minimum_scale": 0.05},
        "quality": {"offsets": list(EVAL_OFFSETS), "minimum": 1, "maximum": 95, "normalizer": 95},
        "codec": codec,
        "allocation": {"function": "select_near_budget", "minimum_ratio": 0.98, "baseline_fallback": True, "container": "TAC1/version2"},
        "receiver": {"baseline_confidence": 0.12, "learned_confidence": 0.14500000000000002, "ap_floor": 0.001},
        "provenance": provenance,
        "tensor_file": TENSORS,
        "tensor_file_sha256": sha256(folder / TENSORS),
        "state_dict": shapes,
    }
    (folder / SPEC).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def load_export(folder):
    """Load only numeric state into the existing allocator implementation."""
    import numpy as np
    import torch
    from task_aware_compression.learned_allocation import FEATURE_NAMES, TileRateAllocator
    from task_aware_compression.types import CompressionConfig

    folder = Path(folder)
    spec = json.loads((folder / SPEC).read_text(encoding="utf-8"))
    if spec["format"] != "tac-allocator-npz-v1" or spec["method_version"] != METHOD_VERSION:
        raise ValueError("unsupported allocator format/method")
    if spec["tensor_file"] != TENSORS or sha256(folder / TENSORS) != spec["tensor_file_sha256"]:
        raise ValueError("tensor archive identity mismatch")
    if spec["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("feature names/order mismatch")
    if spec["architecture"]["type"] != "TileRateAllocator":
        raise ValueError("unsupported architecture")
    if spec["codec"]["importance_semantics"] != "truncated_6sigma":
        raise ValueError("V2 requires truncated_6sigma")
    CompressionConfig(**spec["codec"])
    model = TileRateAllocator(hidden=spec["architecture"]["hidden"], max_shift=spec["architecture"]["max_shift"])
    expected = model.state_dict()
    with np.load(folder / TENSORS, allow_pickle=False) as archive:
        if set(archive.files) != set(expected) or set(spec["state_dict"]) != set(expected):
            raise ValueError("model tensor names mismatch")
        tensors = {}
        for name, reference in expected.items():
            array = archive[name]
            if array.dtype != np.float32 or tuple(array.shape) != tuple(reference.shape) or not np.isfinite(array).all():
                raise ValueError("model tensor shape/dtype/value mismatch")
            if spec["state_dict"][name] != {"shape": list(array.shape), "dtype": "float32"}:
                raise ValueError("tensor specification mismatch")
            tensors[name] = torch.from_numpy(array.copy())
    model.load_state_dict(tensors, strict=True)
    return model.eval(), spec


def export_checkpoint(checkpoint, output):
    import numpy as np
    import torch
    from task_aware_compression.learned_allocation import FEATURE_NAMES, TileRateAllocator
    from task_aware_compression.types import CompressionConfig

    checkpoint, output = Path(checkpoint), Path(output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("release export must be outside the Git repository")
    if sha256(checkpoint) != SOURCE_SHA256:
        raise ValueError("checkpoint is not the exact frozen V2 allocator")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    codec = dict(state["compression_config"], importance_semantics="truncated_6sigma")
    CompressionConfig(**codec)
    provenance = {
        "original_allocator_sha256": SOURCE_SHA256,
        "required_detector_sha256": state["split"]["model_sha256"],
        "split_manifest_sha256": sha256(ROOT / "results/learned_v2/training/split.json"),
        "relationship": "Identical inference tensors and normalization buffers; metadata removed; new serialization and hashes.",
    }
    _write_export(output, state["state_dict"], state["model_config"], codec, provenance)
    original = TileRateAllocator(**state["model_config"])
    original.load_state_dict(state["state_dict"])
    original.eval()
    clean, spec = load_export(output)
    if not all(torch.equal(value, clean.state_dict()[name]) for name, value in original.state_dict().items()):
        raise RuntimeError("export changed model tensors")
    rng = np.random.default_rng(731)
    cases = 0
    with torch.no_grad():
        for count in (1, 7, 64, 257):
            features = rng.random((count, len(FEATURE_NAMES)), dtype=np.float32)
            features[:, FEATURE_NAMES.index("baseline_quality")] = rng.integers(1, 96, count) / 95
            qualities = np.clip(features[:, 13, None] * 95 + np.arange(-12, 13)[None, :], 1, 95).astype(np.float32)
            x, q = torch.from_numpy(features), torch.from_numpy(qualities)
            if not torch.equal(original(x), clean(x)) or not torch.equal(original.scores(x, q), clean.scores(x, q)):
                raise RuntimeError("export changed allocator outputs")
            cases += count
    sums = "".join(f"{sha256(output / name)}  {name}\n" for name in (SPEC, TENSORS))
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")
    return {"status": "PASS", "original_sha256": SOURCE_SHA256,
            "tensor_sha256": sha256(output / TENSORS), "spec_sha256": sha256(output / SPEC),
            "state_tensors_identical": True, "normalization_buffers_identical": True,
            "forward_and_score_outputs_identical": True, "synthetic_feature_rows": cases,
            "contains_training_logs_or_paths": False, "method_version": spec["method_version"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="exact locally held original V2 allocator")
    parser.add_argument("--output", type=Path, required=True, help="new empty directory outside the repository")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("missing original allocator; no checkpoint is downloaded")
    try:
        result = export_checkpoint(args.checkpoint, args.output)
    except (ImportError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
