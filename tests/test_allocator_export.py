"""Data-only export roundtrip and rejection tests; no research artifact needed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class AllocatorExportTests(unittest.TestCase):
    def setUp(self):
        from task_aware_compression.learned_allocation import TileRateAllocator
        from task_aware_compression.types import CompressionConfig
        from dataclasses import asdict

        script = Path(__file__).resolve().parents[1] / "scripts/export_allocator_release.py"
        spec = importlib.util.spec_from_file_location("allocator_export_test", script)
        self.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.exporter)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "export"
        torch.manual_seed(37)
        self.model = TileRateAllocator(hidden=8).eval()
        with torch.no_grad():
            self.model.net[-1].weight.fill_(0.17)
            self.model.feature_mean.fill_(0.2)
            self.model.feature_scale.fill_(0.07)
        self.exporter._write_export(self.folder, self.model.state_dict(),
                                    {"hidden": 8, "max_shift": 12.0},
                                    asdict(CompressionConfig()),
                                    {"test_fixture": "synthetic, no dataset or original checkpoint"})

    def test_roundtrip_preserves_nonzero_policy_and_normalization(self):
        clean, metadata = self.exporter.load_export(self.folder)
        x = torch.arange(16 * 9, dtype=torch.float32).reshape(9, 16) / 144
        q = torch.arange(1, 26, dtype=torch.float32).repeat(9, 1)
        self.assertTrue(torch.equal(self.model(x), clean(x)))
        self.assertTrue(torch.equal(self.model.scores(x, q), clean.scores(x, q)))
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, clean.state_dict()[name]))
        self.assertNotIn("config", metadata)

    def test_modified_archive_is_rejected(self):
        with (self.folder / self.exporter.TENSORS).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.exporter.load_export(self.folder)

    def test_changed_feature_order_is_rejected(self):
        path = self.folder / self.exporter.SPEC
        metadata = json.loads(path.read_text())
        metadata["feature_names"].reverse()
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "feature names/order"):
            self.exporter.load_export(self.folder)

    def test_historical_map_cannot_silently_replace_v2(self):
        path = self.folder / self.exporter.SPEC
        metadata = json.loads(path.read_text())
        metadata["codec"]["importance_semantics"] = "historical_full_gaussian"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "truncated_6sigma"):
            self.exporter.load_export(self.folder)
