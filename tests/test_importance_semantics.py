"""Independent dense references and fixed tile-quality regressions for both semantics."""
import contextlib
import importlib.util
import io
import math
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from task_aware_compression.cli import _parser
from task_aware_compression.compressor import encode_adaptive, decode_adaptive
from task_aware_compression.importance import build_importance_map
from task_aware_compression.types import BoundingBox, CompressionConfig, Detection


def dense_reference(image, detections, config, truncate):
    # Historical full-frame equations. Truncated support is expressed as a
    # dense boolean mask, independently of the production cropped-array path.
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    importance = np.zeros((height, width), dtype=np.float32)
    for det in detections:
        box = det.box.clip(width, height)
        if box.area <= 0:
            continue
        area_ratio = det.box.area / (height * width)
        risk = .58 * min(1., math.sqrt(.018 / max(area_ratio, 1e-8))) + .42 * (1 - max(0., min(1., det.confidence)))
        sx = max(2., box.width * (.28 + .22 * risk) + config.context_radius)
        sy = max(2., box.height * (.28 + .22 * risk) + config.context_radius)
        cx, cy = box.center
        strength = .72 + .28 * risk
        field = np.exp(-.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        hard = ((xx >= box.x1 - config.context_radius) & (xx < box.x2 + config.context_radius) &
                (yy >= box.y1 - config.context_radius) & (yy < box.y2 + config.context_radius))
        field = np.maximum(field * strength, hard.astype(np.float32) * strength).astype(np.float32)
        if truncate:
            support = ((xx >= math.floor(cx - 6 * sx)) & (xx < math.ceil(cx + 6 * sx) + 1) &
                       (yy >= math.floor(cy - 6 * sy)) & (yy < math.ceil(cy + 6 * sy) + 1))
            field = np.where(support, field, np.float32(0))
        importance = np.maximum(importance, field)
    gray = image.astype(np.float32).mean(2)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    texture = np.sqrt(gx * gx + gy * gy)
    scale = float(np.percentile(texture, 90))
    if scale > 1e-6:
        importance = np.clip(importance + config.texture_weight * np.clip(texture / scale, 0, 1) * (importance > 0), 0, 1)
    return importance


class ImportanceSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(104).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        self.detections = [Detection(BoundingBox(25, 30, 33, 46), .6)]
        self.config = CompressionConfig(tile_size=16, context_radius=1, background_quality=8, person_quality=82)

    def test_historical_matches_independent_full_frame_equations(self):
        config = replace(self.config, importance_semantics="historical_full_gaussian")
        detections = self.detections + [Detection(BoundingBox(-4, 20, 10, 30), .9)]
        for texture in (0., .12):
            cfg = replace(config, texture_weight=texture)
            np.testing.assert_array_equal(build_importance_map(self.image, detections, cfg), dense_reference(self.image, detections, cfg, False))

    def test_truncated_matches_independent_dense_support_mask(self):
        config = replace(self.config, importance_semantics="truncated_6sigma")
        detections = self.detections + [Detection(BoundingBox(-4, 20, 10, 30), .9)]
        np.testing.assert_array_equal(build_importance_map(self.image, detections, config), dense_reference(self.image, detections, config, True))

    def test_support_change_amplifies_texture_instead_of_only_tiny_tails(self):
        full = replace(self.config, importance_semantics="historical_full_gaussian")
        cropped = replace(self.config, importance_semantics="truncated_6sigma")
        a = build_importance_map(self.image, self.detections, full)
        b = build_importance_map(self.image, self.detections, cropped)
        self.assertGreater(np.count_nonzero((a > 0) & (b == 0)), 0)
        self.assertGreater(float(np.max(a - b)), .05)
        no_texture_a = build_importance_map(self.image, self.detections, replace(full, texture_weight=0))
        no_texture_b = build_importance_map(self.image, self.detections, replace(cropped, texture_weight=0))
        self.assertLess(float(np.max(no_texture_a - no_texture_b)), 2e-8)

    def test_fixed_tile_quality_regressions_and_unchanged_container_schema(self):
        expected = {'historical_full_gaussian': [9, 9, 9, 9, 9, 9, 8, 8, 9, 18, 11, 9, 9, 9, 8, 8, 9, 53, 21, 9, 9, 9, 8, 8, 9, 11, 10, 9, 9, 9, 8, 8, 9, 9, 9, 9, 9, 9, 8, 8, 9, 9, 9, 9, 9, 8, 8, 8], 'truncated_6sigma': [9, 9, 9, 8, 8, 8, 8, 8, 9, 18, 11, 8, 8, 8, 8, 8, 9, 53, 21, 8, 8, 8, 8, 8, 9, 11, 10, 8, 8, 8, 8, 8, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]}
        for mode, values in expected.items():
            cfg = replace(self.config, importance_semantics=mode)
            encoded, _, qualities, header = encode_adaptive(self.image, self.detections, cfg)
            self.assertEqual(qualities, values)
            decoded, parsed = decode_adaptive(encoded)
            self.assertEqual(parsed, header)
            self.assertEqual(decoded.shape, self.image.shape)
            # Keep historical byte accounting: version identity lives in config
            # and experiment metadata, not extra bytes in the frozen container.
            self.assertEqual(set(header), {"version", "width", "height", "tile_size", "tile_workers", "tiles"})

    def test_unknown_mode_is_rejected_without_fallback(self):
        for value in ("historical", "", None, "truncated_7sigma"):
            with self.assertRaises(ValueError):
                replace(self.config, importance_semantics=value)

    def test_api_legacy_default_and_explicit_cli_selection(self):
        self.assertEqual(self.config.importance_semantics, "truncated_6sigma")
        parser = _parser()
        argv = ["workflow", "--images", "external-images", "--labels", "external-labels"]
        self.assertEqual(parser.parse_args(argv).importance_semantics, "historical_full_gaussian")
        self.assertEqual(parser.parse_args(argv + ["--importance-semantics", "truncated_6sigma"]).importance_semantics, "truncated_6sigma")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(argv + ["--importance-semantics", "unknown"])


class AllocatorSemanticsResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "scripts/evaluate_rate_allocator_v2.py"
        spec = importlib.util.spec_from_file_location("public_v2_semantics_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_provenance_verified_legacy_v2_retains_truncated_mode(self):
        original = {"tile_size": 256}
        resolved = self.module.resolve_checkpoint_compression_config(original, self.module.LEGACY_V2_CHECKPOINT_SHA256)
        self.assertEqual(resolved["importance_semantics"], "truncated_6sigma")
        self.assertNotIn("importance_semantics", original)

    def test_unknown_legacy_checkpoint_cannot_inherit_a_mode(self):
        with self.assertRaises(ValueError):
            self.module.resolve_checkpoint_compression_config({}, "unverified")

    def test_explicit_checkpoint_modes_are_honored_and_invalid_modes_rejected(self):
        for mode in ("historical_full_gaussian", "truncated_6sigma"):
            resolved = self.module.resolve_checkpoint_compression_config({"importance_semantics": mode}, "new-artifact")
            self.assertEqual(resolved["importance_semantics"], mode)
        with self.assertRaises(ValueError):
            self.module.resolve_checkpoint_compression_config({"importance_semantics": "unknown"}, "new-artifact")
