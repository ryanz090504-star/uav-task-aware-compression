import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from task_aware_compression.compressor import compress_adaptive, compress_budgeted, compress_floor_constrained, compress_uniform, deblock_tile_seams, decode_adaptive, _quality_for_importance
from task_aware_compression.benchmark import _match_uniform_quality, _random_roi_detections, _record, summarize_records
from task_aware_compression.detector import ColorBlobPersonDetector
from task_aware_compression.importance import build_importance_map, tile_importance_map, _box_risk
from task_aware_compression.metrics import evaluate_detections, dataset_average_precision
from task_aware_compression.synthetic import make_synthetic_dataset
from task_aware_compression.types import BoundingBox, CompressionConfig, Detection

class CompressionTests(unittest.TestCase):
    def test_budgeted_codec_round_trip_and_byte_target(self):
        image = np.random.default_rng(7).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        detection = Detection(BoundingBox(40, 20, 66, 78), 0.2)
        config = CompressionConfig(tile_size=32, background_quality=8, person_quality=82)
        baseline = compress_adaptive(image, [detection], config)
        budgeted = compress_budgeted(image, [detection], baseline.file_size, config)
        decoded, _ = decode_adaptive(budgeted.encoded)
        np.testing.assert_array_equal(decoded, budgeted.image)
        self.assertLessEqual(budgeted.file_size, baseline.file_size)
        self.assertGreater(len(set(budgeted.tile_qualities)), 1)

    def test_floor_constrained_search_uses_byte_target(self):
        image = np.random.default_rng(8).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        detection = Detection(BoundingBox(40, 20, 66, 78), 0.2)
        config = CompressionConfig(tile_size=32, background_quality=8, person_quality=82)
        target = compress_adaptive(image, [detection], config).file_size
        result = compress_floor_constrained(image, [detection], target, 10, config)
        self.assertEqual(result.metadata["target_bytes"], target)
        self.assertGreaterEqual(result.metadata["selected_foreground_quality"], 10)
        self.assertLess(abs(result.file_size / target - 1), 0.1)

    def test_deblock_changes_only_seam_pixels(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, 8:] = 100
        output = deblock_tile_seams(image, 8, 1.0)
        np.testing.assert_array_equal(output[:, :7], image[:, :7])
        np.testing.assert_array_equal(output[:, 9:], image[:, 9:])
        self.assertNotEqual(int(output[3, 7, 0]), int(image[3, 7, 0]))

    def test_adaptive_round_trip_and_metadata(self):
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[:, :, :] = [90, 120, 150]
        detection = Detection(BoundingBox(40, 20, 66, 78), 0.2)
        result = compress_adaptive(image, [detection], CompressionConfig(tile_size=32, tile_workers=2))
        decoded, header = decode_adaptive(result.encoded)
        self.assertEqual(decoded.shape, image.shape)
        self.assertEqual(header["width"], 128)
        self.assertGreater(result.file_size, 0)
        self.assertGreaterEqual(max(result.tile_qualities), min(result.tile_qualities))
        self.assertTrue(np.isfinite(result.importance).all())
        self.assertEqual(result.metadata["tile_workers"], 2)

    def test_shifted_tiles_cover_image_once(self):
        image = np.random.default_rng(11).integers(0, 256, (93, 117, 3), dtype=np.uint8)
        config = CompressionConfig(tile_size=32, tile_offset_x=16, tile_offset_y=16)
        result = compress_adaptive(image, [], config)
        decoded, header = decode_adaptive(result.encoded)
        np.testing.assert_array_equal(decoded, result.image)
        coverage = np.zeros(image.shape[:2], dtype=np.uint8)
        for tile in header["tiles"]:
            coverage[tile["y"]:tile["y"] + tile["h"], tile["x"]:tile["x"] + tile["w"]] += 1
        self.assertTrue(np.all(coverage == 1))

    def test_uniform_round_trip(self):
        image = np.random.default_rng(3).integers(0, 256, (64, 80, 3), dtype=np.uint8)
        result = compress_uniform(image, 35)
        self.assertEqual(result.image.shape, image.shape)
        self.assertGreater(result.compression_ratio, 1.0)

    def test_rate_match_selects_closest_candidate(self):
        candidates = {
            20: SimpleNamespace(file_size=8_200),
            21: SimpleNamespace(file_size=8_010),
            22: SimpleNamespace(file_size=8_090),
        }
        quality, result = _match_uniform_quality(8_000, candidates)
        self.assertEqual(quality, 21)
        self.assertEqual(result.file_size, 8_010)

    def test_adaptive_latency_includes_candidate_detection(self):
        sample = make_synthetic_dataset(1, width=320, height=192)[0]
        result = SimpleNamespace(
            image=sample.image,
            file_size=100,
            raw_size=sample.image.nbytes,
            compression_ratio=sample.image.nbytes / 100,
            bits_per_pixel=100 * 8 / (320 * 192),
            encode_ms=7.0,
            decode_ms=3.0,
        )
        record = _record(sample, "adaptive", "test", result, [], 5.0, candidate_detection_ms=11.0, paired_quality=20, paired_target_file_size_bytes=100, paired_size_error_bytes=2)
        self.assertEqual(record["incremental_pipeline_ms"], 15.0)
        self.assertEqual(record["end_to_end_ms"], 26.0)
        self.assertAlmostEqual(record["paired_size_error_fraction"], 0.02)

class DetectionTests(unittest.TestCase):
    def test_dataset_ap_pools_scores_across_images(self):
        box = BoundingBox(1, 1, 9, 9)
        truth = Detection(box)
        samples = [
            ([Detection(box, 0.9)], [truth]),
            ([Detection(BoundingBox(20, 20, 30, 30), 0.8), Detection(box, 0.7)], [truth]),
        ]
        self.assertAlmostEqual(dataset_average_precision(samples), 5 / 6)

    def test_synthetic_detector_and_metrics(self):
        sample = make_synthetic_dataset(1, width=320, height=192)[0]
        detector = ColorBlobPersonDetector()
        predictions = detector.predict(sample.image)
        metrics = evaluate_detections(predictions, sample.ground_truth, sample.image.shape[:2])
        self.assertGreaterEqual(metrics["recall"], 0.5)
        self.assertGreaterEqual(metrics["mAP50"], 0.0)
        self.assertIn("ap_small", metrics)
        self.assertIn("small_recall", metrics)

    def test_random_roi_baseline_is_deterministic(self):
        detections = [Detection(BoundingBox(10, 20, 30, 60), 0.7)]
        first = _random_roi_detections("frame-001", (100, 160), detections)
        second = _random_roi_detections("frame-001", (100, 160), detections)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0].box.width, detections[0].box.width)
        self.assertAlmostEqual(first[0].box.height, detections[0].box.height)

    def test_importance_is_bounded_and_protects_box(self):
        image = np.full((80, 100, 3), 128, dtype=np.uint8)
        det = Detection(BoundingBox(42, 28, 56, 68), 0.15)
        importance = build_importance_map(image, [det])
        self.assertGreater(float(importance[40, 48]), float(importance[5, 5]))
        self.assertGreaterEqual(float(importance.min()), 0.0)
        self.assertLessEqual(float(importance.max()), 1.0)

    def test_local_importance_matches_full_field_tile_scores(self):
        image = np.full((140, 180, 3), 128, dtype=np.uint8)
        det = Detection(BoundingBox(43, 23, 51, 53), 0.3)
        config = CompressionConfig(tile_size=32, texture_weight=0)
        actual = build_importance_map(image, [det], config)
        yy, xx = np.mgrid[:140, :180]
        risk = _box_risk(det, (140, 180), config.small_object_reference)
        sx = det.box.width * (0.28 + 0.22 * risk) + config.context_radius
        sy = det.box.height * (0.28 + 0.22 * risk) + config.context_radius
        cx, cy = det.box.center
        full = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        hard = (xx >= det.box.x1 - config.context_radius) & (xx < det.box.x2 + config.context_radius) & (yy >= det.box.y1 - config.context_radius) & (yy < det.box.y2 + config.context_radius)
        strength = 0.72 + 0.28 * risk
        expected = np.maximum(full * strength, hard.astype(np.float32) * strength)
        self.assertLess(float(np.max(np.abs(actual - expected))), 5e-8)
        actual_tiles = tile_importance_map(actual, 32)
        expected_tiles = tile_importance_map(expected, 32)
        np.testing.assert_allclose(actual_tiles, expected_tiles, atol=1e-6)
        self.assertEqual([_quality_for_importance(x, config) for x in actual_tiles.flat], [_quality_for_importance(x, config) for x in expected_tiles.flat])

    def test_tile_pooling_orders_sparse_protection(self):
        importance = np.zeros((32, 32), dtype=np.float32)
        importance[:8, :8] = 1.0
        scores = [float(tile_importance_map(importance, 32, mode)[0, 0]) for mode in ("mean", "blend25", "blend50", "p95", "max")]
        self.assertEqual(scores, sorted(scores))

class DatasetTests(unittest.TestCase):
    def test_yolo_dataset_loader(self):
        from task_aware_compression.dataset import YoloPersonDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images = root / "images"; labels = root / "labels"; images.mkdir(); labels.mkdir()
            Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(images / "frame.jpg")
            (labels / "frame.txt").write_text("0 0.5 0.5 0.2 0.4\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
            metadata = root / "metadata"; metadata.mkdir()
            (metadata / "frame.json").write_text(json.dumps({"persons": [{"occlusion": 2, "truncation": 1}]}), encoding="utf-8")
            sample = next(iter(YoloPersonDataset(images, labels, metadata_dir=metadata)))
            self.assertEqual(len(sample.ground_truth), 1)
            self.assertAlmostEqual(sample.ground_truth[0].box.width, 40.0)
            self.assertEqual(sample.metadata["max_occlusion"], 2.0)

    def test_summary_reports_pairing_error_and_strata(self):
        base = {"sample": "frame", "recall": 0.6, "precision": 0.5, "mAP50": 0.5, "ap_small": 0.5, "small_recall": 0.6, "miss_rate": 0.4, "bits_per_pixel": 0.4, "compression_ratio": 5.0, "file_size_bytes": 100, "candidate_detection_ms": 2.0, "encode_ms": 4.0, "decode_ms": 1.0, "detection_ms": 3.0, "incremental_pipeline_ms": 8.0, "end_to_end_ms": 10.0, "fps": 100.0, "target_count": 1, "small_target_count": 1, "mean_target_area": 0.01, "max_occlusion": 2, "mean_occlusion": 1.0, "max_truncation": 0, "mean_truncation": 0.0}
        adaptive = {**base, "mode": "adaptive", "setting": "risk_aware_adaptive_test", "paired_size_error_fraction": 0.01}
        paired = {**base, "mode": "uniform_paired", "setting": "paired_risk_aware_adaptive_test", "recall": 0.5, "paired_size_error_fraction": 0.01}
        summary = summarize_records([adaptive, paired])
        self.assertEqual(summary["paired_rate_matching"]["risk_aware_adaptive_test"]["n"], 1)
        self.assertEqual(summary["paired_strata"]["risk_aware_adaptive_test"]["max_occlusion"]["occlusion_2"]["n"], 1)

if __name__ == "__main__":
    unittest.main()

