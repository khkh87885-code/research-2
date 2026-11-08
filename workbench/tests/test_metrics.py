"""Analytical fixtures only; these are not research images or empirical results."""
import math
import unittest

import numpy as np

from research_platform.metrics import compression_efficiency, evaluate, stability


class MetricTests(unittest.TestCase):
    def test_known_overlapping_rectangles(self):
        reference = np.zeros((8, 9), dtype=bool)
        prediction = reference.copy()
        reference[1:4, 1:4] = True
        prediction[1:4, 2:5] = True
        metrics = evaluate(reference, prediction, boundary_tolerance=0)
        self.assertAlmostEqual(metrics["dice"], 2 / 3)
        self.assertAlmostEqual(metrics["iou"], .5)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertEqual(metrics["sensitivity"], metrics["recall"])
        self.assertAlmostEqual(metrics["hd95"], 1)
        self.assertAlmostEqual(metrics["assd"], .5)
        self.assertAlmostEqual(metrics["boundary_f1"], .5)
        self.assertEqual(metrics["relative_area_error"], 0)

    def test_single_points_distance_and_physical_spacing(self):
        reference = np.zeros((7, 8), dtype=bool)
        prediction = reference.copy()
        reference[1, 1] = True
        prediction[4, 5] = True
        metrics = evaluate(reference, prediction, boundary_tolerance=5)
        self.assertEqual(metrics["hd95"], 5)
        self.assertEqual(metrics["assd"], 5)
        self.assertEqual(metrics["boundary_f1"], 1)
        self.assertEqual(evaluate(reference, prediction, boundary_tolerance=4.9)["boundary_f1"], 0)
        physical = evaluate(reference, prediction, spacing=(2, 1))
        self.assertAlmostEqual(physical["hd95"], math.sqrt(52))
        self.assertEqual(physical["boundary_unit"], "mm")

    def test_identical_full_border_mask(self):
        mask = np.ones((5, 7), dtype=bool)
        metrics = evaluate(mask, mask)
        for name in ("dice", "iou", "precision", "recall", "boundary_f1"):
            self.assertEqual(metrics[name], 1)
        self.assertEqual(metrics["hd95"], 0)
        self.assertEqual(metrics["assd"], 0)

    def test_pooled_boundary_weighting_is_explicit(self):
        reference = np.zeros((5, 5), dtype=bool)
        prediction = reference.copy()
        reference[2, 2] = True
        prediction[1:4, 1:4] = True
        metrics = evaluate(reference, prediction)
        # Eight predicted boundary pixels plus one reference boundary pixel:
        # five directed distances are 1, four are sqrt(2).
        self.assertAlmostEqual(metrics["hd95"], math.sqrt(2))
        self.assertAlmostEqual(metrics["assd"], (5 + 4 * math.sqrt(2)) / 9)
        self.assertEqual(metrics["relative_area_error"], 8)

    def test_empty_reference_does_not_create_perfect_scores(self):
        empty = np.zeros((4, 5), dtype=bool)
        output = evaluate(empty, empty)
        for name in ("dice", "iou", "precision", "recall", "hd95", "assd", "boundary_f1", "relative_area_error"):
            self.assertIsNone(output[name])
        self.assertEqual(output["metric_status"], "invalid_reference_empty")
        self.assertIsNone(stability(empty, empty)["consistency_dice"])

    def test_empty_prediction_retained_as_failure(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        output = evaluate(mask, np.zeros_like(mask))
        self.assertEqual(output["dice"], 0)
        self.assertEqual(output["iou"], 0)
        self.assertEqual(output["recall"], 0)
        self.assertEqual(output["relative_area_error"], -1)
        self.assertIsNone(output["precision"])
        self.assertIsNone(output["hd95"])
        self.assertEqual(output["metric_reasons"]["hd95"], "empty_prediction")

    def test_stability_area_and_empty_baseline(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:3, 1:3] = True
        prediction = mask.copy()
        prediction[3, 1:3] = True
        output = stability(mask, prediction)
        self.assertAlmostEqual(output["consistency_dice"], .8)
        self.assertAlmostEqual(output["prediction_area_change_rate"], .5)
        empty_baseline = stability(np.zeros_like(mask), prediction)
        self.assertEqual(empty_baseline["consistency_dice"], 0)
        self.assertIsNone(empty_baseline["prediction_area_change_rate"])

    def test_rejects_coordinates_values_spacing(self):
        for reference, prediction in ((np.zeros((2, 2)), np.zeros((2, 3))),
                                      (np.full((2, 2), 255), np.zeros((2, 2)))):
            with self.assertRaises(ValueError):
                evaluate(reference, prediction)
        for spacing in ((0, 1), (-1, 1), (1,), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                evaluate(np.ones((2, 2)), np.ones((2, 2)), spacing=spacing)

    def test_compression_formula_and_simulated_time(self):
        output = compression_efficiency(1000000, 250000, bandwidth_mbps=8, rtt_ms=20)
        self.assertEqual(output["compression_ratio"], .25)
        self.assertEqual(output["compression_saving_pct"], 75)
        self.assertAlmostEqual(output["simulated_upload_time_s"], .27)
        self.assertAlmostEqual(output["simulated_original_upload_time_s"], 1.02)
        self.assertEqual(compression_efficiency(100, 200)["compression_saving_pct"], -100)
        for args in ((0, 20), (100, 0), (100, 20, 0), (100, 20, 10, -1), (1.5, 1)):
            with self.assertRaises(ValueError):
                compression_efficiency(*args)


if __name__ == "__main__":
    unittest.main()
