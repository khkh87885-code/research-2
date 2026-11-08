"""Deterministic analytical/synthetic fixtures, excluded from scientific outputs."""
import json
import unittest

import numpy as np

from research_platform.statistics import analyze, holm_adjust, mixed_effects, patient_bootstrap


def row(patient, image, variant, value, **kwargs):
    result = {"dataset": "fixture", "patient_id": patient, "image_id": image,
              "variant": variant, "dice": value, "model": "fixture_model",
              "prompt_strategy": "mechanism", "split": "development"}
    result.update(kwargs)
    return result


class StatisticsTests(unittest.TestCase):
    def test_patient_confirmation_gate(self):
        result = analyze([row("case1", "image1", "baseline", .8)])
        self.assertEqual(result["status"], "descriptive_only")
        self.assertEqual(result["groups"][0]["inference_status"], "blocked_unverified_patient_ids")
        self.assertEqual(result["groups"][0]["contrasts"], [])
        self.assertEqual(mixed_effects([])["status"], "blocked_unverified_patient_ids")

    def test_equal_patient_weight_and_same_image_pairing(self):
        rows = [row("p1", "a", "baseline", .9), row("p1", "a", "jpeg", .8),
                row("p1", "b", "baseline", .8), row("p1", "b", "jpeg", .7),
                row("p2", "c", "baseline", .8), row("p2", "c", "jpeg", .9),
                row("p1", "unpaired", "jpeg", .2)]
        result = analyze(rows, patient_ids_verified=True, n_boot=500)
        contrast = result["groups"][0]["contrasts"][0]
        self.assertEqual(contrast["n_paired_patients"], 2)
        self.assertEqual(contrast["n_paired_images"], 3)
        self.assertEqual(contrast["n_variant_rows_without_valid_baseline_pair"], 1)
        self.assertAlmostEqual(contrast["patient_difference"]["mean"], 0)
        self.assertEqual(contrast["noninferiority"]["status"], "not_configured")

    def test_bootstrap_retains_duplicate_draws(self):
        output = patient_bootstrap([0, 100], seed=20, n_boot=500)
        self.assertEqual(output["ci95"], [0, 100])
        self.assertEqual(output, patient_bootstrap([0, 100], seed=20, n_boot=500))
        self.assertEqual(patient_bootstrap([1], n_boot=100)["status"], "insufficient_patients")

    def test_holm_known_values_and_missing(self):
        adjusted = holm_adjust([.01, .04, .03, None])
        self.assertEqual(adjusted, [.04, .09, .09, None])
        with self.assertRaises(ValueError):
            holm_adjust([1.5])

    def test_noninferiority_requires_explicit_margin_and_correct_direction(self):
        rows = []
        for patient, difference in enumerate((-.006, -.004, -.002, 0, .002, .004)):
            rows.extend((row(patient, "i", "baseline", .8), row(patient, "i", "jpeg", .8 + difference)))
        result = analyze(rows, delta=.02, patient_ids_verified=True, n_boot=500)
        contrast = result["groups"][0]["contrasts"][0]
        self.assertTrue(contrast["noninferiority"]["meets_prespecified_criterion"])
        self.assertEqual(contrast["noninferiority"]["criterion"], "CI95 lower > -margin")
        poor = [dict(item, dice=item["dice"] - .1) if item["variant"] == "jpeg" else item for item in rows]
        result = analyze(poor, delta=.02, patient_ids_verified=True, n_boot=500)
        self.assertFalse(result["groups"][0]["contrasts"][0]["noninferiority"]["meets_prespecified_criterion"])
        hd_rows = [dict(item, hd95=1 + item["dice"]) for item in rows]
        result = analyze(hd_rows, metric="hd95", delta=.05, patient_ids_verified=True, n_boot=500)
        self.assertEqual(result["groups"][0]["contrasts"][0]["noninferiority"]["criterion"], "CI95 upper < margin")
        for delta in (0, -.02, float("nan")):
            with self.assertRaises(ValueError):
                analyze(rows, delta=delta)

    def test_strata_and_duplicate_protection(self):
        rows = [row("same", "i", "baseline", .8), row("same", "i", "jpeg", .7),
                row("same", "i", "baseline", .6, dataset="other"),
                row("same", "i", "jpeg", .7, dataset="other")]
        result = analyze(rows, patient_ids_verified=True, n_boot=100)
        self.assertEqual(len(result["groups"]), 2)
        rows.append(rows[0].copy())
        result = analyze(rows, patient_ids_verified=True, n_boot=100)
        self.assertEqual(result["groups"][0]["inference_status"], "blocked_duplicate_patient_image_variant_rows")

    def test_missing_values_reported_and_json_finite(self):
        rows = [row(1, "i", "baseline", .8), row(1, "i", "jpeg", None),
                row(2, "j", "baseline", .7), row(2, "j", "jpeg", float("nan"))]
        result = analyze(rows, patient_ids_verified=True, n_boot=100)
        self.assertEqual(result["groups"][0]["descriptive"]["jpeg"]["n_missing_or_invalid"], 2)
        self.assertEqual(result["groups"][0]["contrasts"][0]["n_paired_patients"], 0)
        json.dumps(result, allow_nan=False)

    def test_friedman_uses_complete_patient_blocks(self):
        rows = []
        for patient in range(6):
            for variant, difference in (("baseline", 0), ("jpeg", -.01), ("png", .01)):
                rows.append(row(patient, "i", variant, .8 + patient * .001 + difference))
        result = analyze(rows, patient_ids_verified=True, n_boot=100)
        friedman = result["groups"][0]["friedman"]
        self.assertEqual(friedman["n_complete_patients"], 6)
        self.assertEqual(friedman["status"], "estimated")
        self.assertAlmostEqual(friedman["statistic"], 12)

    def test_degenerate_difference_does_not_fabricate_t_or_ni(self):
        rows = []
        for patient in range(4):
            rows.extend((row(patient, "i", "baseline", .8), row(patient, "i", "jpeg", .8)))
        result = analyze(rows, delta=.02, patient_ids_verified=True, n_boot=100)
        contrast = result["groups"][0]["contrasts"][0]
        self.assertIsNone(contrast["paired_t"]["p_value"])
        self.assertIsNone(contrast["cohen_dz"])
        self.assertIsNone(contrast["noninferiority"]["meets_prespecified_criterion"])

    def test_mixed_effects_rank_deficiency_and_small_sample(self):
        self.assertEqual(mixed_effects([], patient_ids_verified=True)["status"], "no_data")
        rows = [row(i, "image", "baseline", .7, compression_format="original", resolution_scale=1) for i in range(4)]
        result = mixed_effects(rows, patient_ids_verified=True)
        self.assertEqual(result["groups"][0]["status"], "rank_deficient")

    def test_mixed_effects_nested_images_actual_fit(self):
        rng = np.random.RandomState(12)
        rows = []
        for patient in range(16):
            intercept = rng.normal(0, .035)
            for image in range(2):
                image_intercept = rng.normal(0, .015)
                for fmt in ("jpeg", "png"):
                    for scale in (.5, 1):
                        value = .75 + intercept + image_intercept + .06 * scale - .02 * (fmt == "jpeg") + rng.normal(0, .004)
                        rows.append(row(patient, str(image), "%s_%s" % (fmt, scale), value,
                                        compression_format=fmt, resolution_scale=scale))
        result = mixed_effects(rows, patient_ids_verified=True)
        fitted = result["groups"][0]
        self.assertEqual(fitted["status"], "converged", fitted)
        self.assertEqual(fitted["random_effects"], "patient intercept + nested image intercept")
        self.assertAlmostEqual(fitted["fixed_effects"]["resolution_scale"]["estimate"], .06, delta=.02)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
