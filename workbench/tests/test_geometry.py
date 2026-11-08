"""Synthetic geometry fixtures are unit tests, never research observations."""
import unittest
import numpy as np
from research_platform.geometry import map_box, restore_mask, application_box


class GeometryTests(unittest.TestCase):
    def test_actual_rounded_dimensions(self):
        np.testing.assert_allclose(map_box([10, 20, 90, 70], (101, 81), (51, 40)), [10*51/101, 20*40/81, 90*51/101, 70*40/81])

    def test_nearest_binary_restore(self):
        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        restored = restore_mask(mask, (4, 4))
        self.assertEqual(restored.shape, (4, 4))
        self.assertEqual(int(restored.sum()), 8)

    def test_application_no_reference_argument(self):
        self.assertEqual(application_box(np.zeros((100, 200, 3), np.uint8)).tolist(), [20, 10, 180, 90])

    def test_invalid_box(self):
        with self.assertRaises(ValueError):
            map_box([0, 0, 0, 2], (4, 4), (2, 2))
