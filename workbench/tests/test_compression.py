"""Artificial images below are codec test fixtures, not research cases."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from research_platform.compression import compress, recipes, capabilities
from research_platform.common import sha256


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "originals").mkdir()
        self.source = self.root / "originals" / "original.png"
        Image.fromarray(np.random.default_rng(4).integers(0, 256, (61, 83), dtype=np.uint8)).save(self.source)
        self.digest = sha256(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def encode(self, key):
        result = compress(self.source, self.root / key, "fixture", recipes()[key])
        self.assertEqual(self.digest, sha256(self.source))
        return result

    def test_baseline_identical(self):
        r = self.encode("baseline")
        self.assertEqual(r["processed_sha256"], self.digest)

    def test_png_lossless(self):
        self.assertTrue(self.encode("png")["decoded_pixels_equal_original"])

    def test_jpeg_qualities(self):
        sizes = [self.encode("jpeg_q%d" % q)["compressed_bytes"] for q in (95,85,75,60,40)]
        self.assertGreater(sizes[0], sizes[-1])

    def test_scale_and_order(self):
        for key in ("scale_50", "scale_50_bicubic", "cr2_q75_s50", "cr3_q75_s50"):
            r = self.encode(key)
            self.assertEqual((r["width"], r["height"]), (42, 30))
        self.assertIn("JPEG encode+decode", self.encode("cr3_q85_s50")["operations"][1])

    def test_high_bit_rejected(self):
        image = self.root / "originals" / "16.png"
        Image.fromarray(np.full((10, 10), 1024, dtype=np.uint16)).save(image)
        with self.assertRaises(ValueError):
            compress(image, self.root / "encoded", "16bit", recipes()["jpeg_q85"])

    def test_preserve_alpha_png(self):
        path = self.root / "originals" / "alpha.png"
        Image.new("RGBA", (10,10), (220,50,25,64)).save(path)
        r = compress(path, self.root / "alpha_out", "alpha", recipes()["png"])
        self.assertTrue(r["decoded_pixels_equal_original"])

    def test_original_protection(self):
        with self.assertRaises(ValueError):
            compress(self.source, self.source.parent, "fixture", recipes()["png"])
        self.encode("png")
        with self.assertRaises(FileExistsError):
            self.encode("png")

    def test_optional_codecs(self):
        caps = capabilities()
        for key, cap in (("webp_lossless","webp"),("webp_q85","webp"),("jp2_lossless","jpeg2000"),("jp2_lossy","jpeg2000")):
            if caps[cap]:
                self.encode(key)
