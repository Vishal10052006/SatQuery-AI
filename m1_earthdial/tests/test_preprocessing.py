"""
Unit tests for M1 EarthDial image preprocessing and validation.
"""

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from m1_earthdial.preprocessing import (
    validate_image_path,
    load_and_preprocess_image,
    ImagePreprocessingError,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a valid test image
        self.valid_image_path = self.temp_path / "valid_sat.jpg"
        img = Image.new("RGB", (256, 256), color=(50, 150, 50))
        img.save(self.valid_image_path, format="JPEG")

        # Create an empty file (0 bytes)
        self.empty_file_path = self.temp_path / "empty.jpg"
        self.empty_file_path.touch()

        # Create an unsupported extension file
        self.unsupported_file_path = self.temp_path / "sat_data.pdf"
        self.unsupported_file_path.write_text("Dummy PDF content")

        # Create a corrupt image file
        self.corrupt_file_path = self.temp_path / "corrupt.png"
        self.corrupt_file_path.write_bytes(b"This is not a real PNG image header")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_image_file(self):
        non_existent = self.temp_path / "does_not_exist.jpg"
        with self.assertRaises(ImagePreprocessingError) as ctx:
            validate_image_path(str(non_existent))
        self.assertIn("not found", str(ctx.exception).lower())

    def test_empty_image_file(self):
        with self.assertRaises(ImagePreprocessingError) as ctx:
            validate_image_path(str(self.empty_file_path))
        self.assertIn("empty", str(ctx.exception).lower())

    def test_unsupported_extension(self):
        with self.assertRaises(ImagePreprocessingError) as ctx:
            validate_image_path(str(self.unsupported_file_path))
        self.assertIn("unsupported", str(ctx.exception).lower())

    def test_corrupt_image_file(self):
        with self.assertRaises(ImagePreprocessingError) as ctx:
            load_and_preprocess_image(str(self.corrupt_file_path))
        self.assertIn("corrupt", str(ctx.exception).lower())

    def test_valid_image_loading_and_metadata(self):
        img, metadata = load_and_preprocess_image(str(self.valid_image_path))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(metadata["width"], 256)
        self.assertEqual(metadata["height"], 256)
        self.assertEqual(metadata["mode"], "RGB")
        self.assertEqual(metadata["original_format"], "JPEG")


if __name__ == "__main__":
    unittest.main()

