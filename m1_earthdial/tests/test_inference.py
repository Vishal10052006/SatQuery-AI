"""
End-to-end integration tests for M1 EarthDial inference and error handling.
Validates the clean M4 interface with multiple natural language EO questions.
"""

import json
import unittest
from pathlib import Path

from m1_earthdial.earthdial_adapter import EarthDialAdapter, analyze_image, set_adapter
from m1_earthdial.config import EarthDialConfig


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_image = Path("m1_earthdial/examples/sample_satellite.jpg")
        if not cls.sample_image.exists():
            from m1_earthdial.examples.generate_sample_image import create_sample_satellite_image
            create_sample_satellite_image(str(cls.sample_image))

        # Explicitly configure mock backend for offline test suite
        cls.config = EarthDialConfig(backend="mock")
        cls.adapter = EarthDialAdapter(config=cls.config)
        set_adapter(cls.adapter)

    @classmethod
    def tearDownClass(cls):
        set_adapter(None)

    def test_question_1_general_overview(self):
        question = "What can you see in this satellite image?"
        response = self.adapter.analyze(str(self.sample_image), question)
        
        self.assertTrue(response.success)
        self.assertIsNone(response.error)
        self.assertEqual(response.task, "image_vqa")
        self.assertEqual(response.model, "EarthDial")
        self.assertIsNone(response.confidence)  # Must be strictly null
        self.assertIsInstance(response.answer, str)
        self.assertGreater(len(response.answer), 10)
        self.assertEqual(response.evidence.image_size, (512, 512))

    def test_question_2_land_cover_features(self):
        question = "What are the major features and land cover types visible in this image?"
        result = analyze_image(str(self.sample_image), question)

        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["task"], "image_vqa")
        self.assertEqual(result["question"], question)
        self.assertIsNone(result["confidence"])
        self.assertIn("land cover", result["answer"].lower())
        self.assertIn("evidence", result)
        self.assertEqual(result["evidence"]["image_format"], "JPEG")

    def test_question_3_water_and_urban_features(self):
        question = "Is there a water body or river present in this scene?"
        result = analyze_image(str(self.sample_image), question)

        self.assertTrue(result["success"])
        self.assertIsNone(result["confidence"])
        self.assertIn("water", result["answer"].lower())

    def test_error_missing_image(self):
        non_existent = "m1_earthdial/examples/non_existent_satellite.jpg"
        result = analyze_image(non_existent, "What is here?")

        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"].lower())
        self.assertEqual(result["answer"], "")
        self.assertIsNone(result["confidence"])

    def test_error_invalid_question(self):
        result = analyze_image(str(self.sample_image), "")

        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIn("invalid question", result["error"].lower())
        self.assertEqual(result["answer"], "")

    def test_json_serializability(self):
        question = "Describe the major objects in this scene."
        result = analyze_image(str(self.sample_image), question)
        
        # Verify strict JSON compliance
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        
        self.assertEqual(deserialized["task"], "image_vqa")
        self.assertIsNone(deserialized["confidence"])
        self.assertIsInstance(deserialized["evidence"], dict)


if __name__ == "__main__":
    unittest.main()

