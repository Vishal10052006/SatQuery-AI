"""
Integration and contract tests for M1 EarthDial.

The real EarthDial weights are intentionally not downloaded during unit tests.
The mock backend validates the M4 contract offline, while the remote test uses
a mocked HTTP response to verify transport and response validation.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from m1_earthdial.config import EarthDialConfig
from m1_earthdial.earthdial_adapter import (
    EarthDialAdapter,
    analyze_image,
    set_adapter,
)


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_image = Path("m1_earthdial/examples/sample_satellite.jpg")
        if not cls.sample_image.exists():
            from m1_earthdial.examples.generate_sample_image import create_sample_satellite_image

            create_sample_satellite_image(str(cls.sample_image))

        # Offline tests must never attempt to download or load the 4B model.
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
        self.assertIsNone(response.confidence)
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
        self.assertIn("max_new_tokens", result["evidence"]["extra"])

    def test_question_3_water_and_urban_features(self):
        question = "Is there a water body or river present in this scene?"
        result = analyze_image(str(self.sample_image), question)

        self.assertTrue(result["success"])
        self.assertIsNone(result["confidence"])
        self.assertIn("water", result["answer"].lower())

    def test_remote_backend_sends_expected_payload(self):
        """Verify the real remote transport without requiring a live Colab server."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "answer": "A river and agricultural parcels are visible.",
            "model": "akshaydudhane/EarthDial_4B_RGB",
        }

        adapter = EarthDialAdapter(
            config=EarthDialConfig(
                backend="remote",
                api_url="https://example-earthdial.test",
            )
        )

        with patch("requests.post", return_value=response) as post:
            result = adapter.analyze(
                str(self.sample_image),
                "What features are visible?",
                num_beams=3,
                temperature=0.0,
                max_new_tokens=64,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "A river and agricultural parcels are visible.")
        self.assertEqual(result.evidence.backend, "remote_colab")
        post.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertIn("image_base64", payload)
        self.assertEqual(payload["question"], "What features are visible?")
        self.assertEqual(payload["num_beams"], 3)
        self.assertEqual(payload["max_new_tokens"], 64)

    def test_remote_backend_rejects_empty_answer(self):
        """A successful HTTP status must still contain a usable VLM answer."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"answer": "", "model": "EarthDial"}

        adapter = EarthDialAdapter(
            config=EarthDialConfig(
                backend="remote",
                api_url="https://example-earthdial.test",
            )
        )

        with patch("requests.post", return_value=response):
            result = adapter.analyze(str(self.sample_image), "Describe the scene.")

        self.assertFalse(result.success)
        self.assertIn("empty answer", result.error.lower())

    def test_invalid_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            EarthDialAdapter(config=EarthDialConfig(backend="not-a-backend"))

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

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)

        self.assertEqual(deserialized["task"], "image_vqa")
        self.assertIsNone(deserialized["confidence"])
        self.assertIsInstance(deserialized["evidence"], dict)


if __name__ == "__main__":
    unittest.main()
