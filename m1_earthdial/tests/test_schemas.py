"""
Tests for M1 EarthDial Pydantic schemas and serialization contracts.
"""

import json
import unittest
from m1_earthdial.schemas import EvidenceMetadata, ImageVQARequest, ImageVQAResponse


class TestSchemas(unittest.TestCase):

    def test_evidence_metadata_defaults(self):
        evidence = EvidenceMetadata(
            image_path="sample.jpg",
            image_format="JPEG",
            image_size=(512, 512),
            model_name="EarthDial_4B_RGB",
            backend="mock"
        )
        self.assertEqual(evidence.image_path, "sample.jpg")
        self.assertEqual(evidence.image_size, (512, 512))
        self.assertIsNotNone(evidence.timestamp)

    def test_image_vqa_request_validation(self):
        req = ImageVQARequest(
            image_path="sample.jpg",
            question="What is visible?",
            num_beams=5
        )
        self.assertEqual(req.question, "What is visible?")
        self.assertEqual(req.num_beams, 5)

    def test_image_vqa_response_confidence_is_null(self):
        """Verify that confidence is strictly null and serializes to null in JSON."""
        evidence = EvidenceMetadata(
            image_path="sample.jpg",
            image_format="JPEG",
            image_size=(512, 512),
            backend="mock"
        )
        resp = ImageVQAResponse(
            task="image_vqa",
            question="What can you see?",
            answer="Agricultural fields and a river.",
            model="EarthDial",
            confidence=None,
            evidence=evidence
        )
        data = resp.to_dict()
        self.assertIsNone(data["confidence"])

        # Test JSON serialization matches contract
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        self.assertIsNone(loaded["confidence"])
        self.assertEqual(loaded["model"], "EarthDial")
        self.assertEqual(loaded["task"], "image_vqa")
        self.assertIn("evidence", loaded)
        self.assertEqual(loaded["evidence"]["image_size"], [512, 512])


if __name__ == "__main__":
    unittest.main()

