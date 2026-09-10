"""
Comprehensive unit tests for M1 EarthDial remote backend and HTTP client handling.
All tests use mock requests and do NOT download model weights or require a GPU.
"""

import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import requests

from m1_earthdial.config import EarthDialConfig
from m1_earthdial.earthdial_adapter import EarthDialAdapter


class TestRemoteBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_image = Path("m1_earthdial/examples/sample_satellite.jpg")
        if not cls.sample_image.exists():
            from m1_earthdial.examples.generate_sample_image import create_sample_satellite_image
            create_sample_satellite_image(str(cls.sample_image))

    def test_missing_api_url_error(self):
        """Verifies that remote backend raises a clear error when EARTHDIAL_API_URL is missing."""
        config = EarthDialConfig(backend="remote", api_url="")
        adapter = EarthDialAdapter(config=config)
        
        response = adapter.analyze(str(self.sample_image), "What can you see?")
        self.assertFalse(response.success)
        self.assertIn("Missing EARTHDIAL_API_URL", response.error)
        self.assertEqual(response.answer, "")
        self.assertIsNone(response.confidence)

    @patch("requests.post")
    def test_unreachable_server_error(self, mock_post):
        """Verifies clear reporting when Colab server is unreachable (ConnectionError)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        config = EarthDialConfig(backend="remote", api_url="https://offline-tunnel.trycloudflare.com")
        adapter = EarthDialAdapter(config=config)

        response = adapter.analyze(str(self.sample_image), "What is visible?")
        self.assertFalse(response.success)
        self.assertIn("Unreachable Colab server", response.error)
        self.assertIn("https://offline-tunnel.trycloudflare.com", response.error)

    @patch("requests.post")
    def test_http_timeout_error(self, mock_post):
        """Verifies clear reporting when remote inference times out."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out after 90s")

        config = EarthDialConfig(backend="remote", api_url="https://slow-tunnel.trycloudflare.com", timeout_seconds=90)
        adapter = EarthDialAdapter(config=config)

        response = adapter.analyze(str(self.sample_image), "What is visible?")
        self.assertFalse(response.success)
        self.assertIn("Request timed out", response.error)
        self.assertIn("90 seconds", response.error)

    @patch("requests.post")
    def test_server_side_500_error(self, mock_post):
        """Verifies extraction and reporting of server-side model errors (e.g. CUDA OOM)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"detail": "CUDA out of memory during generation on Colab"}
        mock_resp.text = '{"detail": "CUDA out of memory during generation on Colab"}'
        mock_post.return_value = mock_resp

        config = EarthDialConfig(backend="remote", api_url="https://active-tunnel.trycloudflare.com")
        adapter = EarthDialAdapter(config=config)

        response = adapter.analyze(str(self.sample_image), "What is visible?")
        self.assertFalse(response.success)
        self.assertIn("Server-side EarthDial error (HTTP 500)", response.error)
        self.assertIn("CUDA out of memory", response.error)

    @patch("requests.post")
    def test_malformed_json_response_error(self, mock_post):
        """Verifies clear error reporting when server returns non-JSON text (e.g. HTML 502 Bad Gateway)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "<html>502 Bad Gateway</html>", 0)
        mock_resp.text = "<html>502 Bad Gateway</html>"
        mock_post.return_value = mock_resp

        config = EarthDialConfig(backend="remote", api_url="https://active-tunnel.trycloudflare.com")
        adapter = EarthDialAdapter(config=config)

        response = adapter.analyze(str(self.sample_image), "What is visible?")
        self.assertFalse(response.success)
        self.assertIn("Invalid API response", response.error)
        self.assertIn("non-JSON", response.error)

    @patch("requests.post")
    def test_missing_answer_key_error(self, mock_post):
        """Verifies clear reporting when server returns JSON missing the 'answer' field."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output_text": "Missing answer key"}
        mock_post.return_value = mock_resp

        config = EarthDialConfig(backend="remote", api_url="https://active-tunnel.trycloudflare.com")
        adapter = EarthDialAdapter(config=config)

        response = adapter.analyze(str(self.sample_image), "What is visible?")
        self.assertFalse(response.success)
        self.assertIn("missing the required 'answer' key", response.error)

    @patch("requests.post")
    def test_successful_remote_response(self, mock_post):
        """Verifies parsing of a successful remote EarthDial inference response."""
        expected_answer = "The satellite image shows an agricultural region with distinct field parceling and a river."
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "answer": expected_answer,
            "model": "akshaydudhane/EarthDial_4B_RGB"
        }
        mock_post.return_value = mock_resp

        config = EarthDialConfig(backend="remote", api_url="https://active-tunnel.trycloudflare.com")
        adapter = EarthDialAdapter(config=config)

        response = adapter.analyze(str(self.sample_image), "What can you see in this satellite image?")
        self.assertTrue(response.success)
        self.assertIsNone(response.error)
        self.assertEqual(response.answer, expected_answer)
        self.assertEqual(response.model, "EarthDial")
        self.assertIsNone(response.confidence)  # Must be strictly null
        self.assertEqual(response.evidence.backend, "remote_colab")
        self.assertEqual(response.evidence.image_size, (512, 512))

    @patch("requests.post")
    def test_remote_payload_construction(self, mock_post):
        """Verifies that the HTTP POST request is correctly constructed with Base64 image and params."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"answer": "Test answer"}
        mock_post.return_value = mock_resp

        config = EarthDialConfig(
            backend="remote",
            api_url="https://mock-tunnel.trycloudflare.com/",
            timeout_seconds=75
        )
        adapter = EarthDialAdapter(config=config)

        adapter.analyze(
            image_path=str(self.sample_image),
            question="Describe this scene.",
            num_beams=3,
            temperature=0.7,
            max_new_tokens=64
        )

        mock_post.assert_called_once()
        call_url, call_kwargs = mock_post.call_args
        self.assertEqual(call_url[0], "https://mock-tunnel.trycloudflare.com/analyze")
        self.assertEqual(call_kwargs["timeout"], 75)

        payload = call_kwargs["json"]
        self.assertEqual(payload["question"], "Describe this scene.")
        self.assertEqual(payload["num_beams"], 3)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_new_tokens"], 64)

        # Verify Base64 decoding matches original image bytes
        decoded_bytes = base64.b64decode(payload["image_base64"])
        with open(self.sample_image, "rb") as f:
            original_bytes = f.read()
        self.assertEqual(decoded_bytes, original_bytes)

    def test_vram_protection_on_laptop(self):
        """Verifies that direct GPU mode rejects running on GPUs with <10GB VRAM (e.g. RTX 2050 4GB)."""
        config = EarthDialConfig(backend="direct", min_vram_gb=10.0)
        adapter = EarthDialAdapter(config=config)

        # Mock torch, CUDA, and transformers with a 4GB GPU (RTX 2050)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 2050"
        mock_transformers = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            with patch.object(adapter, "_get_local_vram_gb", return_value=4.0):
                response = adapter.analyze(str(self.sample_image), "What is here?")
                self.assertFalse(response.success)
                self.assertIn("requires at least 10.0 GB GPU VRAM", response.error)
                self.assertIn("RTX 2050", response.error)
                self.assertIn("EARTHDIAL_BACKEND=remote", response.error)

    def test_m4_execute_interface(self):
        """Verifies M4 specialist execute() and __call__() interface compatibility."""
        config = EarthDialConfig(backend="mock")
        adapter = EarthDialAdapter(config=config)

        # Test execute() with land cover query
        resp1 = adapter.execute(
            image_paths=[str(self.sample_image)],
            params={"query": "What type of land cover is visible?"}
        )
        self.assertTrue(resp1.success)
        self.assertIn("land cover", resp1.answer.lower())

        # Test __call__() with water query
        resp2 = adapter(images=[str(self.sample_image)], target="Is there water?")
        self.assertTrue(resp2.success)
        self.assertIn("water", resp2.answer.lower())


if __name__ == "__main__":
    unittest.main()
