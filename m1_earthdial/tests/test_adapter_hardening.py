"""Regression tests for M1 EarthDial adapter hardening."""

import unittest
from unittest.mock import Mock, patch

from m1_earthdial.config import EarthDialConfig
from m1_earthdial.earthdial_adapter import EarthDialAdapter


class TestAdapterHardening(unittest.TestCase):
    def test_healthcheck_accepts_common_healthy_json_status(self):
        adapter = EarthDialAdapter(EarthDialConfig(backend="remote"))
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok"}

        with patch("requests.get", return_value=response):
            self.assertTrue(adapter._remote_healthcheck())

    def test_healthcheck_accepts_ready_status(self):
        adapter = EarthDialAdapter(EarthDialConfig(backend="remote"))
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ready"}

        with patch("requests.get", return_value=response):
            self.assertTrue(adapter._remote_healthcheck())

    def test_healthcheck_accepts_non_json_200_response(self):
        adapter = EarthDialAdapter(EarthDialConfig(backend="remote"))
        response = Mock(status_code=200)
        response.json.side_effect = ValueError("not json")

        with patch("requests.get", return_value=response):
            self.assertTrue(adapter._remote_healthcheck())

    def test_healthcheck_rejects_explicit_unhealthy_status(self):
        adapter = EarthDialAdapter(EarthDialConfig(backend="remote"))
        response = Mock(status_code=200)
        response.json.return_value = {"status": "unhealthy"}

        with patch("requests.get", return_value=response):
            self.assertFalse(adapter._remote_healthcheck())

    def test_healthcheck_rejects_non_200_response(self):
        adapter = EarthDialAdapter(EarthDialConfig(backend="remote"))
        response = Mock(status_code=503)

        with patch("requests.get", return_value=response):
            self.assertFalse(adapter._remote_healthcheck())


if __name__ == "__main__":
    unittest.main()
