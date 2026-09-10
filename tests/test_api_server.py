"""FastAPI contract tests for the M6 live frontend boundary."""

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


client = TestClient(app)


def _png_bytes(value: int = 50) -> bytes:
    """Create a tiny deterministic PNG for upload tests."""
    image = Image.new("L", (16, 16), value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_compatibility_alias() -> None:
    """M6's historical /api/health route must remain available."""
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "satquery-ai"


def test_live_analyze_rejects_missing_image() -> None:
    """Image-understanding mode should return a clear 400 without an upload."""
    response = client.post(
        "/api/analyze",
        data={
            "query": "Describe this image.",
            "mode": "image-understanding",
        },
    )

    assert response.status_code == 400
    assert "requires an 'image' upload" in response.json()["detail"]


def test_live_analyze_accepts_m6_multipart_contract(monkeypatch) -> None:
    """M6 multipart uploads must reach the shared live mission handler."""

    captured = {}

    def fake_run_live_mission(**kwargs):
        captured.update(kwargs)
        return {
            "status": "success",
            "answer": "stub",
            "confidence": 0.5,
        }

    monkeypatch.setattr(
        "api.main._run_live_mission",
        fake_run_live_mission,
    )

    response = client.post(
        "/api/analyze",
        data={
            "query": "Describe this image.",
            "mode": "image-understanding",
        },
        files={
            "image": (
                "scene.png",
                _png_bytes(),
                "image/png",
            ),
        },
    )

    assert response.status_code == 200
    assert captured["query"] == "Describe this image."
    assert captured["mode"] == "image-understanding"
    assert captured["image"] is not None


def test_live_change_detection_requires_both_images() -> None:
    """Change detection should reject incomplete temporal pairs."""
    response = client.post(
        "/api/analyze",
        data={
            "query": "What changed?",
            "mode": "change-detection",
        },
        files={
            "before_image": (
                "before.png",
                _png_bytes(50),
                "image/png",
            ),
        },
    )

    assert response.status_code == 400
    assert "before_image" in response.json()["detail"]
    assert "after_image" in response.json()["detail"]
