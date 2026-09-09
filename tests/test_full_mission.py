"""Smoke and integration tests for the M1-M4 working foundation."""
from pathlib import Path

from PIL import Image

from mission.orchestrator import run_mission


def test_scene_query_routes_to_vqa():
    result = run_mission("Describe this satellite image")
    assert result["plan"] == ["vqa"]
    assert result["status"] == "awaiting_models"


def test_change_query_exposes_composed_temporal_pipeline():
    result = run_mission("Find changes between optical and SAR images from 2024 to 2026")
    assert result["query_spec"]["start_date"] == "2024-01-01"
    assert result["query_spec"]["end_date"] == "2026-01-01"
    assert "change_detection" in result["plan"]
    assert "optical_sar" in result["plan"]


def test_working_change_baseline(tmp_path: Path):
    """The demo detects a deliberately changed region without model weights."""
    before = Image.new("L", (32, 32), 0)
    after = Image.new("L", (32, 32), 0)
    for x in range(10, 20):
        for y in range(10, 20):
            after.putpixel((x, y), 255)
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    before.save(before_path)
    after.save(after_path)

    result = run_mission(
        "Find changes between the two images",
        {
            "before": str(before_path),
            "after": str(after_path),
            "output_dir": str(tmp_path / "artifacts"),
        },
    )
    change = next(item for item in result["results"] if item["task"] == "change_detection")
    assert change["status"] == "success"
    assert change["evidence"]["changed_pixels"] == 100
    assert change["evidence"]["regions"][0]["pixel_count"] == 100
    assert len(change["artifacts"]) == 1
    assert (tmp_path / "artifacts" / change["artifacts"][0]).exists()
