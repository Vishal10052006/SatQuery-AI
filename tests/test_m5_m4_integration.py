"""
Integration tests for M4 -> native M5 GIS execution.
"""

from pathlib import Path

from app.agents.bootstrap import build_m4_controller
from app.query.schemas import (
    ExecutionStatus,
    ToolName,
    ToolResult,
)

from data.mock.generate_mock import create_sample_mock_data


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mock"
SAMPLE_TIF = DATA_DIR / "sample.tif"
SAMPLE_MASK = DATA_DIR / "change_mask.npy"


def _ensure_mock_data() -> None:
    if not SAMPLE_TIF.exists() or not SAMPLE_MASK.exists():
        create_sample_mock_data(DATA_DIR)


def test_m5_native_m4_integration():
    """
    Verify:

        M4 ToolResult
            ->
        M5 payload conversion
            ->
        native M5 pipeline
            ->
        M4 ToolResult
    """

    _ensure_mock_data()

    controller = build_m4_controller()

    # Construct an upstream M2-style ToolResult carrying the
    # standardized fields that M5 needs.
    upstream = ToolResult(
        tool=ToolName.M2_CHANGE_DETECTION,
        status=ExecutionStatus.SUCCESS,
        confidence=0.94,
        data={
            "answer": "Deforestation detected.",
            "task": "change_detection",
            "model": "M2",
            "native_status": "success",
            "reference_geotiff": str(SAMPLE_TIF),
            "change_mask_path": str(SAMPLE_MASK),
            "bounding_boxes_pixel": [
                [40, 50, 90, 100],
            ],
            "change_detected": True,
        },
        evidence=[],
    )

    result = controller.registry.execute(
        ToolName.M5_GIS,
        previous_result=upstream,
        target="deforestation",
        output_dir="output/test_m5_m4",
    )

    assert isinstance(result, ToolResult)
    assert result.tool == ToolName.M5_GIS
    assert result.status == ExecutionStatus.SUCCESS
    assert result.confidence == 0.94

    assert result.data["model"] == "M5-GIS"
    assert result.data["target"] == "deforestation"
    assert result.data["change_detected"] is True

    assert "geographic_coordinates" in result.data
    assert "bounding_boxes" in result.data
    assert "polygons" in result.data
    assert "area" in result.data

    assert Path(result.data["evidence_path"]).exists()
    assert Path(result.data["geojson_path"]).exists()
    assert Path(result.data["map_path"]).exists()

    # Native M5 output must remain available.
    assert "m5" in result.data
    assert result.data["m5"]["target"] == "deforestation"
