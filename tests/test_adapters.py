"""Tests for M4 specialist adapters."""

from types import SimpleNamespace

from app.adapters.common import convert_tool_output
from app.query.schemas import (
    ExecutionStatus,
    ToolName,
)


def test_convert_external_tool_output() -> None:
    """External ToolOutput should become M4 ToolResult."""

    raw = SimpleNamespace(
        answer="Detected urban change.",
        confidence=0.91,
        metrics={
            "changed_area_m2": 1250
        },
        warnings=[],
        mask_geojson={
            "type": "FeatureCollection",
            "features": [],
        },
        bboxes=None,
    )

    result = convert_tool_output(
        raw,
        ToolName.M2_CHANGE_DETECTION,
    )

    assert result.tool == ToolName.M2_CHANGE_DETECTION
    assert result.status == ExecutionStatus.SUCCESS
    assert result.confidence == 0.91
    assert result.data["answer"] == "Detected urban change."
    assert result.data["metrics"]["changed_area_m2"] == 1250
    assert len(result.evidence) == 1


def test_m5_gis_adapter_converts_external_output() -> None:
    """External M5 GIS output should become ToolResult."""

    from types import SimpleNamespace

    from app.adapters.m5_gis_adapter import adapt_m5_gis_output
    from app.query.schemas import (
        ExecutionStatus,
        ToolName,
    )

    raw = SimpleNamespace(
        answer="Geospatial regions generated.",
        confidence=0.87,
        metrics={
            "area_m2": 4200,
        },
        warnings=[],
        mask_geojson={
            "type": "FeatureCollection",
            "features": [],
        },
        bboxes=[
            [10, 20, 30, 40],
        ],
    )

    result = adapt_m5_gis_output(
        raw,
    )

    assert result.tool == ToolName.M5_GIS
    assert result.status == ExecutionStatus.SUCCESS
    assert result.confidence == 0.87
    assert result.data["answer"] == (
        "Geospatial regions generated."
    )
    assert result.data["metrics"]["area_m2"] == 4200
    assert len(result.evidence) >= 1
