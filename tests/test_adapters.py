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
