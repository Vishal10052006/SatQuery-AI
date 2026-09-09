"""M4 specialist integration contract tests."""

from __future__ import annotations

from types import SimpleNamespace

from app.adapters.register import register_specialists
from app.query.registry import ToolRegistry
from app.query.schemas import (
    ExecutionStatus,
    ToolName,
)


def external_result(answer: str) -> SimpleNamespace:
    """Create a generic external specialist response."""

    return SimpleNamespace(
        answer=answer,
        confidence=0.90,
        metrics={},
        warnings=[],
        mask_geojson=None,
        bboxes=None,
    )


class FakeSpecialist:
    """Minimal external specialist used for contract testing."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def __call__(self, **kwargs):
        return external_result(self.answer)


def test_all_available_specialists_can_register() -> None:
    """M4 should expose a stable ToolRegistry contract."""

    registry = ToolRegistry()

    register_specialists(
        registry,
        vqa=FakeSpecialist("VQA complete."),
        change_detection=FakeSpecialist(
            "Change detection complete."
        ),
        grounding=FakeSpecialist(
            "Grounding complete."
        ),
        optical_sar=FakeSpecialist(
            "Optical SAR complete."
        ),
        gis=FakeSpecialist(
            "GIS complete."
        ),
    )

    expected_tools = {
        ToolName.M1_VQA,
        ToolName.M2_CHANGE_DETECTION,
        ToolName.M2_GROUNDING,
        ToolName.M3_OPTICAL_SAR,
        ToolName.M5_GIS,
    }

    assert set(registry.list_tools()) == expected_tools


def test_registered_specialists_return_tool_results() -> None:
    """Every registered specialist must satisfy the M4 contract."""

    registry = ToolRegistry()

    register_specialists(
        registry,
        vqa=FakeSpecialist("VQA complete."),
        change_detection=FakeSpecialist(
            "Change detection complete."
        ),
        grounding=FakeSpecialist(
            "Grounding complete."
        ),
        optical_sar=FakeSpecialist(
            "Optical SAR complete."
        ),
        gis=FakeSpecialist(
            "GIS complete."
        ),
    )

    # Each specialist has a different runtime input contract.
    # Provide representative inputs so the test validates the
    # adapter boundary rather than intentionally triggering
    # missing-input failures.
    inputs = {
        ToolName.M1_VQA: {
            "images": ["image.tif"],
        },
        ToolName.M2_CHANGE_DETECTION: {
            "before": "before.tif",
            "after": "after.tif",
        },
        ToolName.M2_GROUNDING: {
            "images": ["image.tif"],
        },
        ToolName.M3_OPTICAL_SAR: {
            "optical": "optical.tif",
            "sar": "sar.tif",
        },
        ToolName.M5_GIS: {},
    }

    for tool in registry.list_tools():
        result = registry.execute(
            tool,
            **inputs[tool],
        )

        assert result.tool == tool
        assert result.status == ExecutionStatus.SUCCESS
        assert isinstance(result.confidence, float)
        assert isinstance(result.data, dict)
