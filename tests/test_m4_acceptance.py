"""Final acceptance tests for the M4 agentic controller."""

from __future__ import annotations

import pytest

from app.agents.controller import AgentController
from app.query.registry import ToolRegistry
from app.query.schemas import (
    ExecutionStatus,
    QueryRequest,
    ToolName,
    ToolResult,
)


def result(
    tool: ToolName,
    answer: str,
    confidence: float = 0.90,
) -> ToolResult:
    """Create a deterministic specialist result."""

    return ToolResult(
        tool=tool,
        status=ExecutionStatus.SUCCESS,
        confidence=confidence,
        data={
            "answer": answer,
        },
    )


@pytest.fixture
def controller() -> AgentController:
    """Create a complete deterministic M4 runtime."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda **kwargs: result(
            ToolName.M1_VQA,
            "Satellite scene analyzed.",
        ),
    )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        lambda **kwargs: result(
            ToolName.M2_CHANGE_DETECTION,
            "Changed regions detected.",
        ),
    )

    registry.register(
        ToolName.M2_GROUNDING,
        lambda **kwargs: result(
            ToolName.M2_GROUNDING,
            "Changed regions localized.",
        ),
    )

    registry.register(
        ToolName.M3_OPTICAL_SAR,
        lambda **kwargs: result(
            ToolName.M3_OPTICAL_SAR,
            "Optical and SAR analysis completed.",
        ),
    )

    registry.register(
        ToolName.M5_GIS,
        lambda **kwargs: result(
            ToolName.M5_GIS,
            "Geospatial evidence generated.",
        ),
    )

    return AgentController(
        registry=registry,
    )


def test_vqa_acceptance(controller: AgentController) -> None:
    """M4 should route visual questions to M1."""

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image.",
        ),
        context={
            "images": ["image.tif"],
        },
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.intent.value == "VQA"
    assert response.results[0].tool == ToolName.M1_VQA
    assert response.answer == "Satellite scene analyzed."


def test_change_detection_acceptance(
    controller: AgentController,
) -> None:
    """M4 should route temporal change queries to M2."""

    response = controller.run(
        QueryRequest(
            query="What changed between these two images?",
        ),
        context={
            "before": "before.tif",
            "after": "after.tif",
        },
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.intent.value == "CHANGE_DETECTION"
    assert (
        response.results[0].tool
        == ToolName.M2_CHANGE_DETECTION
    )


def test_multi_step_building_acceptance(
    controller: AgentController,
) -> None:
    """
    Complex building-change queries should execute:

    M2 Change Detection → M2 Grounding → M5 GIS
    """

    response = controller.run(
        QueryRequest(
            query=(
                "Find newly constructed buildings "
                "between these two images."
            ),
        ),
        context={
            "before": "before.tif",
            "after": "after.tif",
        },
    )

    assert response.status == ExecutionStatus.SUCCESS

    assert [
        result.tool
        for result in response.results
    ] == [
        ToolName.M2_CHANGE_DETECTION,
        ToolName.M2_GROUNDING,
        ToolName.M5_GIS,
    ]

    assert len(response.trace) == 3
    assert len(response.results) == 3


def test_optical_sar_acceptance(
    controller: AgentController,
) -> None:
    """M4 should route multimodal queries to M3."""

    response = controller.run(
        QueryRequest(
            query=(
                "Use optical and SAR images "
                "to identify water regions."
            ),
        ),
        context={
            "optical": "optical.tif",
            "sar": "sar.tif",
        },
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.intent.value == "OPTICAL_SAR"
    assert response.results[0].tool == ToolName.M3_OPTICAL_SAR


def test_response_is_structured(
    controller: AgentController,
) -> None:
    """The final output must expose machine-readable fields."""

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image.",
        ),
        context={
            "images": ["image.tif"],
        },
    )

    assert response.status is not None
    assert response.intent is not None
    assert isinstance(response.answer, str)
    assert isinstance(response.confidence, float)
    assert isinstance(response.results, list)
    assert isinstance(response.trace, list)


def test_missing_input_is_controlled_failure(
    controller: AgentController,
) -> None:
    """
    M4 must fail safely when required runtime data is absent.
    """

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image.",
        ),
    )

    assert response.status == ExecutionStatus.FAILED
    assert response.confidence == 0.0
    assert response.error is not None
