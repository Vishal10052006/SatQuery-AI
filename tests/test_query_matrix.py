"""End-to-end query routing tests for M4."""

import pytest

from app.agents.controller import AgentController
from app.query.registry import ToolRegistry
from app.query.schemas import (
    ExecutionStatus,
    QueryRequest,
    ToolName,
    ToolResult,
)


def make_result(tool: ToolName, answer: str) -> ToolResult:
    """Create a deterministic specialist result for routing tests."""

    return ToolResult(
        tool=tool,
        status=ExecutionStatus.SUCCESS,
        confidence=0.90,
        data={"answer": answer},
    )


@pytest.fixture
def controller() -> AgentController:
    """Build an M4 controller with deterministic test specialists."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda **kwargs: make_result(
            ToolName.M1_VQA,
            "Satellite image described.",
        ),
    )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        lambda **kwargs: make_result(
            ToolName.M2_CHANGE_DETECTION,
            "Changes detected.",
        ),
    )

    registry.register(
        ToolName.M2_GROUNDING,
        lambda **kwargs: make_result(
            ToolName.M2_GROUNDING,
            "Objects localized.",
        ),
    )

    registry.register(
        ToolName.M3_OPTICAL_SAR,
        lambda **kwargs: make_result(
            ToolName.M3_OPTICAL_SAR,
            "Optical and SAR analysis completed.",
        ),
    )

    registry.register(
        ToolName.M5_GIS,
        lambda **kwargs: make_result(
            ToolName.M5_GIS,
            "Geospatial analysis completed.",
        ),
    )

    return AgentController(registry=registry)


@pytest.mark.parametrize(
    (
        "query",
        "expected_intent",
        "expected_tool",
        "context",
    ),
    [
        (
            "Describe this satellite image.",
            "VQA",
            ToolName.M1_VQA,
            {
                "images": ["image.tif"],
            },
        ),
        (
            "What changed between these two images?",
            "CHANGE_DETECTION",
            ToolName.M2_CHANGE_DETECTION,
            {
                "before": "before.tif",
                "after": "after.tif",
            },
        ),
        (
            "Where did the change occur?",
            "CHANGE_DETECTION",
            ToolName.M2_CHANGE_DETECTION,
            {
                "before": "before.tif",
                "after": "after.tif",
            },
        ),
        (
            "Use optical and SAR images to identify water regions.",
            "OPTICAL_SAR",
            ToolName.M3_OPTICAL_SAR,
            {
                "optical": "optical.tif",
                "sar": "sar.tif",
            },
        ),
    ],
)
def test_query_routes_to_expected_specialist(
    controller: AgentController,
    query: str,
    expected_intent: str,
    expected_tool: ToolName,
    context: dict,
) -> None:
    """Natural-language queries should reach the correct specialist."""

    response = controller.run(
        QueryRequest(query=query),
        context=context,
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.intent.value == expected_intent

    executed_tools = [
        result.tool
        for result in response.results
    ]

    assert expected_tool in executed_tools


def test_new_building_query_creates_multi_step_execution(
    controller: AgentController,
) -> None:
    """A complex change query should execute the planned chain."""

    response = controller.run(
        QueryRequest(
            query=(
                "Find newly constructed buildings "
                "between these two images."
            )
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


def test_optical_sar_query_executes_m3(
    controller: AgentController,
) -> None:
    """Optical/SAR queries should execute the M3 specialist."""

    response = controller.run(
        QueryRequest(
            query=(
                "Use optical and SAR images "
                "to identify water regions."
            )
        ),
        context={
            "optical": "optical.tif",
            "sar": "sar.tif",
        },
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.results[0].tool == ToolName.M3_OPTICAL_SAR
