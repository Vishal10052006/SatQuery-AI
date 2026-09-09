"""Tests for the M4 AgentController."""

from app.agents.controller import AgentController
from app.query.registry import ToolRegistry
from app.query.schemas import (
    ExecutionStatus,
    QueryRequest,
    ToolName,
    ToolResult,
)


def test_controller_runs_complete_pipeline() -> None:
    """Controller should connect parser → planner → executor → synthesizer."""

    registry = ToolRegistry()

    def vqa(images, **kwargs):
        return ToolResult(
            tool=ToolName.M1_VQA,
            status=ExecutionStatus.SUCCESS,
            confidence=0.95,
            data={
                "answer": "A road is visible."
            },
        )

    registry.register(
        ToolName.M1_VQA,
        vqa,
    )

    controller = AgentController(
        registry=registry,
    )

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image.",
            images=["image.tif"],
        )
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.answer == "A road is visible."
    assert response.confidence == 0.95
    assert response.intent is not None
    assert len(response.results) == 1
    assert response.results[0].tool == ToolName.M1_VQA


def test_controller_passes_runtime_context() -> None:
    """Additional runtime inputs should reach specialist tools."""

    registry = ToolRegistry()

    received = {}

    def change_detection(before, after, **kwargs):
        received["before"] = before
        received["after"] = after

        return ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.SUCCESS,
            confidence=0.91,
            data={
                "answer": "Change detected."
            },
        )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        change_detection,
    )

    controller = AgentController(
        registry=registry,
    )

    response = controller.run(
        QueryRequest(
            query="What changed between these two images?"
        ),
        context={
            "before": "before.tif",
            "after": "after.tif",
        },
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert received["before"] == "before.tif"
    assert received["after"] == "after.tif"


def test_controller_handles_missing_input() -> None:
    """Controller should return a controlled failure."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda images, **kwargs: ToolResult(
            tool=ToolName.M1_VQA,
            status=ExecutionStatus.SUCCESS,
            confidence=0.95,
            data={"answer": "test"},
        ),
    )

    controller = AgentController(
        registry=registry,
    )

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image."
        )
    )

    assert response.status == ExecutionStatus.FAILED
    assert response.confidence == 0.0
    assert response.error is not None


def test_controller_returns_response_when_parser_fails() -> None:
    """Controller-level exceptions should become structured failures."""

    from app.query.registry import ToolRegistry

    class BrokenParser:
        def parse(self, *args, **kwargs):
            raise RuntimeError("Parser unavailable")

    controller = AgentController(
        registry=ToolRegistry(),
        parser=BrokenParser(),
    )

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image."
        )
    )

    assert response.status == ExecutionStatus.FAILED
    assert response.intent is None
    assert response.confidence == 0.0
    assert response.error == "Parser unavailable"
