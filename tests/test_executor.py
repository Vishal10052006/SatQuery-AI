"""Tests for the M4 agent executor."""

from app.agents.executor import AgentExecutor
from app.agents.planner import QueryPlanner
from app.query.parser import QueryParser
from app.query.registry import ToolRegistry
from app.query.schemas import ExecutionStatus, ToolName, ToolResult


def make_result(tool: ToolName, data: dict) -> ToolResult:
    """Create a successful fake specialist-tool result."""
    return ToolResult(
        tool=tool,
        status=ExecutionStatus.SUCCESS,
        confidence=0.95,
        data=data,
    )


def test_executor_runs_single_step() -> None:
    """A simple VQA plan should execute through the registry."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda images, **kwargs: make_result(
            ToolName.M1_VQA,
            {"answer": "A road is visible."},
        ),
    )

    parsed = QueryParser().parse("Describe this satellite image.")
    plan = QueryPlanner().create_plan(parsed)

    executor = AgentExecutor(registry)

    report = executor.execute(
        plan,
        context={"images": ["image.jpg"]},
    )

    assert report.status == ExecutionStatus.SUCCESS
    assert len(report.results) == 1
    assert report.results[0].data["answer"] == "A road is visible."


def test_executor_runs_multi_step_pipeline() -> None:
    """Results from earlier steps must be available to later steps."""

    registry = ToolRegistry()
    calls: list[str] = []

    def change_detection(before, after, **kwargs):
        calls.append("change_detection")

        return make_result(
            ToolName.M2_CHANGE_DETECTION,
            {"changed_regions": ["region_1"]},
        )

    def grounding(previous_result, **kwargs):
        calls.append("grounding")

        assert previous_result is not None
        assert previous_result.data["changed_regions"] == ["region_1"]

        return make_result(
            ToolName.M2_GROUNDING,
            {"locations": ["region_1"]},
        )

    def gis(previous_result, **kwargs):
        calls.append("gis")

        assert previous_result is not None
        assert previous_result.data["locations"] == ["region_1"]

        return make_result(
            ToolName.M5_GIS,
            {"map": "generated"},
        )

    registry.register(ToolName.M2_CHANGE_DETECTION, change_detection)
    registry.register(ToolName.M2_GROUNDING, grounding)
    registry.register(ToolName.M5_GIS, gis)

    parsed = QueryParser().parse(
        "Find newly constructed buildings between these two images."
    )
    plan = QueryPlanner().create_plan(parsed)

    executor = AgentExecutor(registry)

    report = executor.execute(
        plan,
        context={
            "before": "before.jpg",
            "after": "after.jpg",
        },
    )

    assert report.status == ExecutionStatus.SUCCESS
    assert len(report.results) == 3
    assert calls == [
        "change_detection",
        "grounding",
        "gis",
    ]


def test_executor_stops_after_failure() -> None:
    """A failed tool should stop downstream execution."""

    registry = ToolRegistry()
    calls: list[str] = []

    def failing_tool(before, after, **kwargs):
        calls.append("change_detection")

        return ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.FAILED,
            confidence=0.0,
            error="Model unavailable",
        )

    def grounding(**kwargs):
        calls.append("grounding")

        return make_result(
            ToolName.M2_GROUNDING,
            {"locations": []},
        )

    registry.register(ToolName.M2_CHANGE_DETECTION, failing_tool)
    registry.register(ToolName.M2_GROUNDING, grounding)

    parsed = QueryParser().parse(
        "Find newly constructed buildings between these two images."
    )
    plan = QueryPlanner().create_plan(parsed)

    report = AgentExecutor(registry).execute(
        plan,
        context={
            "before": "before.jpg",
            "after": "after.jpg",
        },
    )

    assert report.status == ExecutionStatus.FAILED
    assert len(report.results) == 1
    assert calls == ["change_detection"]


def test_executor_handles_missing_runtime_input() -> None:
    """Missing required inputs should become a controlled failure."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda images, **kwargs: make_result(
            ToolName.M1_VQA,
            {"answer": "test"},
        ),
    )

    parsed = QueryParser().parse("Describe this satellite image.")
    plan = QueryPlanner().create_plan(parsed)

    report = AgentExecutor(registry).execute(plan, context={})

    assert report.status == ExecutionStatus.FAILED
    assert len(report.results) == 1
    assert "Missing runtime input 'images'" in report.results[0].error


def test_executor_reports_partial_after_success_then_failure() -> None:
    """Useful results followed by failure should produce PARTIAL."""

    from app.query.registry import ToolRegistry
    from app.query.schemas import (
        ExecutionPlan,
        ExecutionStatus,
        PlanStep,
        ToolName,
        ToolResult,
    )
    from app.agents.executor import AgentExecutor

    registry = ToolRegistry()

    def success_tool(**kwargs):
        return ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.SUCCESS,
            confidence=0.90,
            data={"answer": "Change detected."},
        )

    def failing_tool(**kwargs):
        raise RuntimeError("Grounding model unavailable")

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        success_tool,
    )

    registry.register(
        ToolName.M2_GROUNDING,
        failing_tool,
    )

    plan = ExecutionPlan(
        intent="CHANGE_DETECTION",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.M2_CHANGE_DETECTION,
                operation="detect_change",
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.M2_GROUNDING,
                operation="localize",
                inputs=["previous_result"],
            ),
        ],
    )

    report = AgentExecutor(registry).execute(plan)

    assert len(report.results) == 2
    assert report.results[0].status == ExecutionStatus.SUCCESS
    assert report.results[1].status == ExecutionStatus.FAILED
    assert report.status == ExecutionStatus.PARTIAL


def test_executor_reports_failed_when_first_tool_fails() -> None:
    """A failure before any useful result is a complete failure."""

    from app.query.registry import ToolRegistry
    from app.query.schemas import (
        ExecutionPlan,
        ExecutionStatus,
        PlanStep,
        ToolName,
        ToolResult,
    )
    from app.agents.executor import AgentExecutor

    registry = ToolRegistry()

    def failing_tool(**kwargs):
        return ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.FAILED,
            confidence=0.0,
            error="Input image unavailable",
        )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        failing_tool,
    )

    plan = ExecutionPlan(
        intent="CHANGE_DETECTION",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.M2_CHANGE_DETECTION,
                operation="detect_change",
            ),
        ],
    )

    report = AgentExecutor(registry).execute(plan)

    assert len(report.results) == 1
    assert report.status == ExecutionStatus.FAILED
