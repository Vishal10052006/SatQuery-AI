"""
Tests for SatQuery AI M4 schemas.

These tests verify the contracts used between the M4
agent components and specialist modules.
"""

import pytest
from pydantic import ValidationError

from app.query.schemas import (
    AgentResponse,
    ExecutionPlan,
    ExecutionStatus,
    Intent,
    PlanStep,
    QueryRequest,
    ToolName,
    ToolResult,
)


def test_query_request():
    """Verify that a basic user query is valid."""

    request = QueryRequest(
        query="Find newly constructed buildings.",
        images=["before.tif", "after.tif"],
    )

    assert request.query == "Find newly constructed buildings."
    assert len(request.images) == 2


def test_plan_step():
    """Verify that an executable plan step is valid."""

    step = PlanStep(
        step_id=1,
        tool=ToolName.M2_CHANGE_DETECTION,
        operation="detect_new",
        inputs=["before.tif", "after.tif"],
    )

    assert step.step_id == 1
    assert step.tool == ToolName.M2_CHANGE_DETECTION


def test_execution_plan():
    """Verify that a complete execution plan is valid."""

    plan = ExecutionPlan(
        intent=Intent.CHANGE_DETECTION,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.M2_CHANGE_DETECTION,
                operation="detect_new",
                inputs=["before.tif", "after.tif"],
            )
        ],
    )

    assert plan.intent == Intent.CHANGE_DETECTION
    assert len(plan.steps) == 1


def test_tool_result_confidence():
    """Verify confidence validation."""

    result = ToolResult(
        tool=ToolName.M2_CHANGE_DETECTION,
        status=ExecutionStatus.SUCCESS,
        confidence=0.91,
        data={
            "changed_regions": 7,
        },
    )

    assert result.confidence == 0.91


def test_invalid_confidence():
    """Verify that confidence cannot exceed 1."""

    with pytest.raises(ValidationError):
        ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.SUCCESS,
            confidence=1.5,
        )


def test_agent_response():
    """Verify that M4 can construct a final response."""

    response = AgentResponse(
        status=ExecutionStatus.SUCCESS,
        answer="Seven regions show potential new construction.",
        confidence=0.88,
        intent=Intent.CHANGE_DETECTION,
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.confidence == 0.88
