"""Tests for the public M4 API."""

from app.api import SatQueryAPI
from app.query.registry import ToolRegistry
from app.agents.controller import AgentController
from app.query.schemas import (
    ExecutionStatus,
    ToolName,
    ToolResult,
)


def make_controller() -> AgentController:
    """Create a deterministic controller for API testing."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda **kwargs: ToolResult(
            tool=ToolName.M1_VQA,
            status=ExecutionStatus.SUCCESS,
            confidence=0.91,
            data={
                "answer": "Satellite image described."
            },
        ),
    )

    return AgentController(
        registry=registry,
    )


def test_api_ask_returns_agent_response() -> None:
    """Public ask() should return the structured M4 response."""

    api = SatQueryAPI(
        controller=make_controller(),
    )

    response = api.ask(
        "Describe this satellite image.",
        context={
            "images": ["image.tif"],
        },
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.answer == "Satellite image described."
    assert response.confidence == 0.91
    assert response.intent.value == "VQA"


def test_api_forwards_context() -> None:
    """Runtime image context must reach the specialist."""

    received = {}

    registry = ToolRegistry()

    def vqa(**kwargs):
        received.update(kwargs)

        return ToolResult(
            tool=ToolName.M1_VQA,
            status=ExecutionStatus.SUCCESS,
            confidence=0.90,
            data={
                "answer": "Image analyzed."
            },
        )

    registry.register(
        ToolName.M1_VQA,
        vqa,
    )

    api = SatQueryAPI(
        controller=AgentController(
            registry=registry,
        ),
    )

    api.ask(
        "Describe this satellite image.",
        context={
            "images": ["satellite.tif"],
        },
    )

    assert received["images"] == ["satellite.tif"]


def test_api_accepts_injected_controller() -> None:
    """Existing controllers should be directly injectable."""

    controller = make_controller()

    api = SatQueryAPI(
        controller=controller,
    )

    assert api.controller is controller
