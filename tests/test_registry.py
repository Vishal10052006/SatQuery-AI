"""
Tests for the SatQuery M4 Tool Registry.

The tests verify:
    - registration
    - retrieval
    - availability checks
    - successful execution
    - failed execution
    - output contract validation
"""

from app.query.registry import ToolRegistry
from app.query.schemas import (
    ExecutionStatus,
    ToolName,
    ToolResult,
)


def fake_vqa(**kwargs):
    """
    Mock M1 implementation.

    This represents the interface that the real M1 adapter
    will eventually provide.
    """

    return ToolResult(
        tool=ToolName.M1_VQA,
        status=ExecutionStatus.SUCCESS,
        confidence=0.90,
        data={
            "answer": "Agricultural land is visible."
        },
    )


def fake_change_detection(**kwargs):
    """
    Mock M2 implementation.

    This represents the interface that the real M2 adapter
    will eventually provide.
    """

    return ToolResult(
        tool=ToolName.M2_CHANGE_DETECTION,
        status=ExecutionStatus.SUCCESS,
        confidence=0.91,
        data={
            "changed_regions": 7,
            "change_mask": "change_mask.png",
        },
    )


def fake_optical_sar(**kwargs):
    """
    Mock M3 implementation.

    This represents the interface that the real M3 adapter
    will eventually provide.
    """

    return ToolResult(
        tool=ToolName.M3_OPTICAL_SAR,
        status=ExecutionStatus.SUCCESS,
        confidence=0.88,
        data={
            "regions": ["water", "built_up"],
        },
    )


def test_register_tool():
    """A tool can be registered and retrieved."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        fake_vqa,
    )

    assert registry.is_registered(
        ToolName.M1_VQA
    )

    assert registry.get(
        ToolName.M1_VQA
    ) is fake_vqa


def test_list_tools():
    """Registered tools should appear in the tool list."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        fake_vqa,
    )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        fake_change_detection,
    )

    tools = registry.list_tools()

    assert ToolName.M1_VQA in tools
    assert ToolName.M2_CHANGE_DETECTION in tools


def test_execute_tool():
    """A registered tool should execute successfully."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        fake_vqa,
    )

    result = registry.execute(
        ToolName.M1_VQA,
        query="Describe the image.",
    )

    assert result.tool == ToolName.M1_VQA
    assert result.status == ExecutionStatus.SUCCESS
    assert result.confidence == 0.90


def test_execute_change_detection():
    """M2 change detection can be executed through the registry."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        fake_change_detection,
    )

    result = registry.execute(
        ToolName.M2_CHANGE_DETECTION,
        before="before.tif",
        after="after.tif",
    )

    assert result.tool == ToolName.M2_CHANGE_DETECTION
    assert result.data["changed_regions"] == 7


def test_execute_optical_sar():
    """M3 optical + SAR can be executed through the registry."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M3_OPTICAL_SAR,
        fake_optical_sar,
    )

    result = registry.execute(
        ToolName.M3_OPTICAL_SAR,
        optical="optical.tif",
        sar="sar.tif",
    )

    assert result.tool == ToolName.M3_OPTICAL_SAR
    assert result.status == ExecutionStatus.SUCCESS


def test_missing_tool_returns_failure():
    """Executing an unregistered tool should return a failure result."""

    registry = ToolRegistry()

    result = registry.execute(
        ToolName.M1_VQA,
        query="Describe image.",
    )

    assert result.tool == ToolName.M1_VQA
    assert result.status == ExecutionStatus.FAILED
    assert result.confidence == 0.0
    assert result.error is not None


def test_invalid_tool_output_returns_failure():
    """A specialist returning the wrong type must fail safely."""

    registry = ToolRegistry()

    def invalid_tool(**kwargs):
        """Incorrect specialist implementation for testing."""

        return {
            "answer": "This is invalid."
        }

    registry.register(
        ToolName.M1_VQA,
        invalid_tool,
    )

    result = registry.execute(
        ToolName.M1_VQA,
        query="Describe image.",
    )

    assert result.status == ExecutionStatus.FAILED
    assert result.error is not None
