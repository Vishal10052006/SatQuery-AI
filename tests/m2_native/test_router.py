"""Tests for the M4 deterministic query router."""

from agent.router import route_query
from agent.schemas import Intent, ToolName


def test_change_query_routes_to_change_detection() -> None:
    decision = route_query("What changed between these two satellite images?")

    assert decision.intent is Intent.CHANGE_ANALYSIS
    assert ToolName.CHANGE_DETECTION in decision.tools
    assert decision.requires_temporal is True


def test_optical_sar_query_routes_to_multimodal_tool() -> None:
    decision = route_query("Compare the optical and SAR images.")

    assert decision.intent is Intent.MULTIMODAL_ANALYSIS
    assert decision.tools == [ToolName.OPTICAL_SAR]
    assert decision.requires_optical is True
    assert decision.requires_sar is True


def test_scene_query_routes_to_vqa() -> None:
    decision = route_query("What objects are visible in this satellite image?")

    assert decision.intent is Intent.SCENE_UNDERSTANDING
    assert decision.tools == [ToolName.VQA]


def test_unknown_query_does_not_guess() -> None:
    decision = route_query("Hello, how are you?")

    assert decision.intent is Intent.UNKNOWN
    assert decision.tools == []
    assert decision.confidence == 0.0
