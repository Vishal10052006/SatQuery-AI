"""
Tests for the SatQuery M4 execution planner.

The planner is responsible for converting structured queries
into ordered specialist-tool execution plans.
"""

from app.agents.planner import QueryPlanner
from app.query.parser import QueryParser
from app.query.schemas import Intent, ToolName


def test_vqa_plan():
    """VQA requires only M1."""

    parser = QueryParser()
    planner = QueryPlanner()

    parsed = parser.parse(
        "Describe this satellite image."
    )

    plan = planner.create_plan(parsed)

    assert plan.intent == Intent.VQA
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == ToolName.M1_VQA


def test_change_detection_plan():
    """Basic change detection requires M2."""

    parser = QueryParser()
    planner = QueryPlanner()

    parsed = parser.parse(
        "What changed between these two images?",
        images=["before.tif", "after.tif"],
    )

    plan = planner.create_plan(parsed)

    assert plan.intent == Intent.CHANGE_DETECTION
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == ToolName.M2_CHANGE_DETECTION


def test_new_buildings_plan():
    """
    New-building detection requires:

        M2 change detection
        M2 grounding
        M5 GIS
    """

    parser = QueryParser()
    planner = QueryPlanner()

    parsed = parser.parse(
        "Find newly constructed buildings between these two images.",
        images=["before.tif", "after.tif"],
    )

    plan = planner.create_plan(parsed)

    assert plan.intent == Intent.CHANGE_DETECTION

    assert len(plan.steps) == 3

    assert plan.steps[0].tool == ToolName.M2_CHANGE_DETECTION
    assert plan.steps[1].tool == ToolName.M2_GROUNDING
    assert plan.steps[2].tool == ToolName.M5_GIS

    assert plan.steps[0].step_id == 1
    assert plan.steps[1].step_id == 2
    assert plan.steps[2].step_id == 3


def test_optical_sar_plan():
    """Optical + SAR analysis requires M3."""

    parser = QueryParser()
    planner = QueryPlanner()

    parsed = parser.parse(
        "Use optical and SAR images to identify water regions."
    )

    plan = planner.create_plan(parsed)

    assert plan.intent == Intent.OPTICAL_SAR
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == ToolName.M3_OPTICAL_SAR


def test_grounding_plan():
    """Explicit grounding requires M2 grounding and M5 GIS."""

    parser = QueryParser()
    planner = QueryPlanner()

    parsed = parser.parse(
        "Where did the change occur?"
    )

    plan = planner.create_plan(parsed)

    # The current parser identifies this as CHANGE_DETECTION
    # with grounding/geospatial requirements.
    assert len(plan.steps) == 3

    assert plan.steps[0].tool == ToolName.M2_CHANGE_DETECTION
    assert plan.steps[1].tool == ToolName.M2_GROUNDING
    assert plan.steps[2].tool == ToolName.M5_GIS


def test_plan_step_order():
    """Every generated plan must have sequential step IDs."""

    parser = QueryParser()
    planner = QueryPlanner()

    parsed = parser.parse(
        "Find newly constructed buildings between these two images.",
        images=["before.tif", "after.tif"],
    )

    plan = planner.create_plan(parsed)

    step_ids = [
        step.step_id
        for step in plan.steps
    ]

    assert step_ids == list(
        range(1, len(plan.steps) + 1)
    )
