"""Deterministic task planner for SatQuery AI.

The planner converts the router's high-level intent into an ordered analysis
plan. It is intentionally deterministic for the MVP so the execution path is
predictable and easy to audit during the SIH demo.
"""

from __future__ import annotations

from typing import List

from .schemas import Intent, RouteDecision, ToolName


def build_plan(decision: RouteDecision) -> List[ToolName]:
    """Build an ordered tool plan from a routing decision.

    The plan represents analysis steps, not model implementations. Specialist
    teams can register their concrete handlers against these tool names later.
    """

    if decision.intent == Intent.CHANGE_ANALYSIS:
        plan: List[ToolName] = [ToolName.CHANGE_DETECTION]
        if decision.requires_grounding:
            plan.append(ToolName.GROUNDING)
        if decision.requires_optical or decision.requires_sar:
            plan.append(ToolName.OPTICAL_SAR)
        return plan

    if decision.intent == Intent.GROUNDING:
        return [ToolName.GROUNDING]

    if decision.intent == Intent.MULTIMODAL_ANALYSIS:
        return [ToolName.OPTICAL_SAR]

    if decision.intent == Intent.SCENE_UNDERSTANDING:
        return [ToolName.VQA]

    return []
