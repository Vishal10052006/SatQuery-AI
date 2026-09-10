"""M4 mission orchestrator connecting query understanding to specialists."""
from __future__ import annotations

from typing import Any

from agent.planner import build_plan
from agent.router import route_query
from agent.schemas import ToolName
from core.contracts import SpecialistResult
from models.change.adapter import run_change_detection, run_grounding
from models.multimodal.adapter import run_optical_sar
from models.vlm.adapter import run_vqa
from query.parser import parse_query


def _mission_plan(query: str, spec, decision) -> list[ToolName]:
    """Build a composed plan so one natural-language mission can use many tools."""
    plan = list(build_plan(decision))

    # Parser semantics can extend a coarse first-pass router for combined tasks.
    if spec.intent == "change_analysis" and ToolName.CHANGE_DETECTION not in plan:
        plan.insert(0, ToolName.CHANGE_DETECTION)
    if spec.needs_localization and ToolName.GROUNDING not in plan:
        plan.append(ToolName.GROUNDING)
    if spec.optical and spec.sar and ToolName.OPTICAL_SAR not in plan:
        plan.append(ToolName.OPTICAL_SAR)

    order = {
        ToolName.CHANGE_DETECTION: 0,
        ToolName.GROUNDING: 1,
        ToolName.OPTICAL_SAR: 2,
        ToolName.VQA: 3,
    }
    return sorted(dict.fromkeys(plan), key=lambda tool: order[tool])


def run_mission(query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Understand, plan and execute a complete M1-M4 mission request."""
    context = dict(context or {})
    spec = parse_query(query)
    decision = route_query(query)
    plan = _mission_plan(query, spec, decision)
    results: list[SpecialistResult] = []

    for tool in plan:
        if tool == ToolName.VQA:
            results.append(run_vqa(context.get("image"), query))
        elif tool == ToolName.CHANGE_DETECTION:
            results.append(
                run_change_detection(
                    context.get("before"),
                    context.get("after"),
                    spec.target,
                    context.get("output_dir"),
                )
            )
        elif tool == ToolName.GROUNDING:
            results.append(run_grounding(context.get("image"), spec.target or "requested target"))
        elif tool == ToolName.OPTICAL_SAR:
            results.append(run_optical_sar(context.get("optical"), context.get("sar")))

    confidences = [r.confidence for r in results]
    successful = sum(r.status in {"ready", "success"} for r in results)
    return {
        "query": query,
        "query_spec": spec.to_dict(),
        "route": decision.to_dict(),
        "plan": [t.value for t in plan],
        "results": [r.to_dict() for r in results],
        "confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0,
        "evidence_summary": {
            "total_tools": len(results),
            "successful_tools": successful,
            "tool_coverage": round(successful / len(results), 2) if results else 0.0,
        },
        "status": "complete" if results and successful == len(results) else ("awaiting_models" if results else "no_action"),
    }
