"""End-to-end M4 agent pipeline for SatQuery AI.

The pipeline connects query routing, planning, tool execution, evidence
collection, and evidence-derived confidence into one callable interface.
Specialist handlers remain pluggable through the shared tool registry.
"""

from __future__ import annotations

from typing import Any, Mapping

from .confidence import calculate_confidence
from .evidence import collect_evidence, summarize_evidence
from .executor import execute_plan
from .planner import build_plan
from .router import route_query
from .trace import ExecutionTrace


def run_agent(query: str, context: Mapping[str, Any] | None = None) -> dict:
    """Run the complete deterministic M4 orchestration pipeline."""

    decision = route_query(query)
    plan = build_plan(decision)

    execution = execute_plan(plan, context=dict(context or {}))
    evidence = collect_evidence(execution.to_dict()["steps"])
    evidence_summary = summarize_evidence(evidence)

    confidence = calculate_confidence(evidence_summary["items"])

    trace = ExecutionTrace(
        query=query,
        intent=decision.intent.value,
        plan=[tool.value for tool in plan],
        execution=execution,
    )

    return {
        "query": query,
        "route": decision.to_dict(),
        "plan": [tool.value for tool in plan],
        "execution": execution,
        "evidence": evidence_summary,
        "confidence": confidence,
        "trace": trace.to_dict(),
    }
