"""Auditable execution trace utilities for the SatQuery M4 agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from .schemas import Intent, RouteDecision, ToolName
from .executor import ExecutionResult


@dataclass
class AgentTrace:
    """Structured record of how the agent reached an answer."""

    query: str
    intent: Intent
    plan: List[ToolName] = field(default_factory=list)
    execution: ExecutionResult | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_run(
        cls,
        query: str,
        decision: RouteDecision,
        plan: List[ToolName],
        execution: ExecutionResult | None = None,
    ) -> "AgentTrace":
        """Build a trace from one routing/planning/execution cycle."""

        return cls(
            query=query,
            intent=decision.intent,
            plan=list(plan),
            execution=execution,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a compact API/UI-friendly execution trace."""

        intent_val = self.intent.value if hasattr(self.intent, "value") else str(self.intent)
        plan_vals = [tool.value if hasattr(tool, "value") else str(tool) for tool in self.plan]
        return {
            "query": self.query,
            "intent": intent_val,
            "plan": plan_vals,
            "execution": self.execution.to_dict() if self.execution else None,
            "created_at": self.created_at,
        }


# Alias for backward compatibility with agent/pipeline.py
ExecutionTrace = AgentTrace
