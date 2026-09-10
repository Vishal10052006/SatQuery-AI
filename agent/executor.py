"""Safe execution layer for the SatQuery M4 agent.

The executor takes an ordered plan from the task planner and invokes only
registered specialist handlers. Missing handlers are reported explicitly
instead of being silently guessed or replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schemas import ToolName
from .tool_registry import get_tool


@dataclass
class ExecutionStep:
    """Result of one planned tool invocation."""

    tool: ToolName
    status: str
    result: Any = None
    error: str | None = None


@dataclass
class ExecutionResult:
    """Aggregate execution result for an agent run."""

    steps: List[ExecutionStep] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when all planned steps completed successfully."""

        return bool(self.steps) and all(step.status == "success" for step in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable execution record."""

        return {
            "success": self.success,
            "steps": [
                {
                    "tool": step.tool.value,
                    "status": step.status,
                    "result": step.result,
                    "error": step.error,
                }
                for step in self.steps
            ],
        }


def execute_plan(plan: List[ToolName], context: Dict[str, Any] | None = None) -> ExecutionResult:
    """Execute a planned sequence using registered handlers.

    Handlers are expected to accept a single context dictionary. A tool without
    a registered handler receives a clear ``unavailable`` status, allowing the
    higher-level agent to fall back or explain what is missing.
    """

    context = dict(context or {})
    execution = ExecutionResult()

    for tool_name in plan:
        spec = get_tool(tool_name)
        if spec.handler is None:
            execution.steps.append(
                ExecutionStep(
                    tool=tool_name,
                    status="unavailable",
                    error=f"No handler registered for tool '{tool_name.value}'",
                )
            )
            continue

        try:
            result = spec.handler(context)
            execution.steps.append(
                ExecutionStep(tool=tool_name, status="success", result=result)
            )
        except Exception as exc:  # pragma: no cover - defensive integration boundary
            execution.steps.append(
                ExecutionStep(tool=tool_name, status="error", error=str(exc))
            )

    return execution
