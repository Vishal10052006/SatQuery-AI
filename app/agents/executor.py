"""
M4 Agent Executor

Executes an ExecutionPlan step-by-step through the ToolRegistry.

The executor is deliberately independent of the actual M1/M2/M3/M5
implementations. Specialist modules only need to satisfy the ToolResult
contract registered in ToolRegistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.query.registry import ToolRegistry
from app.query.schemas import ExecutionPlan, ExecutionStatus, ToolResult


@dataclass
class ExecutionReport:
    """Runtime report produced after executing a plan."""

    results: list[ToolResult] = field(default_factory=list)

    @property
    def status(self) -> ExecutionStatus:
        """Return overall execution status."""
        if not self.results:
            return ExecutionStatus.FAILED

        failed = any(
            result.status == ExecutionStatus.FAILED
            for result in self.results
        )

        partial = any(
            result.status == ExecutionStatus.PARTIAL
            for result in self.results
        )

        successful = any(
            result.status == ExecutionStatus.SUCCESS
            for result in self.results
        )

        # A failure after useful work has already completed is
        # a partial execution, not a total failure.
        if failed and successful:
            return ExecutionStatus.PARTIAL

        if failed:
            return ExecutionStatus.FAILED

        if partial:
            return ExecutionStatus.PARTIAL

        return ExecutionStatus.SUCCESS


class AgentExecutor:
    """
    Execute planner-generated steps through the specialist tool registry.

    Runtime context contains actual inputs such as:
        before
        after
        images
        optical
        sar

    A previous ToolResult is automatically exposed as:
        previous_result
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        plan: ExecutionPlan,
        context: dict[str, Any] | None = None,
    ) -> ExecutionReport:
        """
        Execute all steps in order.

        Execution stops immediately if a tool fails.
        """
        runtime_context = dict(context or {})

        # output_dir is runtime infrastructure rather than a user-facing
        # input. Production callers may provide an explicit directory;
        # direct callers/tests receive the historical local default.
        runtime_context.setdefault("output_dir", "output")

        results: list[ToolResult] = []

        for step in plan.steps:
            try:
                kwargs = self._build_kwargs(
                    step=step,
                    context=runtime_context,
                    previous_result=results[-1] if results else None,
                )

                result = self.registry.execute(
                    step.tool.value,
                    **kwargs,
                )

            except Exception as exc:
                result = ToolResult(
                    tool=step.tool,
                    status=ExecutionStatus.FAILED,
                    confidence=0.0,
                    error=str(exc),
                )

            results.append(result)

            # Make the latest result available to downstream steps.
            runtime_context["previous_result"] = result

            # Fail-fast execution for the MVP.
            if result.status == ExecutionStatus.FAILED:
                break

        return ExecutionReport(results=results)

    @staticmethod
    def _build_kwargs(
        step: Any,
        context: dict[str, Any],
        previous_result: ToolResult | None,
    ) -> dict[str, Any]:
        """Resolve planner input references into runtime tool arguments."""

        kwargs: dict[str, Any] = {}

        for input_name in step.inputs:
            if input_name == "previous_result":
                kwargs[input_name] = previous_result
                continue

            if input_name not in context:
                raise ValueError(
                    f"Missing runtime input '{input_name}' "
                    f"required by tool '{step.tool.value}'"
                )

            kwargs[input_name] = context[input_name]

        # Planner parameters are passed directly to the specialist tool.
        for key, value in step.parameters.items():
            if key not in kwargs:
                kwargs[key] = value

        return kwargs
