"""
SatQuery AI - M4 Response Synthesizer

Converts specialist ToolResults into the final AgentResponse
consumed by the application/frontend.

Architecture:

    ExecutionReport
          ↓
      Synthesizer
          ↓
     AgentResponse

The synthesizer does not perform satellite analysis itself.
It only orchestrates and presents results produced by
specialist tools.
"""

from __future__ import annotations

import json
from typing import Any

from app.agents.executor import ExecutionReport
from app.query.schemas import (
    AgentResponse,
    Evidence,
    ExecutionStatus,
    Intent,
    ToolResult,
)


class ResponseSynthesizer:
    """
    Deterministic response synthesizer for the M4 MVP.

    Responsibilities:
        - combine specialist results
        - aggregate evidence
        - calculate conservative confidence
        - generate a human-readable answer
        - expose execution trace
        - preserve failures
    """

    def synthesize(
        self,
        intent: Intent,
        report: ExecutionReport,
    ) -> AgentResponse:
        """
        Convert an execution report into the final agent response.

        Confidence is conservative:
        the minimum confidence among executed results is used,
        because a multi-step answer should not claim more confidence
        than its weakest successful dependency.
        """

        results = report.results

        evidence = self._collect_evidence(results)
        trace = self._build_trace(results)
        confidence = self._aggregate_confidence(results)

        answer = self._build_answer(results)

        error = self._extract_error(results)

        return AgentResponse(
            status=report.status,
            answer=answer,
            confidence=confidence,
            intent=intent,
            evidence=evidence,
            results=results,
            trace=trace,
            error=error,
        )

    # ------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------

    @staticmethod
    def _build_answer(results: list[ToolResult]) -> str:
        """
        Build a readable answer from specialist output.

        Specialist tools may provide natural-language fields such as:
            answer
            summary
            description
            message
            finding

        If none are present, structured data is serialized instead.
        """

        if not results:
            return "No analysis result was produced."

        messages: list[str] = []

        preferred_fields = (
            "answer",
            "summary",
            "description",
            "finding",
            "message",
            "result",
        )

        for result in results:
            if result.status == ExecutionStatus.FAILED:
                continue

            text = ResponseSynthesizer._extract_text(
                result.data,
                preferred_fields,
            )

            if text:
                messages.append(text)

        if messages:
            return " ".join(messages)

        structured_outputs: list[str] = []

        for result in results:
            if result.status == ExecutionStatus.FAILED:
                continue

            if result.data:
                structured_outputs.append(
                    json.dumps(
                        result.data,
                        default=str,
                        sort_keys=True,
                    )
                )

        if structured_outputs:
            return "Analysis results: " + " | ".join(
                structured_outputs
            )

        if any(
            result.status == ExecutionStatus.PARTIAL
            for result in results
        ):
            return "Analysis completed partially."

        if any(
            result.status == ExecutionStatus.FAILED
            for result in results
        ):
            return "Analysis could not be completed because a required specialist failed."

        return "Analysis completed successfully."

    @staticmethod
    def _extract_text(
        data: dict[str, Any],
        preferred_fields: tuple[str, ...],
    ) -> str | None:
        """Extract the first useful natural-language result field."""

        for field in preferred_fields:
            value = data.get(field)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    # ------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------

    @staticmethod
    def _aggregate_confidence(
        results: list[ToolResult],
    ) -> float:
        """
        Aggregate specialist confidence conservatively.

        Failed results do not contribute because the overall status
        already records the failure.
        """

        successful = [
            result.confidence
            for result in results
            if result.status != ExecutionStatus.FAILED
        ]

        if not successful:
            return 0.0

        return min(successful)

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    @staticmethod
    def _collect_evidence(
        results: list[ToolResult],
    ) -> list[Evidence]:
        """Flatten evidence from all specialist results."""

        evidence: list[Evidence] = []

        for result in results:
            evidence.extend(result.evidence)

        return evidence

    # ------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------

    @staticmethod
    def _build_trace(
        results: list[ToolResult],
    ) -> list[str]:
        """Create a simple execution trace for M6/frontend."""

        trace: list[str] = []

        for index, result in enumerate(results, start=1):
            trace.append(
                f"STEP {index}: "
                f"{result.tool.value} → "
                f"{result.status.value}"
            )

        return trace

    # ------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------

    @staticmethod
    def _extract_error(
        results: list[ToolResult],
    ) -> str | None:
        """Return the first specialist error, if one exists."""

        for result in results:
            if result.status == ExecutionStatus.FAILED:
                return result.error

        return None


# ------------------------------------------------------------
# Default synthesizer instance
# ------------------------------------------------------------

synthesizer = ResponseSynthesizer()
