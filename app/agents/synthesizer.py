"""M4 deterministic response synthesis for specialist ToolResults."""
from __future__ import annotations

from app.agents.executor import ExecutionReport
from app.query.schemas import (
    AgentResponse,
    Evidence,
    ExecutionStatus,
    Intent,
    ParsedQuery,
    ToolResult,
)


class ResponseSynthesizer:
    """Turn specialist evidence into a query-specific, auditable answer."""

    def synthesize(
        self,
        intent: Intent,
        report: ExecutionReport,
        parsed_query: ParsedQuery | None = None,
    ) -> AgentResponse:
        results = report.results
        return AgentResponse(
            status=report.status,
            answer=self._build_answer(intent, results, parsed_query),
            confidence=self._aggregate_confidence(results),
            intent=intent,
            evidence=self._collect_evidence(results),
            results=results,
            trace=self._build_trace(results),
            error=self._extract_error(results),
        )

    @staticmethod
    def _build_answer(
        intent: Intent,
        results: list[ToolResult],
        parsed_query: ParsedQuery | None = None,
    ) -> str:
        """Generate an answer focused on what the user actually asked."""
        if not results:
            return "No analysis result was produced."

        if intent == Intent.CHANGE_DETECTION:
            m2 = next((r for r in results if r.tool.value == "m2_change_detection"), None)
            m5 = next((r for r in results if r.tool.value == "m5_gis"), None)
            if m2:
                data = m2.data or {}
                target = parsed_query.target if parsed_query and parsed_query.target else data.get("target")
                operation = parsed_query.operation.value if parsed_query else "detect_change"
                region_values = data.get("regions", [])
                regions = len(region_values) if isinstance(region_values, list) else int(data.get("number_of_regions", 0) or 0)
                detected = bool(data.get("change_detected", False)) or regions > 0
                area = data.get("changed_area_sq_m")
                changed_fraction = data.get("changed_fraction", data.get("change_fraction"))
                quality = data.get("quality") or {}
                detector = str(data.get("detector", data.get("model", "M2 change detector")))
                baseline = "fallback" in detector.lower() or "baseline" in detector.lower()

                if operation == "detect_new":
                    subject = target or "new construction"
                    if detected:
                        first = (
                            f"1. Candidate {subject} areas: {regions} change region"
                            f"{'s' if regions != 1 else ''} detected between the two images."
                        )
                    else:
                        first = f"1. Candidate {subject} areas: no change region was returned."

                    second = ResponseSynthesizer._format_changed_surface(
                        area=area,
                        changed_fraction=changed_fraction,
                        quality=quality,
                    )
                    third = ResponseSynthesizer._format_location(quality)
                    if baseline:
                        fourth = (
                            "4. Evidence note: these are candidate temporal-change regions; "
                            "the current M2 baseline does not semantically prove that every region is a new building or construction."
                        )
                    else:
                        fourth = (
                            "4. Method: target-guided M2 change detection was used for the requested new-change target."
                        )
                    return "\n".join((first, second, third, fourth))

                if detected:
                    first = (
                        f"1. Change detected: {regions} temporal change zone"
                        f"{'s' if regions != 1 else ''}."
                    )
                else:
                    first = "1. Change detected: no temporal change zone was returned."

                second = ResponseSynthesizer._format_changed_surface(
                    area=area,
                    changed_fraction=changed_fraction,
                    quality=quality,
                )
                third = ResponseSynthesizer._format_location(quality)
                if baseline:
                    fourth = (
                        "4. Method: deterministic temporal image-difference baseline; "
                        "it detects visual change rather than assigning semantic object classes."
                    )
                else:
                    fourth = "4. Method: M2 change-detection specialist result."
                if m5 and m5.status == ExecutionStatus.PARTIAL:
                    fourth += " GIS enrichment is partial because spatial reference data is unavailable."
                return "\n".join((first, second, third, fourth))

        preferred = ("answer", "summary", "description", "finding", "message", "result")
        messages: list[str] = []
        for result in results:
            if result.status == ExecutionStatus.FAILED:
                continue
            for field in preferred:
                value = result.data.get(field)
                if isinstance(value, str) and value.strip():
                    messages.append(value.strip())
                    break
        if messages:
            return " ".join(messages)
        if any(r.status == ExecutionStatus.PARTIAL for r in results):
            return "Analysis completed partially; review the returned evidence artifacts."
        if any(r.status == ExecutionStatus.FAILED for r in results):
            return "Analysis could not be completed because a required specialist failed."
        return "Analysis completed successfully."

    @staticmethod
    def _format_changed_surface(
        area: object,
        changed_fraction: object,
        quality: dict,
    ) -> str:
        if area is not None and quality.get("georeferenced", False):
            try:
                hectares = float(area) / 10000.0
                return f"2. Changed surface: approximately {hectares:.2f} hectares."
            except (TypeError, ValueError):
                pass
        if changed_fraction is not None:
            try:
                return f"2. Changed surface: {float(changed_fraction) * 100:.2f}% of the image area."
            except (TypeError, ValueError):
                pass
        return "2. Changed surface: not available from the returned evidence."

    @staticmethod
    def _format_location(quality: dict) -> str:
        if quality.get("georeferenced", False):
            return "3. Location: geographic coordinates are available from the supplied raster reference."
        return "3. Location: image space only; the supplied images are not georeferenced."

    @staticmethod
    def _aggregate_confidence(results: list[ToolResult]) -> float:
        values = [r.confidence for r in results if r.status != ExecutionStatus.FAILED]
        return min(values) if values else 0.0

    @staticmethod
    def _collect_evidence(results: list[ToolResult]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for result in results:
            evidence.extend(result.evidence)
        return evidence

    @staticmethod
    def _build_trace(results: list[ToolResult]) -> list[str]:
        return [f"STEP {i}: {result.tool.value} → {result.status.value}" for i, result in enumerate(results, 1)]

    @staticmethod
    def _extract_error(results: list[ToolResult]) -> str | None:
        for result in results:
            if result.status == ExecutionStatus.FAILED:
                return result.error
        return None


synthesizer = ResponseSynthesizer()
