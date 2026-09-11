"""M4 deterministic response synthesis for specialist ToolResults."""
from __future__ import annotations

from app.agents.executor import ExecutionReport
from app.query.schemas import AgentResponse, Evidence, ExecutionStatus, Intent, ToolResult


class ResponseSynthesizer:
    """Turn specialist evidence into a query-specific, auditable answer."""

    def synthesize(self, intent: Intent, report: ExecutionReport) -> AgentResponse:
        results = report.results
        return AgentResponse(
            status=report.status,
            answer=self._build_answer(intent, results),
            confidence=self._aggregate_confidence(results),
            intent=intent,
            evidence=self._collect_evidence(results),
            results=results,
            trace=self._build_trace(results),
            error=self._extract_error(results),
        )

    @staticmethod
    def _build_answer(intent: Intent, results: list[ToolResult]) -> str:
        """Generate the answer from structured specialist evidence."""
        if not results:
            return "No analysis result was produced."

        if intent == Intent.CHANGE_DETECTION:
            m2 = next((r for r in results if r.tool.value == "m2_change_detection"), None)
            m5 = next((r for r in results if r.tool.value == "m5_gis"), None)
            if m2:
                data = m2.data or {}
                target = data.get("target")
                region_values = data.get("regions", [])
                regions = len(region_values) if isinstance(region_values, list) else int(data.get("number_of_regions", 0))
                detected = bool(data.get("change_detected", False)) or regions > 0
                area = data.get("changed_area_sq_m")
                changed_fraction = data.get("changed_fraction", data.get("change_fraction"))
                quality = data.get("quality") or {}
                water_excluded = any("water" in str(w).lower() or "river" in str(w).lower() or "lake" in str(w).lower() for w in data.get("warnings", []))
                detector = str(data.get("detector", data.get("model", "M2 change detector")))
                baseline = "fallback" in detector.lower() or "baseline" in detector.lower()

                subject = f" for '{target}'" if target else ""
                if detected:
                    if baseline:
                        first = f"1. Change detected: {regions} temporal difference zone{'s' if regions != 1 else ''}{subject}."
                    else:
                        first = f"1. Change detected: {regions} change zone{'s' if regions != 1 else ''}{subject}."

                    if area is not None and quality.get("georeferenced", False):
                        try:
                            hectares = float(area) / 10000.0
                            second = f"2. Changed surface: approximately {hectares:.2f} hectares."
                        except (TypeError, ValueError):
                            second = "2. Changed surface: not available from the supplied spatial reference."
                    elif changed_fraction is not None:
                        try:
                            second = f"2. Changed surface: {float(changed_fraction) * 100:.2f}% of the image area."
                        except (TypeError, ValueError):
                            second = "2. Changed surface: not available."
                    else:
                        second = "2. Changed surface: not available from the returned evidence."
                else:
                    first = f"1. Change detected: no temporal difference zone was returned{subject}."
                    second = "2. Changed surface: 0% of the image area based on the returned change mask." 

                third = "3. Location: image space only; the supplied images are not georeferenced."
                if quality.get("georeferenced", False):
                    third = "3. Location: geographic coordinates are available from the supplied raster reference."

                if baseline:
                    fourth = "4. Method note: M2 is using a deterministic temporal image-difference baseline, not a semantic building/road classifier."
                else:
                    fourth = "4. Method: M2 change-detection specialist result."

                fifth = "5. Water: persistent water signatures were excluded from the land-change mask to reduce river/lake false positives." if water_excluded else "5. Water: not separately classified by M2."
                sixth = "6. GIS: geographic enrichment is partial because spatial reference data is unavailable." if m5 and m5.status == ExecutionStatus.PARTIAL else "6. GIS: geographic enrichment available where raster reference data is present."
                return "\n".join((first, second, third, fourth, fifth, sixth))

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
