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
                water_excluded = any("water" in str(w).lower() for w in data.get("warnings", []))
                detector = str(data.get("detector", data.get("model", "M2 change detector")))
                baseline = "fallback" in detector.lower() or "baseline" in detector.lower()

                subject = f" for '{target}'" if target else ""
                if detected:
                    if baseline:
                        answer = f"Detected {regions} temporal difference zone{'s' if regions != 1 else ''}{subject} between the supplied observations."
                    else:
                        answer = f"Detected {regions} meaningful change zone{'s' if regions != 1 else ''}{subject} between the supplied observations."
                    if area is not None and quality.get("georeferenced", False):
                        try:
                            hectares = float(area) / 10000.0
                            answer += f" Estimated changed surface is {hectares:.2f} ha."
                        except (TypeError, ValueError):
                            pass
                    elif changed_fraction is not None:
                        try:
                            answer += f" The detected difference covers {float(changed_fraction) * 100:.2f}% of the image area."
                        except (TypeError, ValueError):
                            pass
                else:
                    answer = f"No temporal difference region was detected{subject} between the supplied observations."

                if baseline:
                    answer += " The current M2 fallback is a temporal image-difference baseline; it does not prove that every detected zone is a new building, road, or other semantic land-use change."
                if water_excluded:
                    answer += " Persistent water signatures were excluded from land-change evidence to reduce river/lake boundary false positives."
                if not quality.get("georeferenced", False):
                    answer += " The supplied images are not georeferenced, so localization is reported in image space rather than as geographic coordinates."
                if m5 and m5.status == ExecutionStatus.PARTIAL:
                    answer += " GIS enrichment is partial because geographic reference data is unavailable."
                return answer

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
