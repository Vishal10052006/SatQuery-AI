"""
Tests for the M4 response synthesizer.
"""

from app.agents.executor import ExecutionReport
from app.agents.synthesizer import ResponseSynthesizer
from app.query.schemas import (
    Evidence,
    ExecutionStatus,
    Intent,
    ToolName,
    ToolResult,
)


def make_result(
    tool: ToolName,
    confidence: float,
    data: dict,
    evidence: list[Evidence] | None = None,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    error: str | None = None,
) -> ToolResult:
    """Create a standardized specialist result."""

    return ToolResult(
        tool=tool,
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence or [],
        error=error,
    )


def test_synthesizer_builds_success_response() -> None:
    """Successful tool results should become an AgentResponse."""

    result = make_result(
        ToolName.M1_VQA,
        confidence=0.94,
        data={
            "answer": "Agricultural land is visible."
        },
    )

    report = ExecutionReport(
        results=[result]
    )

    response = ResponseSynthesizer().synthesize(
        intent=Intent.VQA,
        report=report,
    )

    assert response.status == ExecutionStatus.SUCCESS
    assert response.intent == Intent.VQA
    assert response.answer == "Agricultural land is visible."
    assert response.confidence == 0.94
    assert response.results == [result]


def test_synthesizer_combines_multiple_results() -> None:
    """Multiple specialist outputs should be combined."""

    change_result = make_result(
        ToolName.M2_CHANGE_DETECTION,
        confidence=0.92,
        data={
            "answer": "New structures were detected."
        },
    )

    grounding_result = make_result(
        ToolName.M2_GROUNDING,
        confidence=0.88,
        data={
            "answer": "The structures were localized."
        },
    )

    gis_result = make_result(
        ToolName.M5_GIS,
        confidence=0.90,
        data={
            "answer": "Geospatial evidence was generated."
        },
    )

    report = ExecutionReport(
        results=[
            change_result,
            grounding_result,
            gis_result,
        ]
    )

    response = ResponseSynthesizer().synthesize(
        intent=Intent.CHANGE_DETECTION,
        report=report,
    )

    assert response.status == ExecutionStatus.SUCCESS

    assert (
        response.answer
        == "New structures were detected. "
        "The structures were localized. "
        "Geospatial evidence was generated."
    )

    # Conservative aggregation = weakest successful result.
    assert response.confidence == 0.88

    assert len(response.results) == 3

    assert response.trace == [
        "STEP 1: m2_change_detection → success",
        "STEP 2: m2_grounding → success",
        "STEP 3: m5_gis → success",
    ]


def test_synthesizer_collects_evidence() -> None:
    """Evidence from every specialist must reach AgentResponse."""

    evidence_1 = Evidence(
        type="change_mask",
        reference="change_mask.png",
        description="Detected change regions.",
    )

    evidence_2 = Evidence(
        type="polygon",
        reference="building_region.geojson",
        description="Localized building region.",
    )

    result_1 = make_result(
        ToolName.M2_CHANGE_DETECTION,
        confidence=0.91,
        data={"answer": "Change detected."},
        evidence=[evidence_1],
    )

    result_2 = make_result(
        ToolName.M5_GIS,
        confidence=0.89,
        data={"answer": "Region mapped."},
        evidence=[evidence_2],
    )

    report = ExecutionReport(
        results=[result_1, result_2]
    )

    response = ResponseSynthesizer().synthesize(
        intent=Intent.CHANGE_DETECTION,
        report=report,
    )

    assert len(response.evidence) == 2
    assert response.evidence[0].type == "change_mask"
    assert response.evidence[1].type == "polygon"


def test_synthesizer_handles_failure() -> None:
    """Failed execution should propagate failure and error."""

    failed_result = make_result(
        ToolName.M2_CHANGE_DETECTION,
        confidence=0.0,
        data={},
        status=ExecutionStatus.FAILED,
        error="Model unavailable.",
    )

    report = ExecutionReport(
        results=[failed_result]
    )

    response = ResponseSynthesizer().synthesize(
        intent=Intent.CHANGE_DETECTION,
        report=report,
    )

    assert response.status == ExecutionStatus.FAILED
    assert response.confidence == 0.0
    assert response.error == "Model unavailable."
    assert response.answer == "Analysis completed successfully."


def test_synthesizer_handles_partial_execution() -> None:
    """Partial results should preserve partial status."""

    partial_result = make_result(
        ToolName.M3_OPTICAL_SAR,
        confidence=0.71,
        data={
            "summary": "Optical analysis completed."
        },
        status=ExecutionStatus.PARTIAL,
    )

    report = ExecutionReport(
        results=[partial_result]
    )

    response = ResponseSynthesizer().synthesize(
        intent=Intent.OPTICAL_SAR,
        report=report,
    )

    assert response.status == ExecutionStatus.PARTIAL
    assert response.answer == "Optical analysis completed."
    assert response.confidence == 0.71


def test_synthesizer_preserves_partial_execution() -> None:
    """Partial execution should produce a PARTIAL AgentResponse."""

    from app.agents.executor import ExecutionReport
    from app.agents.synthesizer import ResponseSynthesizer
    from app.query.schemas import (
        ExecutionStatus,
        Intent,
        ToolName,
        ToolResult,
    )

    results = [
        ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.SUCCESS,
            confidence=0.90,
            data={
                "answer": "Changed building regions were detected."
            },
        ),
        ToolResult(
            tool=ToolName.M2_GROUNDING,
            status=ExecutionStatus.FAILED,
            confidence=0.0,
            error="Grounding unavailable.",
        ),
    ]

    response = ResponseSynthesizer().synthesize(
        intent=Intent.CHANGE_DETECTION,
        report=ExecutionReport(results=results),
    )

    assert response.status == ExecutionStatus.PARTIAL
    assert response.confidence == 0.90
    assert "Changed building regions were detected." in response.answer
    assert response.error == "Grounding unavailable."
    assert len(response.results) == 2
