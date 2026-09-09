"""
SatQuery AI — M4 Standalone Demonstration.

This demo exercises the complete M4 orchestration layer without
requiring teammate implementations.

The specialist functions below are DEMO adapters only.
Production integration happens through app/adapters/.
"""

from __future__ import annotations

from typing import Any

from app.agents.controller import AgentController
from app.query.registry import ToolRegistry
from app.query.schemas import (
    Evidence,
    ExecutionStatus,
    QueryRequest,
    ToolName,
    ToolResult,
)


# ============================================================
# DEMO SPECIALIST IMPLEMENTATIONS
# ============================================================
# These simulate M1/M2/M3/M5.
# M4 itself does NOT depend on these implementations.


def demo_vqa(**kwargs: Any) -> ToolResult:
    """Simulate M1 visual question answering."""

    return ToolResult(
        tool=ToolName.M1_VQA,
        status=ExecutionStatus.SUCCESS,
        confidence=0.91,
        data={
            "answer": (
                "The satellite scene contains visible "
                "urban and land-use features."
            )
        },
        evidence=[
            Evidence(
                type="image",
                reference="satellite_image.tif",
                description="Input satellite image.",
            )
        ],
    )


def demo_change_detection(
    before: Any,
    after: Any,
    target: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Simulate M2 change detection."""

    return ToolResult(
        tool=ToolName.M2_CHANGE_DETECTION,
        status=ExecutionStatus.SUCCESS,
        confidence=0.93,
        data={
            "answer": (
                f"Changes related to {target or 'the scene'} "
                "were detected between the two images."
            ),
            "changed_regions": [
                "region_1",
                "region_2",
            ],
        },
        evidence=[
            Evidence(
                type="change_mask",
                reference="change_mask.png",
                description="Detected changed regions.",
            )
        ],
    )


def demo_grounding(
    previous_result: ToolResult,
    target: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Simulate M2 spatial grounding."""

    return ToolResult(
        tool=ToolName.M2_GROUNDING,
        status=ExecutionStatus.SUCCESS,
        confidence=0.90,
        data={
            "answer": (
                f"The detected {target or 'changed'} regions "
                "were spatially localized."
            ),
            "locations": [
                "region_1",
                "region_2",
            ],
        },
        evidence=[
            Evidence(
                type="bounding_box",
                reference="building_boxes.json",
                description="Spatial localization output.",
            )
        ],
    )


def demo_optical_sar(
    optical: Any,
    sar: Any,
    target: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Simulate M3 optical/SAR analysis."""

    return ToolResult(
        tool=ToolName.M3_OPTICAL_SAR,
        status=ExecutionStatus.SUCCESS,
        confidence=0.89,
        data={
            "answer": (
                f"Optical and SAR imagery was jointly analyzed "
                f"for {target or 'the requested region'}."
            )
        },
        evidence=[
            Evidence(
                type="multimodal_analysis",
                reference="optical_sar_analysis.json",
                description="Optical/SAR analysis result.",
            )
        ],
    )


def demo_gis(
    previous_result: ToolResult,
    target: str | None = None,
    **kwargs: Any,
) -> ToolResult:
    """Simulate M5 GIS/evidence generation."""

    return ToolResult(
        tool=ToolName.M5_GIS,
        status=ExecutionStatus.SUCCESS,
        confidence=0.88,
        data={
            "answer": (
                "Geospatial evidence was generated for "
                "the localized regions."
            )
        },
        evidence=[
            Evidence(
                type="polygon",
                reference="building_regions.geojson",
                description="Generated geospatial polygons.",
            )
        ],
    )


# ============================================================
# BUILD DEMO RUNTIME
# ============================================================


def build_demo_controller() -> AgentController:
    """Create a complete M4 controller using demo specialists."""

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        demo_vqa,
    )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        demo_change_detection,
    )

    registry.register(
        ToolName.M2_GROUNDING,
        demo_grounding,
    )

    registry.register(
        ToolName.M3_OPTICAL_SAR,
        demo_optical_sar,
    )

    registry.register(
        ToolName.M5_GIS,
        demo_gis,
    )

    return AgentController(
        registry=registry,
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================


def print_separator() -> None:
    """Print a visual separator."""

    print("-" * 64)


def print_response(
    number: int,
    title: str,
    query: str,
    response: Any,
) -> None:
    """Display one M4 demonstration result."""

    print()
    print(f"[{number}] {title}")
    print_separator()

    print("QUERY:")
    print(f"  {query}")

    print()
    print("STATUS:")
    print(f"  {response.status.value}")

    print()
    print("INTENT:")
    print(
        f"  {response.intent.value}"
        if response.intent
        else "  None"
    )

    print()
    print("TRACE:")

    for item in response.trace:
        print(f"  {item}")

    print()
    print("ANSWER:")
    print(f"  {response.answer}")

    print()
    print("CONFIDENCE:")
    print(f"  {response.confidence:.2f}")

    if response.evidence:
        print()
        print("EVIDENCE:")

        for evidence in response.evidence:
            print(
                f"  - {evidence.type}: "
                f"{evidence.reference}"
            )


# ============================================================
# DEMONSTRATION
# ============================================================


def run_demo() -> None:
    """Run the complete standalone M4 demonstration."""

    controller = build_demo_controller()

    print()
    print("=" * 64)
    print("              SATQUERY AI — M4 DEMO")
    print("=" * 64)

    # --------------------------------------------------------
    # 1. VQA
    # --------------------------------------------------------

    query = "Describe this satellite image."

    response = controller.run(
        QueryRequest(query=query),
        context={
            "images": ["satellite_image.tif"],
        },
    )

    print_response(
        1,
        "VISUAL QUESTION ANSWERING",
        query,
        response,
    )

    # --------------------------------------------------------
    # 2. CHANGE DETECTION
    # --------------------------------------------------------

    query = "What changed between these two images?"

    response = controller.run(
        QueryRequest(query=query),
        context={
            "before": "before.tif",
            "after": "after.tif",
        },
    )

    print_response(
        2,
        "CHANGE DETECTION",
        query,
        response,
    )

    # --------------------------------------------------------
    # 3. MULTI-STEP NEW CONSTRUCTION
    # --------------------------------------------------------

    query = (
        "Find newly constructed buildings "
        "between these two images."
    )

    response = controller.run(
        QueryRequest(query=query),
        context={
            "before": "before.tif",
            "after": "after.tif",
        },
    )

    print_response(
        3,
        "MULTI-STEP BUILDING CHANGE ANALYSIS",
        query,
        response,
    )

    # --------------------------------------------------------
    # 4. OPTICAL + SAR
    # --------------------------------------------------------

    query = (
        "Use optical and SAR images "
        "to identify water regions."
    )

    response = controller.run(
        QueryRequest(query=query),
        context={
            "optical": "optical.tif",
            "sar": "sar.tif",
        },
    )

    print_response(
        4,
        "OPTICAL + SAR MULTIMODAL ANALYSIS",
        query,
        response,
    )

    # --------------------------------------------------------
    # 5. FAILURE HANDLING
    # --------------------------------------------------------

    query = "Describe this satellite image."

    response = controller.run(
        QueryRequest(query=query),
    )

    print_response(
        5,
        "CONTROLLED FAILURE HANDLING",
        query,
        response,
    )

    print()
    print("=" * 64)
    print("              M4 DEMONSTRATION COMPLETE")
    print("=" * 64)
    print()


if __name__ == "__main__":
    run_demo()
