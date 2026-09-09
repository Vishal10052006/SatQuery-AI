"""
Command-line entry point for SatQuery AI M4.

This CLI demonstrates the M4 orchestration layer without
requiring the real teammate specialist implementations.

The demo tools are intentionally isolated inside this file.
They are not part of the production M4 architecture.
"""

from __future__ import annotations

import argparse

from app.agents.controller import AgentController
from app.query.registry import ToolRegistry
from app.query.schemas import (
    ExecutionStatus,
    QueryRequest,
    ToolName,
    ToolResult,
)


def build_demo_controller() -> AgentController:
    """
    Build a deterministic controller for local M4 demonstration.

    These tools simulate specialist responses only.
    """

    registry = ToolRegistry()

    registry.register(
        ToolName.M1_VQA,
        lambda **kwargs: ToolResult(
            tool=ToolName.M1_VQA,
            status=ExecutionStatus.SUCCESS,
            confidence=0.90,
            data={
                "answer": (
                    "The satellite image contains "
                    "visible land-use features."
                )
            },
        ),
    )

    registry.register(
        ToolName.M2_CHANGE_DETECTION,
        lambda **kwargs: ToolResult(
            tool=ToolName.M2_CHANGE_DETECTION,
            status=ExecutionStatus.SUCCESS,
            confidence=0.92,
            data={
                "answer": (
                    "Changed regions were detected "
                    "between the supplied images."
                )
            },
        ),
    )

    registry.register(
        ToolName.M2_GROUNDING,
        lambda **kwargs: ToolResult(
            tool=ToolName.M2_GROUNDING,
            status=ExecutionStatus.SUCCESS,
            confidence=0.90,
            data={
                "answer": (
                    "The detected regions were "
                    "spatially localized."
                )
            },
        ),
    )

    registry.register(
        ToolName.M3_OPTICAL_SAR,
        lambda **kwargs: ToolResult(
            tool=ToolName.M3_OPTICAL_SAR,
            status=ExecutionStatus.SUCCESS,
            confidence=0.89,
            data={
                "answer": (
                    "Optical and SAR information "
                    "was jointly analyzed."
                )
            },
        ),
    )

    registry.register(
        ToolName.M5_GIS,
        lambda **kwargs: ToolResult(
            tool=ToolName.M5_GIS,
            status=ExecutionStatus.SUCCESS,
            confidence=0.88,
            data={
                "answer": (
                    "Geospatial evidence was generated."
                )
            },
        ),
    )

    return AgentController(
        registry=registry,
    )


def main() -> None:
    """Run the M4 command-line interface."""

    parser = argparse.ArgumentParser(
        description="SatQuery AI — M4 Agent Controller"
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "Find newly constructed buildings "
            "between these two images."
        ),
        help="Natural-language satellite query.",
    )

    args = parser.parse_args()

    controller = build_demo_controller()

    context = {
        "before": "before.tif",
        "after": "after.tif",
        "images": ["image.tif"],
        "optical": "optical.tif",
        "sar": "sar.tif",
    }

    response = controller.run(
        request=QueryRequest(
            query=args.query,
        ),
        context=context,
    )

    print()
    print("=" * 50)
    print("       SATQUERY AI — M4 AGENT")
    print("=" * 50)

    print()
    print("QUERY:")
    print(args.query)

    print()
    print("STATUS:")
    print(response.status.value)

    print()
    print("INTENT:")
    print(
        response.intent.value
        if response.intent
        else None
    )

    print()
    print("ANSWER:")
    print(response.answer)

    print()
    print("CONFIDENCE:")
    print(response.confidence)

    print()
    print("TRACE:")

    for item in response.trace:
        print(f"  {item}")

    print()
    print("TOOLS:")

    for result in response.results:
        print(
            f"  - {result.tool.value}: "
            f"{result.status.value}"
        )

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()
