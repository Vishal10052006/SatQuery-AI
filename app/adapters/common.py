"""
Common adapter utilities for integrating external specialist tools
with the M4 ToolResult contract.
"""

from __future__ import annotations

from typing import Any

from app.query.schemas import (
    Evidence,
    ExecutionStatus,
    ToolName,
    ToolResult,
)


def convert_tool_output(
    raw_output: Any,
    tool: ToolName,
) -> ToolResult:
    """
    Convert an external specialist ToolOutput into M4 ToolResult.

    The external SatQuery specialists return ToolOutput objects,
    while M4 operates on the stable ToolResult contract.
    """

    answer = getattr(raw_output, "answer", "")
    confidence = float(
        getattr(raw_output, "confidence", 0.0)
    )

    metrics = getattr(
        raw_output,
        "metrics",
        {},
    ) or {}

    warnings = getattr(
        raw_output,
        "warnings",
        [],
    ) or []

    data: dict[str, Any] = {
        "answer": answer,
        "metrics": metrics,
    }

    # Preserve geospatial mask output when available.
    mask_geojson = getattr(
        raw_output,
        "mask_geojson",
        None,
    )

    if mask_geojson is not None:
        data["mask_geojson"] = mask_geojson

    # Preserve bounding-box output when available.
    bboxes = getattr(
        raw_output,
        "bboxes",
        None,
    )

    if bboxes is not None:
        data["bboxes"] = bboxes

    # Preserve specialist warnings.
    if warnings:
        data["warnings"] = warnings

    evidence: list[Evidence] = []

    if mask_geojson is not None:
        evidence.append(
            Evidence(
                type="geojson",
                reference="specialist_output",
                description=(
                    "Geospatial output generated "
                    "by specialist."
                ),
            )
        )

    if bboxes:
        evidence.append(
            Evidence(
                type="bounding_box",
                reference="specialist_output",
                description=(
                    "Bounding boxes generated "
                    "by specialist."
                ),
            )
        )

    return ToolResult(
        tool=tool,
        status=ExecutionStatus.SUCCESS,
        confidence=confidence,
        data=data,
        evidence=evidence,
    )


def invoke_specialist(
    specialist: Any,
    *,
    image_paths: list[Any] | None = None,
    params: dict[str, Any] | None = None,
    fallback_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Invoke an external specialist through a stable compatibility boundary.

    Preferred interface:
        specialist.execute(
            image_paths=[...],
            params={...},
        )

    Compatibility interface:
        specialist(**fallback_kwargs)

    This allows M4 to integrate both the structured specialist
    classes supplied by teammates and lightweight callable
    implementations used during testing/integration.
    """

    if hasattr(specialist, "execute") and callable(
        specialist.execute
    ):
        return specialist.execute(
            image_paths=image_paths or [],
            params=params or {},
        )

    if callable(specialist):
        return specialist(
            **(fallback_kwargs or {}),
        )

    raise TypeError(
        "Specialist must expose execute() or be callable."
    )
