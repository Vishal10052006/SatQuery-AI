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


def _normalize_external_status(
    raw_status: Any,
) -> ExecutionStatus:
    """Map specialist-native status values to the M4 status enum."""
    normalized = str(raw_status or "success").strip().lower()

    if normalized in {"failed", "failure", "error"}:
        return ExecutionStatus.FAILED

    if normalized in {
        "partial",
        "awaiting_input",
        "awaiting_model",
        "incomplete",
    }:
        return ExecutionStatus.PARTIAL

    return ExecutionStatus.SUCCESS


def convert_tool_output(
    raw_output: Any,
    tool: ToolName,
) -> ToolResult:
    """
    Convert an external specialist ToolOutput into M4 ToolResult.

    Native status/error fields are preserved so a specialist failure is
    not incorrectly reported as a successful M4 step.
    """

    answer = getattr(raw_output, "answer", "")
    confidence = float(
        getattr(raw_output, "confidence", 0.0)
    )
    confidence = max(0.0, min(1.0, confidence))

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

    native_status = getattr(
        raw_output,
        "status",
        "success",
    )
    status = _normalize_external_status(native_status)

    native_error = getattr(raw_output, "error", None)
    native_message = getattr(raw_output, "message", None)
    error = (
        str(native_error)
        if native_error
        else (
            str(native_message)
            if status == ExecutionStatus.FAILED and native_message
            else None
        )
    )

    data: dict[str, Any] = {
        "answer": answer,
        "metrics": metrics,
        "native_status": str(native_status),
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
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=error,
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
        specialist.execute(image_paths=[...], params={...})

    Compatibility interface:
        specialist(**fallback_kwargs)

    This supports both structured specialist classes and lightweight
    callables used during testing/integration.
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
