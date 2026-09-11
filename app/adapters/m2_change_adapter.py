"""
M4 adapter for the native M2 bi-temporal change-detection subsystem.

M4 owns the stable ToolResult contract.
M2 owns the native change-detection implementation.

The adapter preserves all native M2 evidence required by M4/M5/M6 so
summary metrics are never lost between the specialist and the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import convert_tool_output, invoke_specialist
from app.query.schemas import Evidence, ExecutionStatus, ToolName, ToolResult
from models.change.adapter import run_change_detection


def _convert_m2_output(raw_output: Any) -> ToolResult:
    """Convert native M2 SpecialistResult into the M4 ToolResult contract."""
    claim = getattr(raw_output, "claim", "") or ""
    model = getattr(raw_output, "model", "M2") or "M2"
    confidence = float(getattr(raw_output, "confidence", 0.0) or 0.0)
    native_status = getattr(raw_output, "status", "failed")
    native_evidence = getattr(raw_output, "evidence", {}) or {}
    artifacts = getattr(raw_output, "artifacts", []) or []
    error = getattr(raw_output, "error", None)

    if native_status == "failed":
        status = ExecutionStatus.FAILED
    elif native_status in {"awaiting_model", "awaiting_input"}:
        status = ExecutionStatus.PARTIAL
    else:
        status = ExecutionStatus.SUCCESS

    data: dict[str, Any] = {
        "answer": claim,
        "task": getattr(raw_output, "task", "change_detection"),
        "model": model,
        "native_status": native_status,
        "evidence": native_evidence,
        "artifacts": artifacts,
    }

    if isinstance(native_evidence, dict):
        for key in (
            "target",
            "change_detected",
            "changed_pixels",
            "changed_fraction",
            "change_fraction",
            "mean_difference",
            "regions",
            "number_of_regions",
            "region_sizes",
            "detector",
            "detector_type",
            "quality",
            "warnings",
            "geospatial_reference_available",
            "crs",
            "transform",
            "geographic_bbox",
            "total_area_sq_m",
            "changed_area_sq_m",
            "change_mask",
            "change_mask_path",
            "overlay_path",
            "mask_path",
            "composite_path",
            "before",
            "after",
            "before_raster",
            "after_raster",
        ):
            if key in native_evidence:
                data[key] = native_evidence[key]

    before_path = native_evidence.get("before") if isinstance(native_evidence, dict) else None
    after_path = native_evidence.get("after") if isinstance(native_evidence, dict) else None

    if before_path:
        data["before"] = str(before_path)
    if after_path:
        data["after"] = str(after_path)
        data["reference_image"] = str(after_path)
        data["reference_image_path"] = str(after_path)
        if Path(str(after_path)).suffix.lower() in {".tif", ".tiff"}:
            data["reference_geotiff"] = str(after_path)
            data["reference_geotiff_path"] = str(after_path)

    evidence: list[Evidence] = []
    if native_evidence:
        evidence.append(Evidence(
            type="change_detection_evidence",
            reference="m2_change_detection",
            description="Bi-temporal change-analysis evidence returned by the M2 subsystem.",
        ))

    return ToolResult(
        tool=ToolName.M2_CHANGE_DETECTION,
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=error,
    )


def build_change_detection_adapter(specialist: Any | None = None):
    """Build the M4 M2 change-detection adapter."""
    def execute(
        before: Any,
        after: Any,
        target: str | None = None,
        temporal: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        if not before or not after:
            raise ValueError("M2 change detection requires both before and after image paths.")

        before_path = Path(before)
        after_path = Path(after)

        if specialist is None or specialist is run_change_detection:
            raw_output = run_change_detection(
                before=str(before_path),
                after=str(after_path),
                target=target,
                output_dir=kwargs.get("output_dir"),
            )
            return _convert_m2_output(raw_output)

        params = {"query_hint": target or "", "temporal": temporal, **kwargs}
        raw_output = invoke_specialist(
            specialist,
            image_paths=[before_path, after_path],
            params=params,
            fallback_kwargs={
                "before": before,
                "after": after,
                "target": target,
                "temporal": temporal,
                **kwargs,
            },
        )
        return convert_tool_output(raw_output, ToolName.M2_CHANGE_DETECTION)

    return execute
