"""
M4 adapter for the native M2 bi-temporal change-detection subsystem.

M4 owns the stable ToolResult contract.
M2 owns the native change-detection implementation.

The adapter also preserves the original M4 specialist compatibility
interface so existing contract tests and external integrations continue
to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import (
    convert_tool_output,
    invoke_specialist,
)
from app.query.schemas import (
    Evidence,
    ExecutionStatus,
    ToolName,
    ToolResult,
)

from models.change.adapter import run_change_detection


def _convert_m2_output(raw_output: Any) -> ToolResult:
    """
    Convert native M2 SpecialistResult into the M4 ToolResult contract.
    """

    claim = getattr(raw_output, "claim", "") or ""
    model = getattr(raw_output, "model", "M2") or "M2"

    confidence = float(
        getattr(raw_output, "confidence", 0.0)
    )

    native_status = getattr(
        raw_output,
        "status",
        "failed",
    )

    native_evidence = getattr(
        raw_output,
        "evidence",
        {},
    ) or {}

    artifacts = getattr(
        raw_output,
        "artifacts",
        [],
    ) or []

    error = getattr(
        raw_output,
        "error",
        None,
    )

    # Translate native M2 status into the M4 execution status.
    if native_status == "failed":
        status = ExecutionStatus.FAILED

    elif native_status == "awaiting_model":
        status = ExecutionStatus.PARTIAL

    elif native_status == "awaiting_input":
        status = ExecutionStatus.FAILED

    else:
        # Normal native M2 success path.
        status = ExecutionStatus.SUCCESS

    data: dict[str, Any] = {
        "answer": claim,
        "task": getattr(
            raw_output,
            "task",
            "change_detection",
        ),
        "model": model,
        "native_status": native_status,
        "evidence": native_evidence,
        "artifacts": artifacts,
    }

    # -------------------------------------------------------------
    # Preserve M2 input/output provenance for downstream M4 tools.
    #
    # Native M2 stores the original temporal image paths inside
    # SpecialistResult.evidence. Promote them into the stable M4
    # ToolResult contract so Grounding and M5 GIS can consume them.
    # -------------------------------------------------------------
    before_path = native_evidence.get("before")
    after_path = native_evidence.get("after")

    if before_path:
        data["before"] = str(before_path)

    if after_path:
        data["after"] = str(after_path)

    # M5 requires a reference raster. M2 uses the "after" observation
    # as the reference image for downstream spatial interpretation.
    if after_path:
        data["reference_image"] = str(after_path)
        data["reference_image_path"] = str(after_path)

        # Preserve an explicit GeoTIFF alias only when the source
        # itself is actually a GeoTIFF/TIFF.
        if Path(str(after_path)).suffix.lower() in {".tif", ".tiff"}:
            data["reference_geotiff"] = str(after_path)
            data["reference_geotiff_path"] = str(after_path)

    # Preserve M2-native spatial evidence.
    if "change_mask" in native_evidence:
        data["change_mask"] = native_evidence["change_mask"]

    if "change_mask_path" in native_evidence:
        data["change_mask_path"] = native_evidence["change_mask_path"]

    if "regions" in native_evidence:
        data["regions"] = native_evidence["regions"]

    if "target" in native_evidence:
        data["target"] = native_evidence["target"]

    if "change_detected" in native_evidence:
        data["change_detected"] = native_evidence["change_detected"]

    if "geospatial_reference_available" in native_evidence:
        data["geospatial_reference_available"] = (
            native_evidence["geospatial_reference_available"]
        )

    evidence: list[Evidence] = []

    if native_evidence:
        evidence.append(
            Evidence(
                type="change_detection_evidence",
                reference="m2_change_detection",
                description=(
                    "Bi-temporal change-analysis evidence "
                    "returned by the M2 subsystem."
                ),
            )
        )

    return ToolResult(
        tool=ToolName.M2_CHANGE_DETECTION,
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=error,
    )


def build_change_detection_adapter(
    specialist: Any | None = None,
):
    """
    Build the M4 M2 change-detection adapter.

    When no specialist is supplied, the native M2 implementation is used.

    When a specialist is supplied, the original M4 compatibility interface
    is preserved. This allows external specialists and existing contract
    tests to continue working without changing the M4 boundary.
    """

    def execute(
        before: Any,
        after: Any,
        target: str | None = None,
        temporal: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute change detection through either:

        1. Native M2 implementation.
        2. Existing external-specialist compatibility interface.
        """

        if not before or not after:
            raise ValueError(
                "M2 change detection requires both "
                "before and after image paths."
            )

        before_path = Path(before)
        after_path = Path(after)

        # ---------------------------------------------------------
        # Native M2 path
        # ---------------------------------------------------------
        #
        # This is the real integration used by:
        #
        #     build_m4_controller(
        #         change_detection=run_change_detection
        #     )
        #
        # The native M2 callable has the contract:
        #
        #     run_change_detection(
        #         before,
        #         after,
        #         target,
        #         output_dir,
        #     )
        #
        if specialist is None or specialist is run_change_detection:
            raw_output = run_change_detection(
                before=str(before_path),
                after=str(after_path),
                target=target,
                output_dir=kwargs.get("output_dir"),
            )

            return _convert_m2_output(raw_output)

        # ---------------------------------------------------------
        # External-specialist compatibility path
        # ---------------------------------------------------------
        #
        # Preserve the original M4 adapter behavior for external
        # specialists supplied by register_specialists().
        #
        params = {
            "query_hint": target or "",
            "temporal": temporal,
            **kwargs,
        }

        raw_output = invoke_specialist(
            specialist,
            image_paths=[
                before_path,
                after_path,
            ],
            params=params,
            fallback_kwargs={
                "before": before,
                "after": after,
                "target": target,
                "temporal": temporal,
                **kwargs,
            },
        )

        # External specialists use the generic M4-compatible output
        # contract: answer, confidence, metrics, warnings, etc.
        return convert_tool_output(
            raw_output,
            ToolName.M2_CHANGE_DETECTION,
        )

    return execute
