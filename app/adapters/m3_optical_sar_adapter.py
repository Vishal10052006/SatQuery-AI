"""
M4 adapter for the native M3 Optical + SAR subsystem.

Architecture:

    M4 request
        -> M3 adapter
        -> run_optical_sar_pipeline()
        -> OpticalSARPipelineResult
        -> M4 ToolResult

M4 owns the stable ToolResult contract.
M3 owns its native pipeline and result contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.query.schemas import (
    Evidence,
    ExecutionStatus,
    ToolName,
    ToolResult,
)

from modules.optical_sar.pipeline import (
    run_optical_sar_pipeline,
)


def _convert_m3_output(raw_output: Any) -> ToolResult:
    """
    Convert native M3 OpticalSARPipelineResult into M4 ToolResult.

    Native M3 output is preserved under:
        data["m3"]

    Important:
        No confidence values are fabricated here.
        The confidence score comes directly from M3.
    """

    native_status = getattr(
        raw_output,
        "status",
        "failed",
    )

    native_confidence = getattr(
        raw_output,
        "confidence",
        {},
    ) or {}

    confidence_score = native_confidence.get(
        "score",
        0.0,
    )

    confidence = float(
        confidence_score
        if confidence_score is not None
        else 0.0
    )

    if native_status == "success":
        status = ExecutionStatus.SUCCESS
    elif native_status == "partial":
        status = ExecutionStatus.PARTIAL
    else:
        status = ExecutionStatus.FAILED

    native_data = raw_output.to_dict()

    prediction = native_data.get(
        "prediction",
        {},
    )

    predicted_class = prediction.get(
        "predicted_class"
    )

    model_status = prediction.get(
        "status"
    )

    if predicted_class is not None:
        answer = (
            "Optical-SAR multimodal analysis completed. "
            f"Predicted class: {predicted_class}."
        )
    else:
        answer = (
            "Optical-SAR multimodal processing completed "
            "without a model class prediction."
        )

    data: dict[str, Any] = {
        "answer": answer,
        "task": "optical_sar_fusion",
        "model": "M3-optical-sar",
        "native_status": native_status,
        "model_status": model_status,
        "confidence": native_confidence,
        "prediction": prediction,
        "registration": native_data.get(
            "registration",
            {},
        ),
        "fusion": native_data.get(
            "fusion",
            {},
        ),
        "features": native_data.get(
            "features",
            {},
        ),
        "optical": native_data.get(
            "optical",
            {},
        ),
        "sar": native_data.get(
            "sar",
            {},
        ),
        "metadata": native_data.get(
            "metadata",
            {},
        ),
        # Preserve the complete native M3 result.
        "m3": native_data,
    }

    evidence: list[Evidence] = []

    evidence.append(
        Evidence(
            type="optical_sar_analysis",
            reference="m3_optical_sar_pipeline",
            description=(
                "Native M3 Optical + SAR multimodal "
                "processing evidence."
            ),
        )
    )

    if native_data.get("registration"):
        evidence.append(
            Evidence(
                type="registration",
                reference="m3_registration",
                description=(
                    "Optical/SAR registration and "
                    "alignment quality metrics."
                ),
            )
        )

    if native_data.get("prediction"):
        evidence.append(
            Evidence(
                type="prediction",
                reference="m3_prediction",
                description=(
                    "Native M3 multimodal model prediction "
                    "and model-status metadata."
                ),
            )
        )

    return ToolResult(
        tool=ToolName.M3_OPTICAL_SAR,
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=None,
    )


def build_optical_sar_adapter(
    specialist: Any | None = None,
):
    """
    Build the M4 M3 Optical-SAR adapter.

    Native M3 is the default execution path.

    `specialist` is retained for compatibility with the
    existing M4 registration contract and tests.
    """

    m3_runner = (
        specialist
        if specialist is not None
        else run_optical_sar_pipeline
    )

    def execute(
        optical: Any,
        sar: Any,
        target: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute native M3 Optical + SAR processing.

        Required:
            optical: Optical GeoTIFF path
            sar: SAR GeoTIFF path

        Optional kwargs:
            cloud_mask_path
            dem_path
            config
            model
            run_inference
        """

        if not optical:
            raise ValueError(
                "M3 Optical-SAR analysis requires "
                "an optical image path."
            )

        if not sar:
            raise ValueError(
                "M3 Optical-SAR analysis requires "
                "a SAR image path."
            )

        # Compatibility path for the existing M4 specialist
        # contract tests. Those tests inject a fake specialist.
        if specialist is not None:
            try:
                raw_output = m3_runner(
                    image_paths=[
                        Path(optical),
                        Path(sar),
                    ],
                    params={
                        "query": target or "",
                        **kwargs,
                    },
                )
            except TypeError:
                raw_output = m3_runner(
                    optical=optical,
                    sar=sar,
                    target=target,
                    **kwargs,
                )

            # Existing fake specialists return the generic
            # external result contract, so normalize it using
            # the existing common adapter behavior.
            from app.adapters.common import convert_tool_output

            return convert_tool_output(
                raw_output,
                ToolName.M3_OPTICAL_SAR,
            )

        # -----------------------------------------------------
        # Native M3 execution path
        # -----------------------------------------------------

        allowed_kwargs = {
            "cloud_mask_path",
            "dem_path",
            "config",
            "model",
            "run_inference",
        }

        pipeline_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in allowed_kwargs
        }

        raw_output = m3_runner(
            optical_path=str(Path(optical)),
            sar_path=str(Path(sar)),
            **pipeline_kwargs,
        )

        return _convert_m3_output(raw_output)

    return execute
