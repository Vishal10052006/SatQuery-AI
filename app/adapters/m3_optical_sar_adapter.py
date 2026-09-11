"""
M4 adapter for the native M3 Optical + SAR subsystem.

M4 owns the stable ToolResult contract while M3 owns its native pipeline and result contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.query.schemas import Evidence, ExecutionStatus, ToolName, ToolResult
from modules.optical_sar.pipeline import run_optical_sar_pipeline


def _convert_m3_output(raw_output: Any) -> ToolResult:
    """Convert native M3 output into the M4 ToolResult without fabricating confidence."""
    native_status = getattr(raw_output, "status", "failed")
    native_data = raw_output.to_dict()

    # OpticalSARPipelineResult stores the serialized confidence assessment under
    # ``confidence`` in to_dict(); do not read a non-existent raw ``confidence`` attribute.
    native_confidence = native_data.get("confidence", {}) or {}
    confidence_score = native_confidence.get("score")
    confidence = float(confidence_score) if confidence_score is not None else 0.0

    if native_status == "success":
        status = ExecutionStatus.SUCCESS
    elif native_status == "partial":
        status = ExecutionStatus.PARTIAL
    else:
        status = ExecutionStatus.FAILED

    prediction = native_data.get("prediction", {}) or {}
    predicted_class = prediction.get("predicted_class")
    model_status = prediction.get("status")

    if predicted_class is not None:
        answer = f"Optical-SAR multimodal analysis completed. Predicted class: {predicted_class}."
    else:
        answer = "Optical-SAR multimodal processing completed without a model class prediction."

    data: dict[str, Any] = {
        "answer": answer,
        "task": "optical_sar_fusion",
        "model": "M3-optical-sar",
        "native_status": native_status,
        "model_status": model_status,
        "confidence": native_confidence,
        "prediction": prediction,
        "registration": native_data.get("registration", {}),
        "fusion": native_data.get("fusion", {}),
        "features": native_data.get("features", {}),
        "optical": native_data.get("optical", {}),
        "sar": native_data.get("sar", {}),
        "metadata": native_data.get("metadata", {}),
        "m3": native_data,
    }

    evidence: list[Evidence] = [
        Evidence(
            type="optical_sar_analysis",
            reference="m3_optical_sar_pipeline",
            description="Native M3 Optical + SAR multimodal processing evidence.",
        )
    ]
    if native_data.get("registration"):
        evidence.append(Evidence(
            type="registration",
            reference="m3_registration",
            description="Optical/SAR registration and alignment quality metrics.",
        ))
    if prediction:
        evidence.append(Evidence(
            type="prediction",
            reference="m3_prediction",
            description="Native M3 multimodal model prediction and model-status metadata.",
        ))

    return ToolResult(
        tool=ToolName.M3_OPTICAL_SAR,
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=None,
    )


def build_optical_sar_adapter(specialist: Any | None = None):
    """Build the M4 M3 Optical-SAR adapter."""
    m3_runner = specialist if specialist is not None else run_optical_sar_pipeline

    def execute(optical: Any, sar: Any, target: str | None = None, **kwargs: Any) -> ToolResult:
        if not optical:
            raise ValueError("M3 Optical-SAR analysis requires an optical image path.")
        if not sar:
            raise ValueError("M3 Optical-SAR analysis requires a SAR image path.")

        if specialist is not None:
            try:
                raw_output = m3_runner(
                    image_paths=[Path(optical), Path(sar)],
                    params={"query": target or "", **kwargs},
                )
            except TypeError:
                raw_output = m3_runner(optical=optical, sar=sar, target=target, **kwargs)

            from app.adapters.common import convert_tool_output
            return convert_tool_output(raw_output, ToolName.M3_OPTICAL_SAR)

        allowed_kwargs = {"cloud_mask_path", "dem_path", "config", "model", "run_inference"}
        pipeline_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}
        raw_output = m3_runner(
            optical_path=str(Path(optical)),
            sar_path=str(Path(sar)),
            **pipeline_kwargs,
        )
        return _convert_m3_output(raw_output)

    return execute
