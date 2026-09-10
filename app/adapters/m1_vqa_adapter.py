"""
M4 adapter for the M1 EarthDial VQA specialist.

M4 owns the stable ToolResult contract.
M1 owns the EarthDial-specific inference contract.

This adapter translates:

    M4 VQA request
        -> M1 EarthDial inference
        -> M4 ToolResult
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


def _convert_m1_output(raw_output: Any) -> ToolResult:
    """
    Convert an M1 ImageVQAResponse into the M4 ToolResult contract.

    M1 deliberately does not fabricate confidence when the model
    cannot provide a calibrated score. Therefore None becomes the
    M4 neutral/default confidence of 0.0 rather than an invented value.
    """

    answer = getattr(raw_output, "answer", "") or ""
    model = getattr(raw_output, "model", "EarthDial")
    evidence_payload = getattr(raw_output, "evidence", None)
    success = bool(getattr(raw_output, "success", True))
    error = getattr(raw_output, "error", None)

    # M1 confidence is intentionally optional.
    raw_confidence = getattr(raw_output, "confidence", None)

    if raw_confidence is None:
        confidence = 0.0
    else:
        confidence = float(raw_confidence)

    # Preserve M1's structured output inside the M4 data boundary.
    data: dict[str, Any] = {
        "answer": answer,
        "model": model,
        "evidence": evidence_payload,
    }

    evidence: list[Evidence] = []

    # M1 evidence is specialist-native evidence. Keep it available
    # to downstream M4 synthesis without forcing it into M4's
    # Evidence schema prematurely.
    if evidence_payload:
        evidence.append(
            Evidence(
                type="specialist_evidence",
                reference="m1_earthdial",
                description="Evidence returned by M1 EarthDial.",
            )
        )

    if not success:
        return ToolResult(
            tool=ToolName.M1_VQA,
            status=ExecutionStatus.FAILED,
            confidence=confidence,
            data=data,
            evidence=evidence,
            error=error or "M1 EarthDial inference failed.",
        )

    return ToolResult(
        tool=ToolName.M1_VQA,
        status=ExecutionStatus.SUCCESS,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=error,
    )


def build_vqa_adapter(
    specialist: Any,
):
    """
    Build an M4 VQA adapter around the supplied M1 specialist.

    Supported M1 interfaces:

    1. EarthDial-style:
        specialist.analyze(
            image_path=...,
            question=...,
            **kwargs,
        )

    2. Top-level M1 interface:
        specialist(image_path, question, **kwargs)

    3. Existing generic M4 specialist interface:
        specialist.execute(
            image_paths=[...],
            params={...},
        )

    The first two are the native M1 integration path. The third
    preserves compatibility with the existing M4 adapter tests and
    teammate-provided specialist implementations.
    """

    def execute(
        images: Any,
        target: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:

        if not images:
            raise ValueError(
                "VQA requires at least one image."
            )

        image_paths = [
            Path(image)
            for image in images
        ]

        question = (
            target
            or "Describe this satellite image."
        )

        # ---------------------------------------------------------
        # Native M1 EarthDial adapter interface
        # ---------------------------------------------------------
        if hasattr(specialist, "analyze") and callable(
            specialist.analyze
        ):
            raw_output = specialist.analyze(
                image_path=str(image_paths[0]),
                question=question,
                **kwargs,
            )

            return _convert_m1_output(raw_output)

        # ---------------------------------------------------------
        # Native M1 top-level callable interface
        # ---------------------------------------------------------
        if callable(specialist) and not hasattr(
            specialist,
            "execute",
        ):
            raw_output = specialist(
                image_path=str(image_paths[0]),
                question=question,
                **kwargs,
            )

            # analyze_image() returns a dictionary, while an
            # EarthDialAdapter returns ImageVQAResponse.
            if isinstance(raw_output, dict):
                return _convert_m1_dict(raw_output)

            return _convert_m1_output(raw_output)

        # ---------------------------------------------------------
        # Existing generic specialist compatibility path
        # ---------------------------------------------------------
        if hasattr(specialist, "execute") and callable(
            specialist.execute
        ):
            raw_output = specialist.execute(
                image_paths=image_paths,
                params={
                    "query": question,
                    "capability": "vqa",
                    **kwargs,
                },
            )

            # Generic specialists already expose the old M4 shape.
            from app.adapters.common import convert_tool_output

            return convert_tool_output(
                raw_output,
                ToolName.M1_VQA,
            )

        raise TypeError(
            "M1 VQA specialist must expose analyze(), "
            "be callable, or expose execute()."
        )

    return execute


def _convert_m1_dict(raw_output: dict[str, Any]) -> ToolResult:
    """
    Convert the dictionary returned by m1_earthdial.analyze_image().
    """

    answer = raw_output.get("answer", "")
    model = raw_output.get("model", "EarthDial")
    confidence = raw_output.get("confidence")

    # M1 uses None when confidence is unavailable.
    if confidence is None:
        confidence_value = 0.0
    else:
        confidence_value = float(confidence)

    success = bool(raw_output.get("success", True))
    error = raw_output.get("error")

    data: dict[str, Any] = {
        "answer": answer,
        "model": model,
        "evidence": raw_output.get("evidence"),
    }

    evidence: list[Evidence] = []

    if raw_output.get("evidence"):
        evidence.append(
            Evidence(
                type="specialist_evidence",
                reference="m1_earthdial",
                description="Evidence returned by M1 EarthDial.",
            )
        )

    return ToolResult(
        tool=ToolName.M1_VQA,
        status=(
            ExecutionStatus.SUCCESS
            if success
            else ExecutionStatus.FAILED
        ),
        confidence=confidence_value,
        data=data,
        evidence=evidence,
        error=error,
    )
