"""M1: Earth-observation VLM adapter boundary."""
from __future__ import annotations

import os
from pathlib import Path

from core.contracts import SpecialistResult

from .geochat import run_geochat


def run_vqa(image_path: str | None, question: str) -> SpecialistResult:
    """Use optional GeoChat inference when configured, otherwise fail closed."""
    exists = bool(image_path and Path(image_path).exists())
    checkpoint_configured = bool(os.environ.get("SATQUERY_GEOCHAT_CHECKPOINT"))

    if exists and checkpoint_configured:
        result = run_geochat(str(image_path), question)
        if result["status"] == "success":
            return SpecialistResult(
                task="vqa",
                model="geochat-compatible-adapter",
                status="success",
                confidence=0.70,
                claim=result["answer"],
                evidence=result,
            )
        return SpecialistResult(
            task="vqa",
            model="geochat-compatible-adapter",
            status=result["status"],
            confidence=0.0,
            claim="Configured VLM inference did not produce an answer.",
            evidence={"image_available": exists, "question": question, **result},
            error=result.get("reason"),
        )

    return SpecialistResult(
        task="vqa",
        model="vlm-adapter-baseline",
        status="awaiting_model",
        confidence=0.0,
        claim=(
            "VLM checkpoint is not activated. Set "
            "SATQUERY_GEOCHAT_CHECKPOINT to a local GeoChat-compatible checkpoint."
        ),
        evidence={
            "image_available": exists,
            "question": question,
            "checkpoint_configured": checkpoint_configured,
        },
    )
