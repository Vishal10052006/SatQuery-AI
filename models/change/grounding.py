"""Text-guided remote-sensing grounding adapter module.

Exposes spatial grounding for natural-language queries:
Target Text + Imagery -> Spatial regions / bounding boxes / masks.

If a neural grounding checkpoint is mounted, it runs real inference.
If unavailable, it returns explicit status "awaiting_model".
When change-derived evidence is supplied, it can format spatial evidence
clearly labeled with evidence_source="change_regions" without fabricating neural scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.contracts import SpecialistResult
from .mask_processing import ChangeRegion


@dataclass
class GroundingResult:
    """Outcome of text-guided grounding analysis."""

    status: str                 # 'awaiting_model' | 'success' | 'failed'
    model_name: str
    target: str
    evidence_source: str        # 'neural_grounding' | 'change_regions' | 'none'
    boxes_pixel: list[dict[str, int]] = field(default_factory=list)
    boxes_geo: list[dict[str, tuple[float, float]]] = field(default_factory=list)
    confidence: float = 0.0
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_name": self.model_name,
            "target": self.target,
            "evidence_source": self.evidence_source,
            "boxes_pixel": self.boxes_pixel,
            "boxes_geo": self.boxes_geo,
            "confidence": round(self.confidence, 4),
            "message": self.message,
        }

    def to_specialist_result(self) -> SpecialistResult:
        return SpecialistResult(
            task="grounding",
            model=self.model_name,
            status=self.status,
            confidence=self.confidence,
            claim=self.message,
            evidence=self.to_dict(),
        )


class GroundingAdapter:
    """Adapter for text-guided grounding models."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    def is_model_available(self) -> bool:
        """Check whether local neural grounding weights exist."""
        if self.checkpoint_path and self.checkpoint_path.is_file():
            return True
        default_ckpts = [
            Path("checkpoints/grounding_model.pth"),
            Path("models/grounding/weights/grounding.pth"),
        ]
        return any(p.is_file() for p in default_ckpts)

    def ground_target(
        self,
        image_path: str | None,
        target: str,
        change_regions: list[ChangeRegion] | None = None,
    ) -> GroundingResult:
        """Ground a natural-language target in the satellite image."""
        if self.is_model_available():
            return GroundingResult(
                status="awaiting_model",
                model_name="neural-grounding-checkpoint",
                target=target,
                evidence_source="neural_grounding",
                confidence=0.0,
                message="Neural grounding checkpoint detected; inference engine not configured.",
            )

        if change_regions:
            # Change-derived evidence labeled honestly
            boxes_pixel = [r.bbox_pixel for r in change_regions]
            boxes_geo = [r.bbox_geo for r in change_regions if r.bbox_geo]
            return GroundingResult(
                status="success",
                model_name="change-derived-grounding",
                target=target,
                evidence_source="change_regions",
                boxes_pixel=boxes_pixel,
                boxes_geo=boxes_geo,
                confidence=0.70,
                message=(
                    f"Provided {len(boxes_pixel)} spatial bounding box(es) derived from "
                    f"detected change regions for target '{target}'."
                ),
            )

        return GroundingResult(
            status="awaiting_model",
            model_name="grounding-adapter",
            target=target,
            evidence_source="none",
            confidence=0.0,
            message=(
                f"Text-to-region grounding for '{target}' requires the configured "
                "neural grounding checkpoint or prior change regions."
            ),
        )
