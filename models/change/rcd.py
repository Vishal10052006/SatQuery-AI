"""Referring Change Detection adapter module.

Provides natural-language target-guided change detection:
Before Image + After Image + Text Target -> Target-specific change mask & regions.

If a trained neural RCD model/checkpoint is available locally, it runs inference.
If unavailable, it returns explicit status "awaiting_model" or an honest
deterministic fallback clearly labeled as 'detector_type="fallback_baseline"'.
Never fabricates neural predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .mask_processing import ChangeRegion, extract_change_regions
from .preprocess import PreprocessedImage


@dataclass
class RCDResult:
    """Structured result of Referring Change Detection."""

    status: str                 # 'success' | 'awaiting_model' | 'fallback_baseline' | 'failed'
    model_name: str
    detector_type: str          # 'neural' | 'fallback_baseline' | 'adapter'
    target: str | None
    change_mask: np.ndarray | None
    difference_map: np.ndarray | None
    regions: list[ChangeRegion] = field(default_factory=list)
    confidence: float = 0.0
    message: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_name": self.model_name,
            "detector_type": self.detector_type,
            "target": self.target,
            "confidence": round(self.confidence, 4),
            "message": self.message,
            "error": self.error,
            "num_regions": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
        }


class RCDAdapter:
    """Adapter for Referring Change Detection networks with deterministic fallback."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model = None
        self._load_attempted = False

    def is_model_available(self) -> bool:
        """Check whether local neural model weights exist."""
        if self.checkpoint_path and self.checkpoint_path.is_file():
            return True
        default_ckpts = [
            Path("checkpoints/rcd_model.pth"),
            Path("checkpoints/rcd/model.pt"),
            Path("models/change/weights/rcd.pth"),
        ]
        return any(p.is_file() for p in default_ckpts)

    def detect_target_change(
        self,
        before: PreprocessedImage,
        after: PreprocessedImage,
        target: str | None,
        threshold: float = 0.15,
        min_pixels: int = 8,
        allow_fallback: bool = True,
    ) -> RCDResult:
        """Execute Referring Change Detection for a specific query target."""
        if self.is_model_available():
            return self._run_neural_inference(before, after, target)

        if not allow_fallback:
            return RCDResult(
                status="awaiting_model",
                model_name="rcd-neural-adapter",
                detector_type="adapter",
                target=target,
                change_mask=None,
                difference_map=None,
                confidence=0.0,
                message=(
                    f"Referring Change Detection for target '{target}' requires "
                    "a trained neural checkpoint (e.g. in checkpoints/rcd/)."
                ),
            )

        # This fallback detects temporal pixel differences only. It does NOT
        # understand the target semantics, so never present its regions as
        # target-specific neural predictions.
        diff = np.abs(after.gray - before.gray)
        valid_mask = before.valid_mask & after.valid_mask
        diff[~valid_mask] = 0.0

        raw_mask = diff >= float(threshold)
        clean_mask, regions = extract_change_regions(raw_mask, diff, min_pixels=min_pixels)

        for r in regions:
            r.target = target

        claim = (
            f"Temporal change regions surfaced for target '{target}' using a "
            "deterministic fallback baseline; target semantics were not modeled."
            if target
            else "Detected changed regions using temporal pixel difference."
        )

        # Fallback confidence is deliberately conservative because the target
        # itself has not been semantically recognized.
        fallback_confidence = 0.35 if regions else 0.50

        return RCDResult(
            status="fallback_baseline" if target else "success",
            model_name="rcd-fallback-baseline",
            detector_type="fallback_baseline",
            target=target,
            change_mask=clean_mask,
            difference_map=diff,
            regions=regions,
            confidence=fallback_confidence,
            message=claim,
        )

    def _run_neural_inference(
        self,
        before: PreprocessedImage,
        after: PreprocessedImage,
        target: str | None,
    ) -> RCDResult:
        """Placeholder for loading real PyTorch weights when mounted."""
        return RCDResult(
            status="awaiting_model",
            model_name="rcd-neural-checkpoint",
            detector_type="neural",
            target=target,
            change_mask=None,
            difference_map=None,
            confidence=0.0,
            message="Checkpoint found but model runtime dependencies not loaded.",
        )
