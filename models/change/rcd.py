"""Referring Change Detection adapter module.

Provides natural-language target-guided change detection:
Before Image + After Image + Text Target -> Target-specific change mask & regions.

If a trained neural RCD model/checkpoint is available locally, it runs inference.
If unavailable, it uses an explicitly labelled deterministic temporal baseline.
The fallback applies a conservative water-body exclusion for land/construction
queries so persistent rivers, lakes, and reservoirs are not presented as land
change merely because their appearance shifts between observations.
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

    status: str
    model_name: str
    detector_type: str
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


def _land_change_query(target: str | None) -> bool:
    """Return True for targets where water is outside the requested class."""
    if not target:
        return False
    text = target.lower()
    land_terms = (
        "land", "construction", "constructed", "building", "built-up",
        "built up", "urban", "road", "infrastructure", "development",
        "vegetation", "agriculture", "crop", "deforestation", "forest",
    )
    return any(term in text for term in land_terms)


def _likely_water_mask(image: PreprocessedImage) -> np.ndarray:
    """Estimate persistent open-water pixels from RGB appearance.

    This is deliberately conservative and is only used by the deterministic
    fallback. It is not a substitute for a multispectral water index such as
    NDWI/MNDWI. The goal is to prevent dark/blue water from becoming a false
    land-change region when the water surface shifts or changes reflectance.
    """
    if image.data.ndim != 3 or image.data.shape[-1] < 3:
        return np.zeros(image.gray.shape, dtype=bool)

    red = image.data[:, :, 0]
    green = image.data[:, :, 1]
    blue = image.data[:, :, 2]
    brightness = (red + green + blue) / 3.0
    eps = 1e-6
    blue_ratio = blue / (red + green + blue + eps)

    # Two conservative signatures cover dark inland water and blue/cyan water.
    dark_water = (brightness < 0.34) & (blue >= red * 1.02) & (green >= red * 0.98)
    blue_water = (blue_ratio > 0.36) & (blue >= red * 1.08) & (green >= red * 0.98)
    return dark_water | blue_water


def _suppress_water_change(
    diff: np.ndarray,
    before: PreprocessedImage,
    after: PreprocessedImage,
    target: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove water pixels from a land-change temporal difference map."""
    if not _land_change_query(target):
        return diff, np.zeros(diff.shape, dtype=bool)

    before_water = _likely_water_mask(before)
    after_water = _likely_water_mask(after)

    # Union is intentional: if the river/lake boundary moves, both its old and
    # new water footprints are excluded from land-change evidence.
    water = before_water | after_water
    filtered = diff.copy()
    filtered[water] = 0.0
    return filtered, water


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

        # The fallback is intentionally target-aware only for exclusion. It does
        # not claim semantic recognition of buildings, roads, or other classes.
        diff = np.abs(after.gray - before.gray)
        valid_mask = before.valid_mask & after.valid_mask
        diff[~valid_mask] = 0.0

        diff, water_mask = _suppress_water_change(diff, before, after, target)
        raw_mask = diff >= float(threshold)
        raw_mask[water_mask] = False

        clean_mask, regions = extract_change_regions(raw_mask, diff, min_pixels=min_pixels)

        for r in regions:
            r.target = target

        claim = (
            f"Land-surface change regions surfaced for '{target}' using a "
            "deterministic temporal baseline; persistent water was excluded "
            "from land-change evidence."
            if target and _land_change_query(target)
            else (
                f"Temporal change regions surfaced for target '{target}' using a "
                "deterministic fallback baseline; target semantics were not modeled."
                if target
                else "Detected changed regions using temporal pixel difference."
            )
        )

        fallback_confidence = 0.35 if regions else 0.50
        if _land_change_query(target):
            fallback_confidence = min(0.55, fallback_confidence + 0.05)

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
