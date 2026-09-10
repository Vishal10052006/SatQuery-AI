"""Bi-temporal image alignment and grid registration module.

Verifies that before and after satellite images correspond spatially:
1. Matching CRS, transform, and dimensions -> already aligned.
2. Differing dimensions -> safe resampling of the after-image onto the reference (before) grid.
3. Incompatible CRS or disjoint bounds -> explicit alignment failure.
4. Preserves reference geospatial metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from .preprocess import PreprocessedImage


@dataclass
class AlignmentResult:
    """Outcome of the bi-temporal alignment / registration step."""

    success: bool
    before: PreprocessedImage
    after: PreprocessedImage
    status: str              # 'already_aligned' | 'resampled' | 'failed'
    method: str              # 'identity' | 'bilinear_grid_resample' | 'none'
    quality_score: float     # 1.0 = identical grid, 0.9 = resampled, 0.0 = failed
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "method": self.method,
            "quality_score": self.quality_score,
            "error": self.error,
            "warnings": self.warnings,
        }


def _resample_array(
    arr: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resample 2D or 3D float32 array to target (height, width) using PIL bilinear filter."""
    th, tw = target_shape
    h, w = arr.shape[:2]
    if (h, w) == (th, tw):
        return arr

    if arr.ndim == 2:
        pil_img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        resized = pil_img.resize((tw, th), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    # Multi-channel
    channels = []
    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        pil_ch = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8))
        res_ch = pil_ch.resize((tw, th), Image.Resampling.BILINEAR)
        channels.append(np.asarray(res_ch, dtype=np.float32) / 255.0)
    return np.stack(channels, axis=-1)


def _resample_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resample boolean mask using nearest neighbor."""
    th, tw = target_shape
    h, w = mask.shape
    if (h, w) == (th, tw):
        return mask
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil_mask.resize((tw, th), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def align_images(
    before: PreprocessedImage,
    after: PreprocessedImage,
) -> AlignmentResult:
    """Align the after image onto the reference grid of the before image."""
    warnings: list[str] = []

    b_meta = before.metadata
    a_meta = after.metadata

    b_shape = before.original_shape
    a_shape = after.original_shape

    # Check CRS compatibility when both are georeferenced
    if b_meta.get("georeferenced") and a_meta.get("georeferenced"):
        b_crs = b_meta.get("crs")
        a_crs = a_meta.get("crs")
        if b_crs and a_crs and b_crs.strip().upper() != a_crs.strip().upper():
            return AlignmentResult(
                success=False,
                before=before,
                after=after,
                status="failed",
                method="none",
                quality_score=0.0,
                error=(
                    f"Incompatible CRS: before has '{b_crs}', after has '{a_crs}'. "
                    "Reprojection required before alignment."
                ),
            )

        # Check bounds overlap if both bounds are available
        b_bounds = b_meta.get("bounds")
        a_bounds = a_meta.get("bounds")
        if b_bounds and a_bounds:
            # Check for disjoint bounds
            disjoint_x = (
                b_bounds["right"] < a_bounds["left"] or a_bounds["right"] < b_bounds["left"]
            )
            disjoint_y = (
                b_bounds["top"] < a_bounds["bottom"] or a_bounds["top"] < b_bounds["bottom"]
            )
            if disjoint_x or disjoint_y:
                return AlignmentResult(
                    success=False,
                    before=before,
                    after=after,
                    status="failed",
                    method="none",
                    quality_score=0.0,
                    error="Images have completely disjoint spatial bounds and do not overlap.",
                )

    # Check if already aligned (same dimensions and matching transform)
    same_shape = (b_shape == a_shape)
    b_transform = b_meta.get("transform")
    a_transform = a_meta.get("transform")
    same_transform = (b_transform == a_transform)

    if same_shape and (not b_meta.get("georeferenced") or same_transform):
        return AlignmentResult(
            success=True,
            before=before,
            after=after,
            status="already_aligned",
            method="identity",
            quality_score=1.0,
            warnings=warnings,
        )

    # Need resampling of after image onto before image grid
    warnings.append(
        f"Resampling after image from {a_shape[1]}x{a_shape[0]} to reference grid {b_shape[1]}x{b_shape[0]}."
    )

    resampled_data = _resample_array(after.data, b_shape)
    resampled_gray = _resample_array(after.gray, b_shape)
    resampled_valid = _resample_mask(after.valid_mask, b_shape)

    # Inherit reference grid metadata
    aligned_meta = dict(after.metadata)
    aligned_meta["resampled_to_reference"] = True
    aligned_meta["reference_shape"] = b_shape
    aligned_meta["transform"] = b_meta.get("transform")
    aligned_meta["crs"] = b_meta.get("crs")
    aligned_meta["bounds"] = b_meta.get("bounds")

    aligned_after = PreprocessedImage(
        data=resampled_data,
        gray=resampled_gray,
        valid_mask=resampled_valid,
        metadata=aligned_meta,
        original_shape=b_shape,
    )

    return AlignmentResult(
        success=True,
        before=before,
        after=aligned_after,
        status="resampled",
        method="bilinear_grid_resample",
        quality_score=0.92,
        warnings=warnings,
    )
