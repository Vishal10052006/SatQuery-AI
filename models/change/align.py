"""Bi-temporal image alignment and grid registration module.

Verifies that before and after satellite images correspond spatially:
1. Matching CRS, transform, and dimensions -> already aligned.
2. Georeferenced rasters are reprojected/resampled onto the before grid.
3. Incompatible CRS or disjoint bounds -> explicit alignment failure.
4. Non-georeferenced images use deterministic pixel-grid resampling.
5. Reference geospatial metadata is preserved.
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
    """Resample 2D or 3D normalized arrays for non-georeferenced imagery."""
    th, tw = target_shape
    h, w = arr.shape[:2]
    if (h, w) == (th, tw):
        return arr

    if arr.ndim == 2:
        pil_img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
        resized = pil_img.resize((tw, th), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    channels = []
    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        pil_ch = Image.fromarray(np.clip(channel * 255.0, 0, 255).astype(np.uint8))
        res_ch = pil_ch.resize((tw, th), Image.Resampling.BILINEAR)
        channels.append(np.asarray(res_ch, dtype=np.float32) / 255.0)
    return np.stack(channels, axis=-1)


def _resample_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resample boolean mask using nearest neighbor."""
    th, tw = target_shape
    if mask.shape == target_shape:
        return mask
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil_mask.resize((tw, th), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def _geospatial_resample(
    after: PreprocessedImage,
    reference: PreprocessedImage,
) -> tuple[PreprocessedImage, str]:
    """Reproject a georeferenced after raster onto the exact reference grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    src_transform = after.metadata.get("transform")
    dst_transform = reference.metadata.get("transform")
    src_crs = after.metadata.get("crs")
    dst_crs = reference.metadata.get("crs")
    if not src_transform or not dst_transform or not src_crs or not dst_crs:
        raise ValueError("Geospatial alignment requires CRS and affine transform metadata on both rasters.")

    dst_h, dst_w = reference.original_shape
    channels: list[np.ndarray] = []
    for c in range(after.data.shape[2]):
        destination = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=after.data[:, :, c].astype(np.float32),
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        channels.append(np.clip(destination, 0.0, 1.0))
    data = np.stack(channels, axis=-1)

    gray = np.zeros((dst_h, dst_w), dtype=np.float32)
    reproject(
        source=after.gray.astype(np.float32),
        destination=gray,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    gray = np.clip(gray, 0.0, 1.0)

    valid_float = np.zeros((dst_h, dst_w), dtype=np.float32)
    reproject(
        source=after.valid_mask.astype(np.float32),
        destination=valid_float,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    valid_mask = valid_float > 0.5

    aligned_meta = dict(after.metadata)
    aligned_meta["resampled_to_reference"] = True
    aligned_meta["reference_shape"] = reference.original_shape
    aligned_meta["transform"] = dst_transform
    aligned_meta["crs"] = dst_crs
    aligned_meta["bounds"] = reference.metadata.get("bounds")
    aligned_meta["valid_pixel_fraction"] = round(float(valid_mask.mean()), 6)

    return (
        PreprocessedImage(
            data=data,
            gray=gray,
            valid_mask=valid_mask,
            metadata=aligned_meta,
            original_shape=reference.original_shape,
        ),
        "rasterio_reproject",
    )


def align_images(before: PreprocessedImage, after: PreprocessedImage) -> AlignmentResult:
    """Align the after image onto the exact reference grid of the before image."""
    warnings: list[str] = []
    b_meta = before.metadata
    a_meta = after.metadata
    b_shape = before.original_shape
    a_shape = after.original_shape
    b_geo = bool(b_meta.get("georeferenced"))
    a_geo = bool(a_meta.get("georeferenced"))

    # A georeferenced pair must remain geospatially meaningful.
    if b_geo and a_geo:
        b_crs = b_meta.get("crs")
        a_crs = a_meta.get("crs")
        if b_crs and a_crs and str(b_crs).strip().upper() != str(a_crs).strip().upper():
            return AlignmentResult(
                success=False, before=before, after=after, status="failed", method="none",
                quality_score=0.0,
                error=f"Incompatible CRS: before has '{b_crs}', after has '{a_crs}'. Reprojection between different CRS is required before alignment.",
            )

        b_bounds = b_meta.get("bounds")
        a_bounds = a_meta.get("bounds")
        if b_bounds and a_bounds:
            disjoint_x = b_bounds["right"] < a_bounds["left"] or a_bounds["right"] < b_bounds["left"]
            disjoint_y = b_bounds["top"] < a_bounds["bottom"] or a_bounds["top"] < b_bounds["bottom"]
            if disjoint_x or disjoint_y:
                return AlignmentResult(
                    success=False, before=before, after=after, status="failed", method="none",
                    quality_score=0.0,
                    error="Images have completely disjoint spatial bounds and do not overlap.",
                )

        same_shape = b_shape == a_shape
        same_transform = b_meta.get("transform") == a_meta.get("transform")
        if same_shape and same_transform:
            return AlignmentResult(
                success=True, before=before, after=after, status="already_aligned", method="identity",
                quality_score=1.0, warnings=warnings,
            )

        warnings.append("After raster is being reprojected/resampled onto the before raster's geospatial grid.")
        try:
            aligned_after, method = _geospatial_resample(after, before)
        except Exception as exc:
            return AlignmentResult(
                success=False, before=before, after=after, status="failed", method="none",
                quality_score=0.0,
                error=f"Geospatial alignment failed: {exc}", warnings=warnings,
            )
        return AlignmentResult(
            success=True, before=before, after=aligned_after, status="resampled", method=method,
            quality_score=0.92, warnings=warnings,
        )

    # Mixing georeferenced and non-georeferenced observations cannot be safely
    # aligned in geographic space. Equal pixel grids are still usable, but make
    # the loss of geospatial comparability explicit.
    if b_geo != a_geo:
        warnings.append("Only one image is georeferenced; alignment is performed in pixel space and geographic metadata is not treated as mutually registered.")

    same_shape = b_shape == a_shape
    if same_shape:
        return AlignmentResult(
            success=True, before=before, after=after, status="already_aligned", method="identity",
            quality_score=1.0, warnings=warnings,
        )

    warnings.append(
        f"Resampling after image from {a_shape[1]}x{a_shape[0]} to reference grid {b_shape[1]}x{b_shape[0]}."
    )
    aligned_meta = dict(after.metadata)
    aligned_meta["resampled_to_reference"] = True
    aligned_meta["reference_shape"] = b_shape
    aligned_after = PreprocessedImage(
        data=_resample_array(after.data, b_shape),
        gray=_resample_array(after.gray, b_shape),
        valid_mask=_resample_mask(after.valid_mask, b_shape),
        metadata=aligned_meta,
        original_shape=b_shape,
    )
    return AlignmentResult(
        success=True, before=before, after=aligned_after, status="resampled",
        method="bilinear_grid_resample", quality_score=0.92, warnings=warnings,
    )
