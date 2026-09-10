"""Dependency-light bi-temporal change detector for the local SIH demo.

This is a real image-processing baseline, not a neural-model placeholder:
1. load before/after raster images,
2. normalize them to the same size,
3. compute grayscale absolute difference,
4. threshold the difference map,
5. extract connected changed regions.

Maintains 100% backward-compatible signature and return dictionary.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .mask_processing import extract_change_regions
from .preprocess import load_and_preprocess


def detect_changes(
    before_path: str,
    after_path: str,
    threshold: float = 0.15,
    min_pixels: int = 8,
) -> dict[str, Any]:
    """Run deterministic bi-temporal change detection on two local images."""
    before = load_and_preprocess(before_path, use_percentiles=False)
    after = load_and_preprocess(after_path, use_percentiles=False)

    # Align dimensions if different
    b_gray = before.gray
    a_gray = after.gray

    if b_gray.shape != a_gray.shape:
        from .align import _resample_array
        a_gray = _resample_array(a_gray, b_gray.shape)

    difference = np.abs(a_gray - b_gray)
    raw_mask = difference >= float(threshold)

    # Extract connected regions
    clean_mask, change_regions = extract_change_regions(
        raw_mask,
        difference,
        min_pixels=max(1, int(min_pixels)),
    )

    regions = [r.to_dict() for r in change_regions]

    return {
        "status": "success",
        "model": "pixel-difference-baseline",
        "threshold": threshold,
        "image_size": {"width": int(b_gray.shape[1]), "height": int(b_gray.shape[0])},
        "changed_pixels": int(clean_mask.sum()),
        "changed_fraction": round(float(clean_mask.mean()), 6),
        "regions": regions,
        "mean_difference": round(float(difference.mean()), 6),
    }
