"""Optical processing package for M3 Optical + SAR module."""

from modules.optical_sar.optical.loader import load_optical
from modules.optical_sar.optical.cloud_mask import cloud_mask_optical, CloudMaskResult
from modules.optical_sar.optical.normalization import (
    normalize_min_max,
    normalize_percentile,
    normalize_optical,
)
from modules.optical_sar.optical.features import compute_optical_features

__all__ = [
    "load_optical",
    "cloud_mask_optical",
    "CloudMaskResult",
    "normalize_min_max",
    "normalize_percentile",
    "normalize_optical",
    "compute_optical_features",
]
