"""Optical data normalization utilities supporting min-max and percentile scaling."""

import logging
from typing import Optional, Tuple, Union
import numpy as np

from modules.optical_sar.config import OpticalData

logger = logging.getLogger(__name__)


def normalize_min_max(
    data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    clip: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalize array values to [0, 1] using min-max scaling per channel.

    Handles NaNs properly and prevents zero-division.

    Args:
        data: Numpy array of shape (channels, height, width) or (height, width).
        min_val: Explicit minimum value. If None, computed from data excluding NaNs.
        max_val: Explicit maximum value. If None, computed from data excluding NaNs.
        clip: Whether to clip output values to [0.0, 1.0].
        epsilon: Small value to prevent division by zero.

    Returns:
        Normalized float32 array in range [0, 1] with NaNs preserved.
    """
    is_2d = data.ndim == 2
    arr = data[np.newaxis, ...] if is_2d else data.copy()
    normalized = np.empty_like(arr, dtype=np.float32)

    for c in range(arr.shape[0]):
        channel = arr[c]
        valid_mask = ~np.isnan(channel)

        if not np.any(valid_mask):
            normalized[c] = np.nan
            continue

        c_min = float(np.nanmin(channel)) if min_val is None else min_val
        c_max = float(np.nanmax(channel)) if max_val is None else max_val

        denom = max(c_max - c_min, epsilon)
        norm_channel = (channel - c_min) / denom

        if clip:
            norm_channel = np.clip(norm_channel, 0.0, 1.0)

        norm_channel[~valid_mask] = np.nan
        normalized[c] = norm_channel

    return normalized[0] if is_2d else normalized


def normalize_percentile(
    data: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    clip: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalize array values to [0, 1] using robust percentile stretching per channel.

    Reduces the impact of outlier pixels (e.g., specular reflections or cloud tops).

    Args:
        data: Numpy array of shape (channels, height, width) or (height, width).
        lower_percentile: Lower percentile (e.g. 1.0 or 2.0).
        upper_percentile: Upper percentile (e.g. 98.0 or 99.0).
        clip: Whether to clip values outside percentiles to [0.0, 1.0].
        epsilon: Small value to avoid division by zero.

    Returns:
        Normalized float32 array in range [0, 1] with NaNs preserved.
    """
    is_2d = data.ndim == 2
    arr = data[np.newaxis, ...] if is_2d else data.copy()
    normalized = np.empty_like(arr, dtype=np.float32)

    for c in range(arr.shape[0]):
        channel = arr[c]
        valid_mask = ~np.isnan(channel)

        if not np.any(valid_mask):
            normalized[c] = np.nan
            continue

        valid_vals = channel[valid_mask]
        p_low = float(np.percentile(valid_vals, lower_percentile))
        p_high = float(np.percentile(valid_vals, upper_percentile))

        denom = max(p_high - p_low, epsilon)
        norm_channel = (channel - p_low) / denom

        if clip:
            norm_channel = np.clip(norm_channel, 0.0, 1.0)

        norm_channel[~valid_mask] = np.nan
        normalized[c] = norm_channel

    return normalized[0] if is_2d else normalized


def normalize_optical(
    optical_data: OpticalData,
    method: str = "percentile",
    percentile_bounds: Tuple[float, float] = (1.0, 99.0),
    clip: bool = True,
) -> OpticalData:
    """Normalize all channels in OpticalData while preserving geospatial metadata.

    Args:
        optical_data: Input OpticalData.
        method: Normalization method ('percentile' or 'minmax').
        percentile_bounds: Tuple of (lower, upper) percentiles if method is 'percentile'.
        clip: Whether to clip values to [0, 1].

    Returns:
        New OpticalData container with normalized data and intact geospatial metadata.
    """
    if method == "percentile":
        norm_arr = normalize_percentile(
            optical_data.data,
            lower_percentile=percentile_bounds[0],
            upper_percentile=percentile_bounds[1],
            clip=clip,
        )
    elif method in ("minmax", "min_max"):
        norm_arr = normalize_min_max(optical_data.data, clip=clip)
    else:
        raise ValueError(
            f"Unsupported normalization method '{method}'. Choose 'percentile' or 'minmax'."
        )

    return OpticalData(
        data=norm_arr,
        crs=optical_data.crs,
        transform=optical_data.transform,
        resolution=optical_data.resolution,
        bounds=optical_data.bounds,
        nodata=optical_data.nodata,
        band_names=list(optical_data.band_names),
        metadata=dict(optical_data.metadata),
        cloud_fraction=optical_data.cloud_fraction,
        valid_fraction=optical_data.valid_fraction,
    )
