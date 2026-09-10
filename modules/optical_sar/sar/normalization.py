"""SAR data normalization utilities for linear and dB backscatter scales."""

import logging
from typing import Optional, Tuple
import numpy as np

from modules.optical_sar.config import SARData

logger = logging.getLogger(__name__)


def normalize_sar_array(
    data: np.ndarray,
    method: str = "percentile",
    percentile_bounds: Tuple[float, float] = (1.0, 99.0),
    is_db: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    clip: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalize SAR backscatter array (dB or linear) to [0, 1].

    Args:
        data: SAR array of shape (channels, height, width) or (height, width).
        method: Normalization method ('percentile' or 'minmax').
        percentile_bounds: Tuple of (lower, upper) percentiles.
        is_db: Whether input data is in dB scale.
        min_val: Explicit minimum value (optional).
        max_val: Explicit maximum value (optional).
        clip: Whether to clip output to [0.0, 1.0].
        epsilon: Floor value to prevent zero division.

    Returns:
        Float32 array in [0, 1] range with NaNs preserved.
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

        if method == "percentile":
            p_low = float(np.percentile(valid_vals, percentile_bounds[0])) if min_val is None else min_val
            p_high = float(np.percentile(valid_vals, percentile_bounds[1])) if max_val is None else max_val
        elif method in ("minmax", "min_max"):
            p_low = float(np.min(valid_vals)) if min_val is None else min_val
            p_high = float(np.max(valid_vals)) if max_val is None else max_val
        else:
            raise ValueError(f"Unknown normalization method '{method}'. Choose 'percentile' or 'minmax'.")

        denom = max(p_high - p_low, epsilon)
        norm_channel = (channel - p_low) / denom

        if clip:
            norm_channel = np.clip(norm_channel, 0.0, 1.0)

        norm_channel[~valid_mask] = np.nan
        normalized[c] = norm_channel

    return normalized[0] if is_2d else normalized


def normalize_sar(
    sar_data: SARData,
    method: str = "percentile",
    percentile_bounds: Tuple[float, float] = (1.0, 99.0),
    clip: bool = True,
) -> SARData:
    """Normalize SARData channels to [0, 1] while preserving geospatial metadata.

    Args:
        sar_data: Input SARData.
        method: Normalization method ('percentile' or 'minmax').
        percentile_bounds: Percentile limits (default 1.0 to 99.0).
        clip: Whether to clip output to [0.0, 1.0].

    Returns:
        New SARData container with normalized data and intact geospatial metadata.
    """
    norm_data = normalize_sar_array(
        sar_data.data,
        method=method,
        percentile_bounds=percentile_bounds,
        is_db=sar_data.is_db,
        clip=clip,
    )

    metadata = dict(sar_data.metadata)
    metadata["normalized"] = True
    metadata["normalization_method"] = method

    return SARData(
        data=norm_data,
        crs=sar_data.crs,
        transform=sar_data.transform,
        resolution=sar_data.resolution,
        bounds=sar_data.bounds,
        nodata=sar_data.nodata,
        band_names=list(sar_data.band_names),
        polarizations=list(sar_data.polarizations),
        metadata=metadata,
        valid_fraction=sar_data.valid_fraction,
        is_calibrated=sar_data.is_calibrated,
        is_db=sar_data.is_db,
        terrain_corrected=sar_data.terrain_corrected,
    )
