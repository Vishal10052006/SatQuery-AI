"""SAR speckle filtering algorithms including Lee filter and median filtering."""

import logging
from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter, median_filter

logger = logging.getLogger(__name__)


def lee_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    num_looks: float = 4.0,
    damping_factor: float = 1.0,
) -> np.ndarray:
    """Apply standard Lee speckle filter on a 2D SAR image.

    The Lee filter assumes a multiplicative noise model for SAR intensity:
        I = R * u, where E[u] = 1, Var(u) = sigma_v^2 = 1 / num_looks.

    The filtered reflectivity estimate is:
        R_hat = local_mean + W * (I - local_mean)
        W = max(0, (local_var - local_mean^2 * sigma_v^2) / (local_var * (1 + sigma_v^2)))

    Fast vectorized implementation using scipy sliding uniform filters.

    Args:
        image: 2D numpy array (height, width).
        kernel_size: Window size (must be an odd integer, e.g. 3, 5, 7).
        num_looks: Equivalent Number of Looks (ENL) of the SAR product (default 4 for S1 GRD).
        damping_factor: Weight scaling factor (default 1.0).

    Returns:
        Filtered 2D float32 array with identical shape, NaNs preserved.
    """
    if image.ndim != 2:
        raise ValueError(f"Lee filter expects a 2D array, got shape {image.shape}")

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")

    nan_mask = np.isnan(image)
    # Fill NaNs with 0 for convolution, track valid count
    filled = np.where(nan_mask, 0.0, image).astype(np.float64)
    valid_weights = np.where(nan_mask, 0.0, 1.0)

    # Compute local valid counts
    local_count = uniform_filter(valid_weights, size=kernel_size, mode="reflect")
    min_valid_thresh = 1.0 / (kernel_size * kernel_size)

    # Local mean
    local_sum = uniform_filter(filled, size=kernel_size, mode="reflect")
    local_mean = np.zeros_like(filled)
    valid_region = local_count > min_valid_thresh
    local_mean[valid_region] = local_sum[valid_region] / local_count[valid_region]

    # Local variance: Var(X) = E[X^2] - (E[X])^2
    local_sq_sum = uniform_filter(filled ** 2, size=kernel_size, mode="reflect")
    local_sq_mean = np.zeros_like(filled)
    local_sq_mean[valid_region] = local_sq_sum[valid_region] / local_count[valid_region]
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    # Noise variance
    sigma_v2 = 1.0 / max(num_looks, 1.0)

    # Adaptive weight W
    denom = local_var * (1.0 + sigma_v2) + 1e-8
    numerator = local_var - (local_mean ** 2) * sigma_v2
    w = np.clip(numerator / denom, 0.0, 1.0) * damping_factor

    # Lee estimate
    filtered = local_mean + w * (filled - local_mean)
    filtered[nan_mask] = np.nan

    return filtered.astype(np.float32)


def apply_speckle_filter(
    image: np.ndarray,
    method: str = "lee",
    kernel_size: int = 5,
    **kwargs,
) -> np.ndarray:
    """Apply speckle filter across single or multi-channel SAR raster data.

    Args:
        image: Numpy array of shape (channels, height, width) or (height, width).
        method: Filter algorithm ('lee', 'median', 'none').
        kernel_size: Filter window size (must be odd, e.g. 3, 5, 7).
        **kwargs: Additional parameters passed to the filter function.

    Returns:
        Filtered array matching input dimensions.
    """
    if method == "none":
        return image.copy()

    is_2d = image.ndim == 2
    arr = image[np.newaxis, ...] if is_2d else image.copy()
    filtered_arr = np.empty_like(arr, dtype=np.float32)

    for c in range(arr.shape[0]):
        channel = arr[c]
        if method == "lee":
            num_looks = kwargs.get("num_looks", 4.0)
            filtered_arr[c] = lee_filter(
                channel, kernel_size=kernel_size, num_looks=num_looks
            )
        elif method == "median":
            nan_mask = np.isnan(channel)
            # Median filter with NaN replacement
            safe_chan = np.where(nan_mask, 0.0, channel)
            med = median_filter(safe_chan, size=kernel_size, mode="reflect")
            med[nan_mask] = np.nan
            filtered_arr[c] = med.astype(np.float32)
        else:
            raise ValueError(
                f"Unknown speckle filtering method '{method}'. Supported: 'lee', 'median', 'none'."
            )

    return filtered_arr[0] if is_2d else filtered_arr
