"""SAR speckle filtering algorithms including Lee filter and median filtering."""

import logging
import numpy as np
from scipy.ndimage import uniform_filter, median_filter

logger = logging.getLogger(__name__)


def lee_filter(image: np.ndarray, kernel_size: int = 5, num_looks: float = 4.0, damping_factor: float = 1.0) -> np.ndarray:
    """Apply a Lee speckle filter to a 2D SAR intensity image.

    Lee filtering assumes multiplicative noise, so callers must provide linear intensity.
    Invalid/non-finite pixels remain NaN.
    """
    if image.ndim != 2:
        raise ValueError(f"Lee filter expects a 2D array, got shape {image.shape}")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")
    if num_looks <= 0:
        raise ValueError(f"num_looks must be > 0, got {num_looks}")

    invalid = ~np.isfinite(image) | (image < 0)
    filled = np.where(invalid, 0.0, image).astype(np.float64)
    valid_weights = (~invalid).astype(np.float64)

    local_count = uniform_filter(valid_weights, size=kernel_size, mode="reflect")
    valid_region = local_count > 0.0
    local_sum = uniform_filter(filled, size=kernel_size, mode="reflect")
    local_mean = np.zeros_like(filled)
    local_mean[valid_region] = local_sum[valid_region] / local_count[valid_region]

    local_sq_sum = uniform_filter(filled ** 2, size=kernel_size, mode="reflect")
    local_sq_mean = np.zeros_like(filled)
    local_sq_mean[valid_region] = local_sq_sum[valid_region] / local_count[valid_region]
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0.0)

    sigma_v2 = 1.0 / num_looks
    denom = local_var * (1.0 + sigma_v2) + 1e-8
    numerator = local_var - (local_mean ** 2) * sigma_v2
    weight = np.clip(numerator / denom, 0.0, 1.0) * damping_factor
    weight = np.clip(weight, 0.0, 1.0)

    filtered = local_mean + weight * (filled - local_mean)
    filtered[invalid] = np.nan
    return filtered.astype(np.float32)


def apply_speckle_filter(
    image: np.ndarray,
    method: str = "lee",
    kernel_size: int = 5,
    is_db: bool = False,
    **kwargs,
) -> np.ndarray:
    """Apply speckle filtering while respecting SAR scale.

    For dB input, the Lee/median operation is performed in linear intensity and the
    result is converted back to dB. This prevents applying a multiplicative-noise
    filter directly in logarithmic space.
    """
    if method == "none":
        return image.copy()

    is_2d = image.ndim == 2
    if image.ndim not in (2, 3):
        raise ValueError(f"SAR image must be 2D or 3D, got shape {image.shape}")

    arr = image[np.newaxis, ...] if is_2d else np.asarray(image).copy()
    filtered_arr = np.full(arr.shape, np.nan, dtype=np.float32)

    for c in range(arr.shape[0]):
        channel = np.asarray(arr[c], dtype=np.float32)
        invalid = ~np.isfinite(channel)

        if is_db:
            linear = np.full(channel.shape, np.nan, dtype=np.float32)
            valid = ~invalid
            linear[valid] = np.power(10.0, channel[valid] / 10.0).astype(np.float32)
            working = linear
        else:
            working = channel

        if method == "lee":
            filtered = lee_filter(working, kernel_size=kernel_size, num_looks=kwargs.get("num_looks", 4.0))
        elif method == "median":
            bad = ~np.isfinite(working)
            safe = np.where(bad, 0.0, working)
            med = median_filter(safe, size=kernel_size, mode="reflect")
            filtered = med.astype(np.float32)
            filtered[bad] = np.nan
        else:
            raise ValueError(f"Unknown speckle filtering method '{method}'. Supported: 'lee', 'median', 'none'.")

        if is_db:
            valid = np.isfinite(filtered) & (filtered > 0.0)
            result = np.full(channel.shape, np.nan, dtype=np.float32)
            result[valid] = 10.0 * np.log10(filtered[valid])
            filtered_arr[c] = result
        else:
            filtered_arr[c] = filtered

    return filtered_arr[0] if is_2d else filtered_arr
