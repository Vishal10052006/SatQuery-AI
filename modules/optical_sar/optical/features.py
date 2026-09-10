"""Optical feature extraction including RGB compositing and spectral indices (NDVI, NDWI)."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from modules.optical_sar.config import OpticalData

logger = logging.getLogger(__name__)


def compute_ndvi(
    nir: np.ndarray,
    red: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index (NDVI).

    Formula:
        NDVI = (NIR - RED) / (NIR + RED + epsilon)

    Values are clipped to [-1.0, 1.0]. Pixels where NIR or RED are NaN remain NaN.

    Args:
        nir: Near-Infrared channel array (e.g. B08).
        red: Red channel array (e.g. B04).
        epsilon: Small constant to prevent division-by-zero in dark/zero-reflectance pixels.

    Returns:
        Float32 array of NDVI values in [-1.0, 1.0].
    """
    valid = (~np.isnan(nir)) & (~np.isnan(red))
    ndvi = np.full_like(nir, fill_value=np.nan, dtype=np.float32)

    numerator = nir[valid] - red[valid]
    denominator = nir[valid] + red[valid] + epsilon

    val = numerator / denominator
    val = np.clip(val, -1.0, 1.0)
    ndvi[valid] = val
    return ndvi


def compute_ndwi(
    green: np.ndarray,
    nir: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute Normalized Difference Water Index (NDWI) - McFeeters 1996.

    Formula:
        NDWI = (GREEN - NIR) / (GREEN + NIR + epsilon)

    Args:
        green: Green channel array (e.g. B03).
        nir: Near-Infrared channel array (e.g. B08).
        epsilon: Small constant to avoid division by zero.

    Returns:
        Float32 array of NDWI values in [-1.0, 1.0].
    """
    valid = (~np.isnan(green)) & (~np.isnan(nir))
    ndwi = np.full_like(green, fill_value=np.nan, dtype=np.float32)

    numerator = green[valid] - nir[valid]
    denominator = green[valid] + nir[valid] + epsilon

    val = numerator / denominator
    val = np.clip(val, -1.0, 1.0)
    ndwi[valid] = val
    return ndwi


def compose_rgb(
    optical_data: OpticalData,
    red_band: str = "B04",
    green_band: str = "B03",
    blue_band: str = "B02",
) -> Optional[np.ndarray]:
    """Compose normalized RGB 3-channel array (H, W, 3) or (3, H, W).

    Args:
        optical_data: OpticalData container.
        red_band: Name of red channel.
        green_band: Name of green channel.
        blue_band: Name of blue channel.

    Returns:
        RGB array of shape (3, H, W) normalized to [0, 1], or None if bands missing.
    """
    needed = [red_band, green_band, blue_band]
    for b in needed:
        if not optical_data.has_band(b):
            logger.warning(f"Cannot compose RGB: missing band '{b}' in {optical_data.band_names}")
            return None

    r = optical_data.get_band(red_band)
    g = optical_data.get_band(green_band)
    b = optical_data.get_band(blue_band)

    # Stack into (3, H, W)
    rgb = np.stack([r, g, b], axis=0).astype(np.float32)
    return rgb


def compute_optical_features(
    optical_data: OpticalData,
    red_band: str = "B04",
    nir_band: str = "B08",
    green_band: str = "B03",
    blue_band: str = "B02",
) -> Dict[str, np.ndarray]:
    """Extract standard optical feature representations.

    Args:
        optical_data: OpticalData container.
        red_band: Red band name (default 'B04').
        nir_band: NIR band name (default 'B08').
        green_band: Green band name (default 'B03').
        blue_band: Blue band name (default 'B02').

    Returns:
        Dictionary containing available features, e.g.:
        {
            "rgb": np.ndarray (3, H, W),
            "ndvi": np.ndarray (H, W),
            "ndwi": np.ndarray (H, W) [optional]
        }
    """
    features: Dict[str, np.ndarray] = {}

    # 1. RGB composition
    rgb = compose_rgb(optical_data, red_band, green_band, blue_band)
    if rgb is not None:
        features["rgb"] = rgb

    # 2. NDVI calculation
    if optical_data.has_band(red_band) and optical_data.has_band(nir_band):
        red_arr = optical_data.get_band(red_band)
        nir_arr = optical_data.get_band(nir_band)
        features["ndvi"] = compute_ndvi(nir_arr, red_arr)
    else:
        logger.warning(
            f"Cannot compute NDVI: required bands '{red_band}' and '{nir_band}' "
            f"not both present in {optical_data.band_names}."
        )

    # 3. NDWI calculation
    if optical_data.has_band(green_band) and optical_data.has_band(nir_band):
        green_arr = optical_data.get_band(green_band)
        nir_arr = optical_data.get_band(nir_band)
        features["ndwi"] = compute_ndwi(green_arr, nir_arr)

    return features
