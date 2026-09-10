"""Remote-sensing image preprocessing module.

Handles:
- Image loading (GeoTIFF, multi-band, RGB, grayscale)
- Normalization (min-max and robust percentile scaling for 8-bit, 12-bit, 16-bit)
- NoData and NaN/Inf handling with valid pixel masks
- Preservation of original raster metadata
- Preparation of standardized arrays for baseline and neural detectors
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geospatial.raster import inspect_raster


@dataclass
class PreprocessedImage:
    """Standardized representation of a preprocessed remote sensing image."""

    data: np.ndarray           # (H, W, C) float32 normalized in [0, 1]
    gray: np.ndarray           # (H, W) float32 normalized in [0, 1]
    valid_mask: np.ndarray     # (H, W) bool where True means valid pixel data
    metadata: dict[str, Any]
    original_shape: tuple[int, int]  # (height, width)


def normalize_band(
    band: np.ndarray,
    valid_mask: np.ndarray | None = None,
    use_percentiles: bool = True,
) -> np.ndarray:
    """Normalize a 2D array to float32 [0, 1] using robust percentile or min-max scaling."""
    arr = np.asarray(band, dtype=np.float32)

    if valid_mask is not None and np.any(valid_mask):
        valid_vals = arr[valid_mask]
    else:
        valid_mask = np.isfinite(arr)
        valid_vals = arr[valid_mask]

    if valid_vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    if not use_percentiles:
        # Standard radiometric scale preservation for 8-bit and unit-range data
        max_val = float(valid_vals.max()) if valid_vals.size > 0 else 0.0
        if max_val <= 1.0:
            normed = np.clip(arr, 0.0, 1.0)
        elif max_val <= 255.0:
            normed = np.clip(arr / 255.0, 0.0, 1.0)
        else:
            # 16-bit imagery without percentiles
            normed = np.clip(arr / 65535.0, 0.0, 1.0)
        normed[~valid_mask] = 0.0
        return normed

    if use_percentiles and valid_vals.size > 100:
        # Standard remote-sensing 2%-98% stretch for high-dynamic-range data
        low = float(np.percentile(valid_vals, 2))
        high = float(np.percentile(valid_vals, 98))
    else:
        low = float(valid_vals.min())
        high = float(valid_vals.max())

    if high <= low:
        # Constant image or uniform region
        normed = np.zeros_like(arr, dtype=np.float32)
    else:
        normed = (arr - low) / (high - low)
        normed = np.clip(normed, 0.0, 1.0)

    # Clean non-valid pixels
    normed[~valid_mask] = 0.0
    return normed


def load_and_preprocess(
    path: str | Path,
    use_percentiles: bool = True,
) -> PreprocessedImage:
    """Load a satellite/aerial raster and preprocess it deterministically."""
    path_str = str(Path(path).resolve())
    meta = inspect_raster(path_str)
    nodata = meta.get("nodata")

    # Try rasterio first if available
    loaded_via_rasterio = False
    arr: np.ndarray | None = None
    try:
        import rasterio  # type: ignore

        with rasterio.open(path_str) as dataset:
            # (count, height, width)
            raw = dataset.read()
            if raw.ndim == 3:
                # Transpose to (height, width, count)
                arr = np.transpose(raw, (1, 2, 0))
            elif raw.ndim == 2:
                arr = raw[:, :, np.newaxis]
            loaded_via_rasterio = True
    except Exception:
        pass

    # PIL fallback
    if arr is None:
        pil_img = Image.open(path_str)
        raw = np.asarray(pil_img)
        if raw.ndim == 2:
            arr = raw[:, :, np.newaxis]
        elif raw.ndim == 3:
            # Drop alpha channel if 4 channels (RGBA)
            if raw.shape[2] == 4:
                arr = raw[:, :, :3]
            else:
                arr = raw
        else:
            arr = np.zeros((1, 1, 1), dtype=np.float32)

    height, width = arr.shape[:2]
    num_channels = arr.shape[2]

    # Compute valid pixel mask
    valid_mask = np.ones((height, width), dtype=bool)
    for c in range(num_channels):
        channel = arr[:, :, c]
        valid_mask &= np.isfinite(channel)
        if nodata is not None:
            valid_mask &= (channel != nodata)

    # Normalize each channel
    norm_channels: list[np.ndarray] = []
    for c in range(num_channels):
        norm_ch = normalize_band(
            arr[:, :, c],
            valid_mask=valid_mask,
            use_percentiles=use_percentiles,
        )
        norm_channels.append(norm_ch)

    data_norm = np.stack(norm_channels, axis=-1)

    # Compute grayscale representation
    if num_channels == 1:
        gray = data_norm[:, :, 0]
    elif num_channels >= 3:
        # Standard photometric perceptual weighting for first 3 bands (R, G, B)
        gray = (
            0.2989 * data_norm[:, :, 0]
            + 0.5870 * data_norm[:, :, 1]
            + 0.1140 * data_norm[:, :, 2]
        ).astype(np.float32)
    else:
        gray = np.mean(data_norm, axis=-1).astype(np.float32)

    gray = np.clip(gray, 0.0, 1.0)
    gray[~valid_mask] = 0.0

    meta["loaded_via"] = "rasterio" if loaded_via_rasterio else "pillow"
    meta["channels"] = num_channels
    meta["valid_pixel_fraction"] = round(float(valid_mask.mean()), 6)

    return PreprocessedImage(
        data=data_norm,
        gray=gray,
        valid_mask=valid_mask,
        metadata=meta,
        original_shape=(height, width),
    )
