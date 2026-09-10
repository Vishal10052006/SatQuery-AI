"""Optical satellite imagery loader for multi-band GeoTIFF data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import rasterio

from modules.optical_sar.config import OpticalData

logger = logging.getLogger(__name__)


def load_optical(
    path: Union[str, Path],
    band_mapping: Optional[Dict[str, int]] = None,
    band_indices: Optional[List[int]] = None,
    band_names: Optional[List[str]] = None,
    mask_nodata: bool = True,
) -> OpticalData:
    """Load optical multi-band raster data (e.g., Sentinel-2 GeoTIFF).

    Args:
        path: Path to the GeoTIFF raster file.
        band_mapping: Optional mapping from band name to 1-based band index,
                      e.g. {"B02": 1, "B03": 2, "B04": 3, "B08": 4}.
        band_indices: Optional list of 1-based band indices to load.
        band_names: Optional list of custom band names corresponding to band_indices.
        mask_nodata: If True, replaces pixels equal to src.nodata with NaN.

    Returns:
        OpticalData container with data of shape (channels, height, width)
        and complete geospatial metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If CRS is missing or invalid band indices are requested.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Optical raster file not found: {file_path}")

    with rasterio.open(file_path) as src:
        # Validate CRS
        if not src.crs:
            raise ValueError(
                f"Optical raster '{file_path.name}' has no valid Coordinate Reference System (CRS)."
            )

        total_bands = src.count
        selected_indices: List[int]
        resolved_names: List[str]

        if band_mapping is not None:
            selected_indices = list(band_mapping.values())
            resolved_names = list(band_mapping.keys())
            for name, idx in band_mapping.items():
                if idx < 1 or idx > total_bands:
                    raise ValueError(
                        f"Requested band index {idx} for '{name}' is out of range. "
                        f"File contains {total_bands} band(s)."
                    )
        elif band_indices is not None:
            selected_indices = band_indices
            for idx in selected_indices:
                if idx < 1 or idx > total_bands:
                    raise ValueError(
                        f"Requested band index {idx} is out of range (1..{total_bands})."
                    )
            if band_names is not None:
                if len(band_names) != len(selected_indices):
                    raise ValueError(
                        f"Length of band_names ({len(band_names)}) does not match "
                        f"band_indices ({len(selected_indices)})."
                    )
                resolved_names = list(band_names)
            else:
                resolved_names = [f"band_{i}" for i in selected_indices]
        else:
            selected_indices = list(range(1, total_bands + 1))
            if band_names is not None and len(band_names) == total_bands:
                resolved_names = list(band_names)
            else:
                resolved_names = [f"band_{i}" for i in selected_indices]

        # Read raster data as float32 for downstream mathematical operations
        data_arr = src.read(selected_indices).astype(np.float32)
        nodata_val = src.nodata

        if mask_nodata and nodata_val is not None:
            is_nodata = np.isclose(data_arr, nodata_val) | np.isnan(data_arr)
            data_arr[is_nodata] = np.nan

        # Calculate initial valid-pixel fraction
        valid_pixels = np.count_nonzero(~np.isnan(data_arr))
        total_pixels = data_arr.size
        valid_fraction = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

        bounds_tuple = (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )
        res_tuple = (float(src.res[0]), float(src.res[1]))

        metadata = dict(src.meta)
        # Update metadata count to reflect selected bands
        metadata["count"] = len(selected_indices)
        metadata["band_names"] = resolved_names

        logger.info(
            f"Loaded optical raster {file_path.name}: shape={data_arr.shape}, "
            f"CRS={src.crs}, res={res_tuple}, valid_frac={valid_fraction:.3f}"
        )

        return OpticalData(
            data=data_arr,
            crs=src.crs,
            transform=src.transform,
            resolution=res_tuple,
            bounds=bounds_tuple,
            nodata=nodata_val,
            band_names=resolved_names,
            metadata=metadata,
            valid_fraction=valid_fraction,
        )
