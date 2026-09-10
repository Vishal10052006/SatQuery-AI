"""SAR satellite imagery loader for single or dual polarization GeoTIFF data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import rasterio

from modules.optical_sar.config import SARData

logger = logging.getLogger(__name__)


def load_sar(
    path: Union[str, Path],
    polarizations: Optional[List[str]] = None,
    polarization_mapping: Optional[Dict[str, int]] = None,
    mask_nodata: bool = True,
) -> SARData:
    """Load SAR raster data (e.g., Sentinel-1 GRD GeoTIFF).

    Supports single-polarization (e.g. VV only) and dual-polarization (VV + VH).

    Args:
        path: Path to SAR GeoTIFF file.
        polarizations: List of polarization names for each band, e.g. ["VV", "VH"].
                       If None, defaults to ["VV", "VH"] if 2 bands, or ["VV"] if 1 band.
        polarization_mapping: Optional dictionary mapping polarization name to 1-based band index,
                              e.g. {"VV": 1, "VH": 2}.
        mask_nodata: Whether to mask nodata values with NaN.

    Returns:
        SARData container with shape (channels, height, width) and geospatial metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If CRS is missing or requested bands are out of bounds.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"SAR raster file not found: {file_path}")

    with rasterio.open(file_path) as src:
        # Validate CRS
        if not src.crs:
            raise ValueError(
                f"SAR raster '{file_path.name}' has no valid Coordinate Reference System (CRS)."
            )

        total_bands = src.count
        selected_indices: List[int]
        resolved_polars: List[str]

        if polarization_mapping is not None:
            selected_indices = list(polarization_mapping.values())
            resolved_polars = list(polarization_mapping.keys())
            for name, idx in polarization_mapping.items():
                if idx < 1 or idx > total_bands:
                    raise ValueError(
                        f"Requested band index {idx} for polarization '{name}' is out of range. "
                        f"File contains {total_bands} band(s)."
                    )
        elif polarizations is not None:
            if len(polarizations) > total_bands:
                raise ValueError(
                    f"Requested {len(polarizations)} polarizations, but file only contains "
                    f"{total_bands} band(s)."
                )
            selected_indices = list(range(1, len(polarizations) + 1))
            resolved_polars = list(polarizations)
        else:
            # Automatic heuristic based on band count
            selected_indices = list(range(1, total_bands + 1))
            if total_bands == 1:
                resolved_polars = ["VV"]
            elif total_bands == 2:
                resolved_polars = ["VV", "VH"]
            else:
                resolved_polars = [f"pol_{i}" for i in range(1, total_bands + 1)]

        data_arr = src.read(selected_indices).astype(np.float32)
        nodata_val = src.nodata

        if mask_nodata and nodata_val is not None:
            is_nodata = np.isclose(data_arr, nodata_val) | np.isnan(data_arr)
            data_arr[is_nodata] = np.nan

        # Calculate valid-pixel fraction
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
        metadata["count"] = len(selected_indices)
        metadata["polarizations"] = resolved_polars

        logger.info(
            f"Loaded SAR raster {file_path.name}: shape={data_arr.shape}, "
            f"polars={resolved_polars}, CRS={src.crs}, res={res_tuple}, valid_frac={valid_fraction:.3f}"
        )

        return SARData(
            data=data_arr,
            crs=src.crs,
            transform=src.transform,
            resolution=res_tuple,
            bounds=bounds_tuple,
            nodata=nodata_val,
            band_names=resolved_polars,
            polarizations=resolved_polars,
            metadata=metadata,
            valid_fraction=valid_fraction,
            is_calibrated=False,
            is_db=False,
            terrain_corrected=False,
        )
