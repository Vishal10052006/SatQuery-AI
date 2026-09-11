"""Optical cloud masking module supporting external cloud/SCL masks."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import rasterio

from modules.optical_sar.config import OpticalData

logger = logging.getLogger(__name__)


@dataclass
class CloudMaskResult:
    """Structured result from cloud masking operation."""
    optical_data: OpticalData
    cloud_mask: Optional[np.ndarray]
    cloud_fraction: Optional[float]
    valid_fraction: float
    status: str
    method: str


def _same_grid(src, optical_data: OpticalData) -> bool:
    """Return True only when CRS and affine grid are equivalent."""
    if optical_data.crs is None or optical_data.transform is None:
        return False
    if src.crs is None or src.transform is None:
        return False
    if str(src.crs) != str(optical_data.crs):
        return False
    if src.shape != (optical_data.height, optical_data.width):
        return False
    return src.transform.almost_equals(optical_data.transform)


def cloud_mask_optical(
    optical_data: OpticalData,
    cloud_mask: Optional[Union[str, Path, np.ndarray]] = None,
    cloud_values: Optional[list] = None,
) -> CloudMaskResult:
    """Apply cloud masking using an external mask/SCL raster or 2D array.

    Raster masks are always checked against the complete optical grid (CRS, transform,
    and dimensions), not dimensions alone. A mismatched geospatial raster is reprojected
    with nearest-neighbour sampling when both grids are georeferenced.
    """
    height = optical_data.height
    width = optical_data.width
    orig_data = optical_data.data.copy()

    if cloud_mask is None:
        valid_pixels = int(np.count_nonzero(np.isfinite(orig_data[0])))
        total_pixels = height * width
        valid_fraction = float(valid_pixels / total_pixels) if total_pixels else 0.0
        updated_optical = OpticalData(
            data=orig_data, crs=optical_data.crs, transform=optical_data.transform,
            resolution=optical_data.resolution, bounds=optical_data.bounds,
            nodata=optical_data.nodata, band_names=list(optical_data.band_names),
            metadata=dict(optical_data.metadata), cloud_fraction=None,
            valid_fraction=valid_fraction,
        )
        return CloudMaskResult(updated_optical, None, None, valid_fraction, "unavailable", "none")

    if isinstance(cloud_mask, np.ndarray):
        mask_arr = np.asarray(cloud_mask).squeeze()
        if mask_arr.shape != (height, width):
            raise ValueError(
                f"Cloud mask array shape {mask_arr.shape} does not match optical "
                f"spatial dimensions ({height}, {width})."
            )
        method_name = "external_array"
    else:
        mask_path = Path(cloud_mask)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Cloud mask file not found: {mask_path}")
        with rasterio.open(mask_path) as src:
            if _same_grid(src, optical_data):
                mask_arr = src.read(1)
            elif (
                src.crs is not None and src.transform is not None
                and optical_data.crs is not None and optical_data.transform is not None
            ):
                from rasterio.warp import reproject, Resampling
                mask_arr = np.zeros((height, width), dtype=src.dtypes[0])
                reproject(
                    source=rasterio.band(src, 1),
                    destination=mask_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=optical_data.transform,
                    dst_crs=optical_data.crs,
                    resampling=Resampling.nearest,
                )
            else:
                raise ValueError(
                    "Cloud mask raster is not on the optical grid and cannot be safely "
                    "reprojected because CRS/transform metadata is incomplete."
                )
        method_name = "external_raster"

    if mask_arr.dtype == bool:
        bool_cloud_mask = mask_arr.astype(bool)
    elif cloud_values is not None:
        bool_cloud_mask = np.isin(mask_arr, cloud_values)
    else:
        finite_mask = np.isfinite(mask_arr)
        if np.any(finite_mask) and np.nanmax(mask_arr[finite_mask]) <= 11 and np.nanmin(mask_arr[finite_mask]) >= 0:
            bool_cloud_mask = np.isin(mask_arr, [3, 8, 9, 10])
        else:
            bool_cloud_mask = finite_mask & (mask_arr > 0)

    for c in range(orig_data.shape[0]):
        orig_data[c, bool_cloud_mask] = np.nan

    total_pixels = height * width
    cloud_count = int(np.count_nonzero(bool_cloud_mask))
    cloud_fraction = float(cloud_count / total_pixels) if total_pixels else 0.0
    valid_pixels = int(np.count_nonzero(np.isfinite(orig_data[0])))
    valid_fraction = float(valid_pixels / total_pixels) if total_pixels else 0.0

    updated_optical = OpticalData(
        data=orig_data, crs=optical_data.crs, transform=optical_data.transform,
        resolution=optical_data.resolution, bounds=optical_data.bounds,
        nodata=optical_data.nodata, band_names=list(optical_data.band_names),
        metadata=dict(optical_data.metadata), cloud_fraction=cloud_fraction,
        valid_fraction=valid_fraction,
    )
    return CloudMaskResult(
        updated_optical, bool_cloud_mask, cloud_fraction, valid_fraction,
        "applied_external", method_name,
    )
