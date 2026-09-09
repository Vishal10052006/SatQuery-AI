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
    """Structured result from cloud masking operation.
    
    Attributes:
        optical_data: OpticalData with cloud-affected pixels masked with NaN.
        cloud_mask: 2D boolean mask where True indicates cloud/shadow pixel,
                    or None if cloud masking was unavailable.
        cloud_fraction: Fraction of pixels identified as cloudy (0.0 to 1.0),
                        or None if cloud masking was unavailable.
        valid_fraction: Fraction of valid (non-cloud, non-nodata) pixels.
        status: Status descriptor ('applied_external', 'applied_array', 'unavailable').
        method: Method used for masking ('external_raster', 'external_array', 'none').
    """
    optical_data: OpticalData
    cloud_mask: Optional[np.ndarray]
    cloud_fraction: Optional[float]
    valid_fraction: float
    status: str
    method: str


def cloud_mask_optical(
    optical_data: OpticalData,
    cloud_mask: Optional[Union[str, Path, np.ndarray]] = None,
    cloud_values: Optional[list] = None,
) -> CloudMaskResult:
    """Apply cloud masking to optical imagery using external masks or SCL data.

    Scientific Principle:
        Accurate cloud and cloud-shadow detection requires dedicated spectral bands
        (such as cirrus band B10, SWIR) or precomputed scene classification layers (SCL).
        When no external cloud mask or SCL layer is provided, this function explicitly
        marks cloud masking as 'unavailable' rather than fabricating a clear-sky claim.

    Args:
        optical_data: The input OpticalData object.
        cloud_mask: Optional path to cloud mask GeoTIFF (e.g. SCL or QA60 band)
                    or 2D boolean/integer numpy array of shape (height, width).
        cloud_values: List of integer values in the mask to treat as cloudy/shadowed.
                      For Sentinel-2 SCL, typical cloud/shadow values are:
                      [3 (cloud shadow), 8 (cloud medium prob), 9 (cloud high prob), 10 (thin cirrus)].

    Returns:
        CloudMaskResult containing the masked OpticalData, mask array,
        cloud fraction, valid fraction, and processing status.
    """
    height = optical_data.height
    width = optical_data.width
    orig_data = optical_data.data.copy()

    # Case 1: No external cloud mask provided
    if cloud_mask is None:
        logger.warning(
            "No cloud mask provided for optical data. Marking cloud masking as unavailable."
        )
        # Calculate valid fraction based solely on existing non-NaN / non-nodata pixels
        valid_pixels = np.count_nonzero(~np.isnan(orig_data[0]))
        total_pixels = height * width
        valid_fraction = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

        updated_optical = OpticalData(
            data=orig_data,
            crs=optical_data.crs,
            transform=optical_data.transform,
            resolution=optical_data.resolution,
            bounds=optical_data.bounds,
            nodata=optical_data.nodata,
            band_names=list(optical_data.band_names),
            metadata=dict(optical_data.metadata),
            cloud_fraction=None,
            valid_fraction=valid_fraction,
        )

        return CloudMaskResult(
            optical_data=updated_optical,
            cloud_mask=None,
            cloud_fraction=None,
            valid_fraction=valid_fraction,
            status="unavailable",
            method="none",
        )

    # Case 2: External numpy array provided
    if isinstance(cloud_mask, np.ndarray):
        mask_arr = cloud_mask.squeeze()
        if mask_arr.shape != (height, width):
            raise ValueError(
                f"Cloud mask array shape {mask_arr.shape} does not match optical "
                f"spatial dimensions ({height}, {width})."
            )
        method_name = "external_array"
    # Case 3: External file path provided
    else:
        mask_path = Path(cloud_mask)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Cloud mask file not found: {mask_path}")
        with rasterio.open(mask_path) as src:
            if src.shape != (height, width):
                if optical_data.crs is not None and optical_data.transform is not None:
                    from rasterio.warp import reproject, Resampling
                    mask_arr = np.empty((height, width), dtype=src.dtypes[0])
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
                        f"Cloud mask raster dimensions {src.shape} do not match "
                        f"optical dimensions ({height}, {width}). Reprojection required."
                    )
            else:
                mask_arr = src.read(1)
        method_name = "external_raster"

    # Convert to boolean cloud mask (True = cloudy / shadowed)
    if cloud_values is not None:
        bool_cloud_mask = np.isin(mask_arr, cloud_values)
    else:
        # If boolean array already, use directly; otherwise non-zero is cloudy
        if mask_arr.dtype == bool:
            bool_cloud_mask = mask_arr
        else:
            # Common Sentinel-2 SCL defaults: 3: cloud shadow, 8: medium prob, 9: high prob, 10: cirrus
            # If values span 0-11, treat as SCL; otherwise non-zero
            if np.max(mask_arr) <= 11 and np.min(mask_arr) >= 0:
                bool_cloud_mask = np.isin(mask_arr, [3, 8, 9, 10])
            else:
                bool_cloud_mask = mask_arr > 0

    # Apply mask: set cloudy pixels across all channels to NaN
    for c in range(orig_data.shape[0]):
        orig_data[c, bool_cloud_mask] = np.nan

    total_pixels = height * width
    cloud_count = int(np.count_nonzero(bool_cloud_mask))
    cloud_fraction = float(cloud_count / total_pixels) if total_pixels > 0 else 0.0

    valid_pixels = int(np.count_nonzero(~np.isnan(orig_data[0])))
    valid_fraction = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    updated_optical = OpticalData(
        data=orig_data,
        crs=optical_data.crs,
        transform=optical_data.transform,
        resolution=optical_data.resolution,
        bounds=optical_data.bounds,
        nodata=optical_data.nodata,
        band_names=list(optical_data.band_names),
        metadata=dict(optical_data.metadata),
        cloud_fraction=cloud_fraction,
        valid_fraction=valid_fraction,
    )

    logger.info(
        f"Applied cloud masking ({method_name}): cloud_fraction={cloud_fraction:.3f}, "
        f"valid_fraction={valid_fraction:.3f}"
    )

    return CloudMaskResult(
        optical_data=updated_optical,
        cloud_mask=bool_cloud_mask,
        cloud_fraction=cloud_fraction,
        valid_fraction=valid_fraction,
        status="applied_external",
        method=method_name,
    )
