"""SAR radiometric calibration and decibel (dB) transformation utilities."""

import logging
from typing import Optional
import numpy as np

from modules.optical_sar.config import SARData

logger = logging.getLogger(__name__)


def linear_to_db(
    linear_array: np.ndarray,
    min_db: float = -35.0,
    max_db: float = 5.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Convert linear SAR backscatter values to decibels (dB).

    Formula:
        dB = 10 * log10(max(linear_value, epsilon))

    Args:
        linear_array: Input array with linear backscatter values (sigma0 / gamma0).
        min_db: Optional lower clip value in dB.
        max_db: Optional upper clip value in dB.
        epsilon: Floor threshold to prevent log10(0) or negative inputs.

    Returns:
        Float32 array in decibels with NaNs preserved.
    """
    valid = ~np.isnan(linear_array)
    db_arr = np.full_like(linear_array, fill_value=np.nan, dtype=np.float32)

    safe_vals = np.maximum(linear_array[valid], epsilon)
    db_vals = 10.0 * np.log10(safe_vals)

    if min_db is not None or max_db is not None:
        db_vals = np.clip(db_vals, a_min=min_db, a_max=max_db)

    db_arr[valid] = db_vals.astype(np.float32)
    return db_arr


def db_to_linear(
    db_array: np.ndarray,
) -> np.ndarray:
    """Convert decibels (dB) back to linear backscatter values.

    Formula:
        linear = 10^(dB / 10)

    Args:
        db_array: Input array in decibels.

    Returns:
        Float32 array of linear backscatter values with NaNs preserved.
    """
    valid = ~np.isnan(db_array)
    linear_arr = np.full_like(db_array, fill_value=np.nan, dtype=np.float32)
    linear_arr[valid] = np.power(10.0, db_array[valid] / 10.0).astype(np.float32)
    return linear_arr


def calibrate_sar(
    sar_data: SARData,
    is_already_calibrated: bool = True,
    calibration_type: str = "sigma0",
    to_db: bool = True,
    min_db: float = -35.0,
    max_db: float = 5.0,
) -> SARData:
    """Perform radiometric calibration or dB conversion on SAR data.

    Scientific Assumptions & Transparency:
        Sentinel-1 Level-1 GRD products require calibration vectors (sigmaNought, betaNought,
        or gamma) provided in product XML annotation files. When consuming standard pre-processed
        GeoTIFFs (such as those exported from GEE, ASF, or Planetary Computer), the raster values
        are frequently already calibrated to linear sigma0 or are stored as digital numbers with an
        applied scale factor.
        If `is_already_calibrated` is True, this function assumes the input data represents linear
        backscatter values. If False, and no calibration LUT is present, a warning is emitted.

    Args:
        sar_data: Input SARData container.
        is_already_calibrated: Explicit configuration indicating if the data is already
                               calibrated to linear backscatter.
        calibration_type: Target calibration convention ('sigma0', 'gamma0', 'beta0').
        to_db: If True, converts linear backscatter values to decibel (dB) scale.
        min_db: Lower clip threshold in dB.
        max_db: Upper clip threshold in dB.

    Returns:
        Updated SARData container with calibrated / dB data and updated metadata.
    """
    data = sar_data.data.copy()

    if not is_already_calibrated:
        logger.warning(
            "Input SAR data is marked as uncalibrated, but no external calibration LUT was provided. "
            "Downstream processing will treat existing intensities as relative backscatter."
        )
        is_calibrated = False
    else:
        is_calibrated = True

    is_db_result = sar_data.is_db

    if to_db and not sar_data.is_db:
        data = linear_to_db(data, min_db=min_db, max_db=max_db)
        is_db_result = True
        logger.info(f"Converted SAR data to dB scale (range [{min_db}, {max_db}] dB).")
    elif not to_db and sar_data.is_db:
        data = db_to_linear(data)
        is_db_result = False
        logger.info("Converted SAR data from dB scale to linear scale.")

    metadata = dict(sar_data.metadata)
    metadata["calibration_type"] = calibration_type
    metadata["is_already_calibrated"] = is_already_calibrated
    metadata["scale"] = "dB" if is_db_result else "linear"

    return SARData(
        data=data,
        crs=sar_data.crs,
        transform=sar_data.transform,
        resolution=sar_data.resolution,
        bounds=sar_data.bounds,
        nodata=sar_data.nodata,
        band_names=list(sar_data.band_names),
        polarizations=list(sar_data.polarizations),
        metadata=metadata,
        valid_fraction=sar_data.valid_fraction,
        is_calibrated=is_calibrated,
        is_db=is_db_result,
        terrain_corrected=sar_data.terrain_corrected,
    )
