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
    """Convert non-negative linear SAR backscatter values to decibels.

    Invalid, negative, and non-finite linear values are preserved as NaN rather than
    being silently converted into plausible-looking backscatter values.
    """
    arr = np.asarray(linear_array)
    finite = np.isfinite(arr)
    valid = finite & (arr >= 0.0)
    db_arr = np.full(arr.shape, np.nan, dtype=np.float32)

    if np.any(valid):
        safe_vals = np.maximum(arr[valid], epsilon)
        db_vals = 10.0 * np.log10(safe_vals)
        if min_db is not None or max_db is not None:
            db_vals = np.clip(db_vals, a_min=min_db, a_max=max_db)
        db_arr[valid] = db_vals.astype(np.float32)

    return db_arr


def db_to_linear(db_array: np.ndarray) -> np.ndarray:
    """Convert finite SAR dB values to linear backscatter, preserving invalid values."""
    arr = np.asarray(db_array)
    valid = np.isfinite(arr)
    linear_arr = np.full(arr.shape, np.nan, dtype=np.float32)
    linear_arr[valid] = np.power(10.0, arr[valid] / 10.0).astype(np.float32)
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

    If no calibration LUT is supplied, an input marked uncalibrated is treated only as
    relative backscatter; this function does not fabricate an absolute calibration.
    """
    calibration_type = str(calibration_type).lower()
    if calibration_type not in {"sigma0", "gamma0", "beta0"}:
        raise ValueError("calibration_type must be one of: sigma0, gamma0, beta0")

    if min_db is not None and max_db is not None and min_db >= max_db:
        raise ValueError("min_db must be smaller than max_db")

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
