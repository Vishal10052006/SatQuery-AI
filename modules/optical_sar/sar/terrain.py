"""SAR terrain correction modular interface and adapter."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from modules.optical_sar.config import SARData

logger = logging.getLogger(__name__)


@dataclass
class TerrainCorrectionResult:
    """Structured result of SAR terrain correction operation.

    Attributes:
        sar_data: SARData container (unmodified or terrain-corrected).
        applied: Boolean flag indicating if terrain correction was truly executed.
        method: Method used ('range_doppler_snap', 'dem_illuminated', 'none').
        status: Status code ('completed', 'dem_not_provided', 'external_processor_required').
        dem_path: Path to DEM file if provided, else None.
        notes: Detailed technical explanation of processing applied or bypassed.
    """
    sar_data: SARData
    applied: bool
    method: str
    status: str
    dem_path: Optional[Path]
    notes: str


def apply_terrain_correction(
    sar_data: SARData,
    dem_path: Optional[Union[str, Path]] = None,
    apply_illumination_correction: bool = False,
) -> TerrainCorrectionResult:
    """Apply or adapt terrain correction on SAR backscatter data.

    Scientific Principle:
        Rigorous SAR Range-Doppler Terrain Correction (RDTC) requires:
        1. Precise orbit ephemeris vectors (Sentinel-1 POEORB / RESORB).
        2. Range-Doppler slant-range to ground-range polynomial geometry.
        3. A Digital Elevation Model (DEM, e.g. Copernicus 30m DEM or SRTM 1-arcsec).
        4. Simulated SAR backscatter for radiometric slope / local incidence angle normalization.

        Full RDTC is typically executed upstream using ESA SNAP Graph Processing Tool (GPT)
        or ISCE. When running in Python without SNAP binaries, this module functions as a clean
        modular adapter. If a DEM is supplied and geometric slope normalization is enabled,
        it performs local DEM slope illumination correction. If no DEM is provided, it explicitly
        marks terrain correction as bypassed rather than fabricating corrections.

    Args:
        sar_data: Input SARData container.
        dem_path: Optional path to reference DEM GeoTIFF.
        apply_illumination_correction: If True and DEM is supplied, applies geometric
                                       local incidence angle slope approximation.

    Returns:
        TerrainCorrectionResult with transparent execution status.
    """
    if dem_path is None:
        notes = (
            "No Digital Elevation Model (DEM) provided. Range-Doppler Terrain Correction (RDTC) "
            "was not performed. In mountainous regions, foreshortening, layover, and shadowing "
            "distortions remain uncorrected."
        )
        logger.info(notes)
        return TerrainCorrectionResult(
            sar_data=sar_data,
            applied=False,
            method="none",
            status="dem_not_provided",
            dem_path=None,
            notes=notes,
        )

    dem_file = Path(dem_path)
    if not dem_file.is_file():
        raise FileNotFoundError(f"Specified DEM raster file not found: {dem_file}")

    if not apply_illumination_correction:
        notes = (
            f"DEM provided at {dem_file.name}, but external SNAP / GDAL-RPC pipeline is required "
            f"for full SAR Doppler orthorectification. Flagged for upstream processing."
        )
        logger.info(notes)
        return TerrainCorrectionResult(
            sar_data=sar_data,
            applied=False,
            method="external_adapter",
            status="external_processor_required",
            dem_path=dem_file,
            notes=notes,
        )

    # DEM-based slope illumination correction
    try:
        with rasterio.open(dem_file) as dem_src:
            dem_data = np.empty((sar_data.height, sar_data.width), dtype=np.float32)
            reproject(
                source=rasterio.band(dem_src, 1),
                destination=dem_data,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=sar_data.transform,
                dst_crs=sar_data.crs,
                resampling=Resampling.bilinear,
            )

        # Compute gradient (slope proxy)
        dy, dx = np.gradient(dem_data)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2) / max(sar_data.resolution[0], 1.0))
        # Incidence angle correction factor: cos(slope)
        cos_slope = np.clip(np.cos(slope_rad), 0.1, 1.0)

        corrected_data = sar_data.data.copy()
        for c in range(corrected_data.shape[0]):
            if sar_data.is_db:
                # In dB, division by cos(slope) corresponds to subtracting 10 * log10(cos(slope))
                corrected_data[c] -= 10.0 * np.log10(cos_slope).astype(np.float32)
            else:
                corrected_data[c] /= cos_slope.astype(np.float32)

        updated_sar = SARData(
            data=corrected_data,
            crs=sar_data.crs,
            transform=sar_data.transform,
            resolution=sar_data.resolution,
            bounds=sar_data.bounds,
            nodata=sar_data.nodata,
            band_names=list(sar_data.band_names),
            polarizations=list(sar_data.polarizations),
            metadata=dict(sar_data.metadata),
            valid_fraction=sar_data.valid_fraction,
            is_calibrated=sar_data.is_calibrated,
            is_db=sar_data.is_db,
            terrain_corrected=True,
        )

        notes = (
            f"Applied DEM slope illumination normalization using DEM from {dem_file.name}. "
            f"Geospatial transform and CRS preserved."
        )
        logger.info(notes)
        return TerrainCorrectionResult(
            sar_data=updated_sar,
            applied=True,
            method="dem_slope_illumination",
            status="completed",
            dem_path=dem_file,
            notes=notes,
        )

    except Exception as e:
        logger.error(f"Failed to apply DEM terrain correction: {e}")
        return TerrainCorrectionResult(
            sar_data=sar_data,
            applied=False,
            method="failed",
            status="error",
            dem_path=dem_file,
            notes=f"Error executing terrain correction: {str(e)}",
        )
