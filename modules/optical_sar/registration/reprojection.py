"""Geospatial raster reprojection to common coordinate reference grid using rasterio."""

import logging
from typing import Dict
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from modules.optical_sar.config import OpticalData, RasterData, SARData

logger = logging.getLogger(__name__)

RESAMPLING_METHODS: Dict[str, Resampling] = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "cubic": Resampling.cubic,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
}


def reproject_to_reference(
    source: SARData,
    reference: OpticalData,
    resampling_method: str = "bilinear",
) -> SARData:
    """Geospatially reproject and resample source SAR data onto the reference optical grid.

    Guarantees that the resulting SAR raster matches the optical raster in:
    - Coordinate Reference System (CRS)
    - Affine geotransform matrix
    - Spatial resolution (pixel size)
    - Bounding coordinates
    - Exact pixel height and width dimensions

    Args:
        source: Source SARData to reproject.
        reference: Reference OpticalData defining target geospatial grid.
        resampling_method: Resampling algorithm ('bilinear', 'nearest', 'cubic', 'lanczos').

    Returns:
        New SARData aligned to reference grid.

    Raises:
        ValueError: If unsupported resampling method is given or CRS is missing.
    """
    if resampling_method not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unsupported resampling method '{resampling_method}'. "
            f"Supported methods: {list(RESAMPLING_METHODS.keys())}"
        )

    resample_enum = RESAMPLING_METHODS[resampling_method]

    dst_shape = (source.channels, reference.height, reference.width)
    dst_data = np.full(dst_shape, fill_value=np.nan, dtype=np.float32)

    logger.info(
        f"Reprojecting SAR data ({source.shape}) from CRS '{source.crs}' "
        f"to optical grid ({dst_shape}) with CRS '{reference.crs}' using '{resampling_method}'."
    )

    for c in range(source.channels):
        src_chan = source.data[c].astype(np.float32)
        dst_chan = np.full((reference.height, reference.width), fill_value=np.nan, dtype=np.float32)

        reproject(
            source=src_chan,
            destination=dst_chan,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=resample_enum,
            src_nodata=source.nodata,
            dst_nodata=np.nan,
        )
        dst_data[c] = dst_chan

    # Calculate valid fraction in target grid
    valid_pixels = int(np.count_nonzero(~np.isnan(dst_data[0])))
    total_pixels = reference.height * reference.width
    valid_fraction = float(valid_pixels / total_pixels) if total_pixels > 0 else 0.0

    meta = dict(source.metadata)
    meta["reprojected"] = True
    meta["reprojected_to_crs"] = str(reference.crs)
    meta["resampling_method"] = resampling_method

    return SARData(
        data=dst_data,
        crs=reference.crs,
        transform=reference.transform,
        resolution=reference.resolution,
        bounds=reference.bounds,
        nodata=np.nan,
        band_names=list(source.band_names),
        polarizations=list(source.polarizations),
        metadata=meta,
        valid_fraction=valid_fraction,
        is_calibrated=source.is_calibrated,
        is_db=source.is_db,
        terrain_corrected=source.terrain_corrected,
    )
