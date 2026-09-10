"""
GeoTIFF metadata extraction module for SatQuery-AI M5.
Reads spatial reference, dimension, resolution, and affine transformation from rasters.
"""

from pathlib import Path
from typing import Any, Dict, Union
import rasterio
from rasterio.crs import CRS


def read_geotiff_metadata(geotiff_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read metadata from a GeoTIFF raster file using rasterio.

    Extracts:
        - CRS (Coordinate Reference System representation & EPSG code if available)
        - width & height in pixels
        - bounds (left, bottom, right, top)
        - resolution (x_resolution, y_resolution)
        - transform (affine transformation matrix)
        - number of bands (count)
        - dtypes and nodata values

    Args:
        geotiff_path: Path to the GeoTIFF file.

    Returns:
        Dictionary containing metadata properties.

    Raises:
        FileNotFoundError: If the GeoTIFF file does not exist.
        rasterio.errors.RasterioIOError: If the file is not a valid raster.
    """
    path = Path(geotiff_path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF file not found at: {path}")

    with rasterio.open(path) as src:
        crs_obj: CRS = src.crs
        crs_str = crs_obj.to_string() if crs_obj else None
        epsg_code = crs_obj.to_epsg() if crs_obj else None
        crs_wkt = crs_obj.to_wkt() if crs_obj else None

        bounds = {
            "left": float(src.bounds.left),
            "bottom": float(src.bounds.bottom),
            "right": float(src.bounds.right),
            "top": float(src.bounds.top),
        }

        # Resolution: (x_res, y_res)
        res = (float(src.res[0]), float(src.res[1]))

        # Transform coefficients as flat tuple / list
        transform_coeffs = [float(val) for val in src.transform]

        metadata: Dict[str, Any] = {
            "file_path": str(path.resolve()),
            "crs": crs_str,
            "crs_epsg": epsg_code,
            "crs_wkt": crs_wkt,
            "is_geographic": crs_obj.is_geographic if crs_obj else False,
            "is_projected": crs_obj.is_projected if crs_obj else False,
            "width": int(src.width),
            "height": int(src.height),
            "bounds": bounds,
            "resolution": res,
            "transform": transform_coeffs,
            "count": int(src.count),
            "dtypes": [str(dt) for dt in src.dtypes],
            "nodata": src.nodata,
            "driver": src.driver,
        }

        return metadata
