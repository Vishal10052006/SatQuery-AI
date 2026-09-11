"""
GeoTIFF metadata extraction module for SatQuery-AI M5.
Reads spatial reference, dimension, resolution, and affine transformation from rasters.
"""

from pathlib import Path
from typing import Any, Dict, Union
import rasterio
from rasterio.crs import CRS


def read_geotiff_metadata(geotiff_path: Union[str, Path]) -> Dict[str, Any]:
    """Read raster metadata without treating pixel coordinates as geography."""
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

        res = (float(src.res[0]), float(src.res[1]))
        dataset_transform = src.transform

        # Rasterio supplies an identity transform for ordinary PNG/JPEG files
        # and for rasters whose pixel grid has not been geographically placed.
        # An identity transform must never be published as latitude/longitude.
        is_identity_transform = (
            dataset_transform.a == 1.0
            and dataset_transform.b == 0.0
            and dataset_transform.c == 0.0
            and dataset_transform.d == 0.0
            and dataset_transform.e == 1.0
            and dataset_transform.f == 0.0
        )

        has_gcps = bool(src.gcps[0])
        has_rpcs = src.rpcs is not None
        georeferenced = bool(crs_obj and not is_identity_transform) or has_gcps or has_rpcs

        transform_coeffs = (
            [float(val) for val in dataset_transform]
            if georeferenced and not is_identity_transform
            else None
        )

        metadata: Dict[str, Any] = {
            "file_path": str(path.resolve()),
            "crs": crs_str if georeferenced else None,
            "crs_epsg": epsg_code if georeferenced else None,
            "crs_wkt": crs_wkt if georeferenced else None,
            "is_geographic": bool(crs_obj.is_geographic) if georeferenced and crs_obj else False,
            "is_projected": bool(crs_obj.is_projected) if georeferenced and crs_obj else False,
            "width": int(src.width),
            "height": int(src.height),
            "bounds": bounds,
            "resolution": res,
            "transform": transform_coeffs,
            "count": int(src.count),
            "dtypes": [str(dt) for dt in src.dtypes],
            "nodata": src.nodata,
            "driver": src.driver,
            "georeferenced": georeferenced,
            "geospatial_reference_available": georeferenced,
            "geospatial_warning": None if georeferenced else "Source raster has no valid CRS/geotransform; pixel coordinates are not geographic coordinates.",
        }

        return metadata
