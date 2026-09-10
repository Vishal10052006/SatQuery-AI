"""
Coordinate transformation module for SatQuery-AI M5.
Handles conversion between pixel coordinates and geographic coordinates (EPSG:4326).
"""

from typing import Any, Dict, List, Sequence, Tuple, Union
from affine import Affine
from pyproj import CRS, Transformer
import rasterio.transform


def _ensure_affine(transform: Union[Affine, Sequence[float]]) -> Affine:
    """Helper to convert flat list/tuple or Affine object to Affine."""
    if isinstance(transform, Affine):
        return transform
    if len(transform) == 9:
        return Affine(
            transform[0], transform[1], transform[2],
            transform[3], transform[4], transform[5]
        )
    elif len(transform) == 6:
        return Affine(*transform)
    raise ValueError(f"Invalid affine transform format: {transform}")


def pixel_to_geo(
    col: float,
    row: float,
    transform: Union[Affine, Sequence[float]],
    crs: Union[str, CRS, Any],
    to_crs: str = "EPSG:4326",
    offset: str = "center",
) -> Tuple[float, float]:
    """
    Convert pixel coordinate (col, row) to geographic coordinate (longitude, latitude).

    Args:
        col: Pixel column (horizontal index, x in pixel space).
        row: Pixel row (vertical index, y in pixel space).
        transform: Affine transform from raster metadata.
        crs: Native CRS of the raster (EPSG code, WKT, or rasterio CRS object).
        to_crs: Target coordinate reference system (default: EPSG:4326).
        offset: Offset within pixel ('center', 'ul', etc.).

    Returns:
        Tuple of (longitude, latitude) in to_crs (EPSG:4326).
    """
    aff = _ensure_affine(transform)
    # Native raster coordinates (x: easting/lon, y: northing/lat in raster CRS)
    native_x, native_y = rasterio.transform.xy(aff, row, col, offset=offset)

    if crs is None:
        # If no CRS provided, return native coordinates directly
        return float(native_x), float(native_y)

    source_crs = CRS.from_user_input(crs)
    target_crs = CRS.from_user_input(to_crs)

    if source_crs == target_crs:
        return float(native_x), float(native_y)

    # always_xy=True ensures (x, y) = (lon, lat) or (easting, northing)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(native_x, native_y)
    return float(lon), float(lat)


def pixel_bbox_to_geo_bbox(
    bbox_pixel: Union[Sequence[float], Dict[str, float]],
    transform: Union[Affine, Sequence[float]],
    crs: Union[str, CRS, Any],
    to_crs: str = "EPSG:4326",
) -> Dict[str, Any]:
    """
    Convert a pixel bounding box to geographic coordinates (EPSG:4326).

    Args:
        bbox_pixel: Pixel bounding box as [xmin, ymin, xmax, ymax] or
                    {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}.
                    xmin/xmax are column indices, ymin/ymax are row indices.
        transform: Affine transform from raster metadata.
        crs: Native CRS of the raster.
        to_crs: Target CRS (default: EPSG:4326).

    Returns:
        Dictionary containing:
            - 'pixel_bbox': [xmin, ymin, xmax, ymax]
            - 'geo_bbox': [min_lon, min_lat, max_lon, max_lat]
            - 'corners_geo': [[lon, lat], ...] for top-left, top-right, bottom-right, bottom-left
            - 'polygon_coords': Closed ring [[lon, lat], ...] suitable for GeoJSON Polygon
            - 'centroid': (centroid_lon, centroid_lat)
    """
    aff = _ensure_affine(transform)

    if isinstance(bbox_pixel, dict):
        col_min = float(bbox_pixel.get("xmin", bbox_pixel.get("col_min", 0)))
        row_min = float(bbox_pixel.get("ymin", bbox_pixel.get("row_min", 0)))
        col_max = float(bbox_pixel.get("xmax", bbox_pixel.get("col_max", 0)))
        row_max = float(bbox_pixel.get("ymax", bbox_pixel.get("row_max", 0)))
    else:
        col_min, row_min, col_max, row_max = [float(v) for v in bbox_pixel]

    # Ensure min <= max
    if col_min > col_max:
        col_min, col_max = col_max, col_min
    if row_min > row_max:
        row_min, row_max = row_max, row_min

    # Four corners in pixel space:
    # 1: Top-Left (col_min, row_min)
    # 2: Top-Right (col_max, row_min)
    # 3: Bottom-Right (col_max, row_max)
    # 4: Bottom-Left (col_min, row_max)
    corners_pixel = [
        (col_min, row_min),
        (col_max, row_min),
        (col_max, row_max),
        (col_min, row_max),
    ]

    corners_geo: List[List[float]] = []
    for c, r in corners_pixel:
        lon, lat = pixel_to_geo(c, r, aff, crs, to_crs=to_crs, offset="ul")
        corners_geo.append([lon, lat])

    # Closed ring for Polygon
    polygon_coords = corners_geo + [corners_geo[0]]

    lons = [pt[0] for pt in corners_geo]
    lats = [pt[1] for pt in corners_geo]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    centroid_lon = (min_lon + max_lon) / 2.0
    centroid_lat = (min_lat + max_lat) / 2.0

    return {
        "pixel_bbox": [col_min, row_min, col_max, row_max],
        "geo_bbox": [min_lon, min_lat, max_lon, max_lat],
        "corners_geo": corners_geo,
        "polygon_coords": polygon_coords,
        "centroid": (centroid_lon, centroid_lat),
    }
