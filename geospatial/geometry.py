"""Deterministic geospatial primitives for model outputs.

Never fabricates coordinates. A real raster transform must be supplied before
pixel geometry can be reported as ground geometry.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple


def pixel_to_geo(x: float, y: float, transform: Tuple[float, float, float, float, float, float]) -> tuple[float, float]:
    """Apply GDAL six-parameter affine transform."""
    a, b, c, d, e, f = transform
    return (round(a * x + b * y + c, 6), round(d * x + e * y + f, 6))


def polygon_area_xy(points: Iterable[tuple[float, float]]) -> float:
    """Calculate planar polygon area with the shoelace formula."""
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    return abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts))) / 2.0)


def calculate_pixel_area_m2(transform: Tuple[float, float, float, float, float, float] | None, crs: str | None) -> float | None:
    """Derive ground area per pixel in square meters only for projected CRS inputs.

    CRS classification is delegated to rasterio/pyproj when available. A
    conservative string fallback is retained for lightweight environments.
    """
    if not transform or not crs:
        return None

    try:
        from pyproj import CRS  # type: ignore
        parsed = CRS.from_user_input(crs)
        if not parsed.is_projected:
            return None
        units = {str(axis.unit_name).lower() for axis in parsed.axis_info if axis.unit_name}
        if units and not any(unit in {"metre", "meter", "metres", "meters"} for unit in units):
            return None
    except (ImportError, Exception):
        # Conservative fallback for environments without pyproj.
        crs_upper = crs.upper().strip()
        geographic_markers = ("4326", "CRS84", "GEOGCS", "DEGREE", "LATITUDE", "LONGITUDE")
        projected_markers = ("UTM", "3857", "326", "327", "PROJCS", "PROJECTED", "METRE", "METER")
        if any(marker in crs_upper for marker in geographic_markers) and not any(marker in crs_upper for marker in projected_markers):
            return None
        if not any(marker in crs_upper for marker in projected_markers):
            return None

    a, b, _, d, e, _ = transform
    det = abs(a * e - b * d)
    if not math.isfinite(det) or det <= 0.0:
        return None
    return round(float(det), 4)


def calculate_ground_area(pixel_count: int, transform: Tuple[float, float, float, float, float, float] | None, crs: str | None) -> float | None:
    """Calculate total ground area in square meters when CRS units are metric."""
    if pixel_count <= 0:
        return 0.0
    px_area = calculate_pixel_area_m2(transform, crs)
    return None if px_area is None else round(float(pixel_count * px_area), 2)
