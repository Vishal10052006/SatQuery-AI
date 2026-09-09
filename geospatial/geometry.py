"""Deterministic geospatial primitives for model outputs.

Never fabricates coordinates. A real raster transform must be supplied before
pixel geometry can be reported as ground geometry.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple


def pixel_to_geo(
    x: float,
    y: float,
    transform: Tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """Apply GDAL six-parameter affine transform (no external dependency)."""
    a, b, c, d, e, f = transform
    return (round(a * x + b * y + c, 6), round(d * x + e * y + f, 6))


def polygon_area_xy(points: Iterable[tuple[float, float]]) -> float:
    """Calculate planar polygon area with the shoelace formula."""
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    return abs(
        sum(
            pts[i][0] * pts[(i + 1) % len(pts)][1]
            - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts))
        )
        / 2.0
    )


def calculate_pixel_area_m2(
    transform: Tuple[float, float, float, float, float, float] | None,
    crs: str | None,
) -> float | None:
    """Derive ground area per pixel in square meters if the CRS is projected.

    Returns None when the coordinate system is geographic degrees (e.g. EPSG:4326),
    unknown, or when no georeferencing transform is available.
    """
    if not transform or not crs:
        return None

    crs_upper = crs.upper().strip()
    # Check if geographic (degrees)
    if "4326" in crs_upper or "CRS84" in crs_upper or "DEGREE" in crs_upper:
        if "PROJCS" not in crs_upper and "UTM" not in crs_upper:
            return None

    # Projected linear coordinates: determinant of the 2x2 linear transform part
    a, b, _, d, e, _ = transform
    det = abs(a * e - b * d)
    if det <= 0.0 or math.isnan(det):
        return None
    return round(float(det), 4)


def calculate_ground_area(
    pixel_count: int,
    transform: Tuple[float, float, float, float, float, float] | None,
    crs: str | None,
) -> float | None:
    """Calculate total ground area in square meters for a given pixel count.

    Returns None if area cannot be reliably calculated in square meters.
    """
    if pixel_count <= 0:
        return 0.0

    px_area = calculate_pixel_area_m2(transform, crs)
    if px_area is None:
        return None
    return round(float(pixel_count * px_area), 2)
