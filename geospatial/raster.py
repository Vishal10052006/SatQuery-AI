"""Raster inspection helpers with optional GeoTIFF/CRS support.

Rasterio is intentionally optional. The lightweight demo can still process
ordinary PNG/JPEG/TIFF images, while a geospatial environment can expose CRS,
transform, bounds, pixel dimensions, and resolution without changing the M4 contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image


def is_projected_crs(crs: str | None) -> bool:
    """Determine whether a CRS string represents a projected coordinate reference system.

    Returns False for geographic degrees (e.g., EPSG:4326, WGS 84) or None.
    Returns True for projected systems (e.g., UTM, Web Mercator, state planes).
    """
    if not crs:
        return False
    crs_clean = crs.upper().strip()
    # Explicit geographic markers
    if "4326" in crs_clean or "OGC:CRS84" in crs_clean or "DEGREE" in crs_clean or "GEOGCS" in crs_clean:
        if "PROJCS" not in crs_clean and "PROJECTED" not in crs_clean:
            return False
    # Projected markers
    projected_indicators = ["UTM", "3857", "326", "327", "METRE", "METER", "PROJCS", "PROJECTED"]
    return any(indicator in crs_clean for indicator in projected_indicators)


def inspect_raster(path: str) -> dict[str, Any]:
    """Return raster metadata without fabricating geographic coordinates."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)

    # Prefer Rasterio when installed because it preserves GeoTIFF metadata.
    try:
        import rasterio  # type: ignore

        with rasterio.open(source) as dataset:
            transform = tuple(float(value) for value in dataset.transform[:6])
            res_x, res_y = abs(float(dataset.res[0])), abs(float(dataset.res[1]))
            crs_str = str(dataset.crs) if dataset.crs else None
            nodata_val = dataset.nodata
            if nodata_val is not None:
                nodata_val = float(nodata_val)

            projected = is_projected_crs(crs_str)
            return {
                "format": source.suffix.lower().lstrip("."),
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bands": int(dataset.count),
                "dtype": str(dataset.dtypes[0]) if dataset.dtypes else None,
                "crs": crs_str,
                "transform": transform,
                "bounds": {
                    "left": float(dataset.bounds.left),
                    "bottom": float(dataset.bounds.bottom),
                    "right": float(dataset.bounds.right),
                    "top": float(dataset.bounds.top),
                },
                "resolution": (res_x, res_y),
                "nodata": nodata_val,
                "georeferenced": dataset.crs is not None,
                "is_projected": projected,
                "provider": "rasterio",
            }
    except ImportError:
        pass

    # Lightweight fallback for non-geospatial images.
    with Image.open(source) as image:
        return {
            "format": source.suffix.lower().lstrip("."),
            "width": int(image.width),
            "height": int(image.height),
            "bands": len(image.getbands()),
            "dtype": "uint8",
            "crs": None,
            "transform": None,
            "bounds": None,
            "resolution": None,
            "nodata": None,
            "georeferenced": False,
            "is_projected": False,
            "provider": "pillow",
        }


def bbox_pixel_to_geo(
    bbox: dict[str, int],
    transform: tuple[float, float, float, float, float, float],
) -> dict[str, tuple[float, float]]:
    """Convert a pixel bounding box into affine ground coordinates."""
    xmin, ymin = bbox["xmin"], bbox["ymin"]
    xmax, ymax = bbox["xmax"] + 1, bbox["ymax"] + 1
    a, b, c, d, e, f = transform

    def apply(x: float, y: float) -> tuple[float, float]:
        return (round(a * x + b * y + c, 6), round(d * x + e * y + f, 6))

    return {
        "top_left": apply(xmin, ymin),
        "top_right": apply(xmax, ymin),
        "bottom_right": apply(xmax, ymax),
        "bottom_left": apply(xmin, ymax),
    }


def point_pixel_to_geo(
    x: float,
    y: float,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """Convert single pixel point (x, y) to geographic (gx, gy)."""
    a, b, c, d, e, f = transform
    return (round(a * x + b * y + c, 6), round(d * x + e * y + f, 6))


def polygon_pixel_to_geo(
    points: Sequence[tuple[float, float]],
    transform: tuple[float, float, float, float, float, float],
) -> list[tuple[float, float]]:
    """Convert a sequence of pixel coordinate points to geographic coordinates."""
    a, b, c, d, e, f = transform
    return [
        (round(a * px + b * py + c, 6), round(d * px + e * py + f, 6))
        for px, py in points
    ]
