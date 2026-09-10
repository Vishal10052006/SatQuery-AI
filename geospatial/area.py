"""
Geodesic area calculation module for SatQuery-AI M5.
Calculates exact surface areas in square meters and hectares.
"""

from collections.abc import Sequence

from pyproj import CRS, Geod
from shapely.geometry import MultiPolygon, Polygon


def calculate_polygon_area(
    polygon: Polygon | MultiPolygon,
    crs: str = "EPSG:4326",
) -> dict[str, float]:
    """
    Calculate the geodesic surface area of a polygon.

    For EPSG:4326 (WGS84 lat/lon coordinates), exact ellipsoidal geodesic
    area is computed using pyproj.Geod(ellps="WGS84"), avoiding distortion
    at non-equatorial latitudes.

    Args:
        polygon: Shapely Polygon or MultiPolygon.
        crs: CRS of the polygon coordinates (default: EPSG:4326).

    Returns:
        Dictionary with:
            - 'area_sq_meters': Area in square meters (m²)
            - 'area_hectares': Area in hectares (ha, 1 ha = 10,000 m²)
            - 'area_sq_km': Area in square kilometers (km²)
            - 'perimeter_meters': Perimeter in meters
    """
    if polygon is None or polygon.is_empty:
        return {
            "area_sq_meters": 0.0,
            "area_hectares": 0.0,
            "area_sq_km": 0.0,
            "perimeter_meters": 0.0,
        }

    crs_obj = CRS.from_user_input(crs)

    if crs_obj.is_geographic:
        # WGS84 Geodesic calculation on ellipsoid
        geod = Geod(ellps="WGS84")
        # geometry_area_perimeter returns (area, perimeter)
        area, perimeter = geod.geometry_area_perimeter(polygon)
        area_sq_m = abs(float(area))
        perimeter_m = abs(float(perimeter))
    else:
        # For projected CRS in meters (e.g. UTM)
        area_sq_m = abs(float(polygon.area))
        perimeter_m = abs(float(polygon.length))

    area_ha = area_sq_m / 10000.0
    area_sq_km = area_sq_m / 1_000_000.0

    return {
        "area_sq_meters": round(area_sq_m, 2),
        "area_hectares": round(area_ha, 4),
        "area_sq_km": round(area_sq_km, 6),
        "perimeter_meters": round(perimeter_m, 2),
    }


def calculate_total_area(
    polygons: Sequence[Polygon | MultiPolygon],
    crs: str = "EPSG:4326",
) -> dict[str, float]:
    """
    Calculate aggregated area across a list of polygons.

    Args:
        polygons: List of Shapely Polygons/MultiPolygons.
        crs: CRS of the coordinates (default: EPSG:4326).

    Returns:
        Dictionary with total area in sq meters, hectares, and sq km.
    """
    total_sq_m = 0.0
    total_perimeter_m = 0.0

    for poly in polygons:
        res = calculate_polygon_area(poly, crs=crs)
        total_sq_m += res["area_sq_meters"]
        total_perimeter_m += res["perimeter_meters"]

    total_ha = total_sq_m / 10000.0
    total_sq_km = total_sq_m / 1_000_000.0

    return {
        "total_area_sq_meters": round(total_sq_m, 2),
        "total_area_hectares": round(total_ha, 4),
        "total_area_sq_km": round(total_sq_km, 6),
        "total_perimeter_meters": round(total_perimeter_m, 2),
        "polygon_count": len(polygons),
    }
