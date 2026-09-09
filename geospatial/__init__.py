"""
SatQuery-AI - Module 5 (M5): GIS, Geospatial Processing and Evidence Generation.

This module provides independent geospatial processing utilities for satellite imagery,
including GeoTIFF metadata extraction, coordinate transformations, mask vectorization,
geodesic area calculations, interactive Folium visualization, and evidence generation.
"""

from geospatial.metadata import read_geotiff_metadata
from geospatial.coordinates import pixel_to_geo, pixel_bbox_to_geo_bbox
from geospatial.polygons import mask_to_polygons, bbox_to_polygon
from geospatial.area import calculate_polygon_area
from geospatial.visualization import generate_folium_map
from geospatial.evidence import generate_geojson, generate_evidence_json
from geospatial.pipeline import run_geospatial_pipeline

__all__ = [
    "read_geotiff_metadata",
    "pixel_to_geo",
    "pixel_bbox_to_geo_bbox",
    "mask_to_polygons",
    "bbox_to_polygon",
    "calculate_polygon_area",
    "generate_folium_map",
    "generate_geojson",
    "generate_evidence_json",
    "run_geospatial_pipeline",
]
