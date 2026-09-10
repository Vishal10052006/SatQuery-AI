"""
SatQuery-AI - Module 5 (M5): GIS, Geospatial Processing and Evidence Generation.

This module provides independent geospatial processing utilities for satellite imagery,
including GeoTIFF metadata extraction, coordinate transformations, mask vectorization,
geodesic area calculations, interactive Folium visualization, and evidence generation.
"""

from geospatial.area import calculate_polygon_area
from geospatial.coordinates import pixel_bbox_to_geo_bbox, pixel_to_geo
from geospatial.evidence import generate_evidence_json, generate_geojson
from geospatial.integration import (
    process_m2_detector_output,
    process_m2_m3_result,
    process_m2_result,
    process_m3_pipeline_payload,
    process_m3_result,
    process_m4_result,
)
from geospatial.m3_adapter import (
    export_all_layers_to_geotiff,
    export_layer_to_geotiff,
    extract_gis_evidence_from_m3,
    process_m3_evidence_to_m5,
)
from geospatial.metadata import read_geotiff_metadata
from geospatial.pipeline import run_geospatial_pipeline
from geospatial.polygons import bbox_to_polygon, mask_to_polygons
from geospatial.schema import (
    EvidenceOutput,
    M2ChangeDetectionResult,
    M2M3Payload,
    M2Region,
    M4SpecialistResult,
)
from geospatial.visualization import generate_folium_map

__all__ = [
    "EvidenceOutput",
    "M2ChangeDetectionResult",
    "M2M3Payload",
    "M2Region",
    "M4SpecialistResult",
    "bbox_to_polygon",
    "calculate_polygon_area",
    "export_all_layers_to_geotiff",
    "export_layer_to_geotiff",
    "extract_gis_evidence_from_m3",
    "generate_evidence_json",
    "generate_folium_map",
    "generate_geojson",
    "mask_to_polygons",
    "pixel_bbox_to_geo_bbox",
    "pixel_to_geo",
    "process_m2_detector_output",
    "process_m2_m3_result",
    "process_m2_result",
    "process_m3_evidence_to_m5",
    "process_m3_pipeline_payload",
    "process_m3_result",
    "process_m4_result",
    "read_geotiff_metadata",
    "run_geospatial_pipeline",
]
