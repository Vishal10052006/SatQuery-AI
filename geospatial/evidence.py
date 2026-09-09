"""
Evidence and GeoJSON generation module for SatQuery-AI M5.
Produces standardized GeoJSON feature collections and evidence.json artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import shapely.geometry
from shapely.geometry import Polygon, MultiPolygon, mapping

from geospatial.area import calculate_polygon_area, calculate_total_area


def generate_geojson(
    polygons: Optional[List[Union[Polygon, MultiPolygon]]] = None,
    bboxes_geo: Optional[List[Dict[str, Any]]] = None,
    target: str = "detected_change",
    confidence: float = 0.85,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate standard GeoJSON FeatureCollection containing polygons and bounding boxes.

    Args:
        polygons: List of Shapely Polygons (in EPSG:4326).
        bboxes_geo: List of bbox dictionaries with 'polygon_coords' and metadata.
        target: Target label string.
        confidence: Detection confidence score.
        output_path: Optional path to write GeoJSON file.

    Returns:
        GeoJSON FeatureCollection dictionary.
    """
    polygons = polygons or []
    bboxes_geo = bboxes_geo or []

    features: List[Dict[str, Any]] = []

    # Add change polygons
    for idx, poly in enumerate(polygons):
        area_info = calculate_polygon_area(poly, crs="EPSG:4326")
        feature = {
            "type": "Feature",
            "id": f"polygon_{idx + 1}",
            "geometry": mapping(poly),
            "properties": {
                "feature_type": "change_polygon",
                "index": idx + 1,
                "target": target,
                "confidence": confidence,
                "area_sq_meters": area_info["area_sq_meters"],
                "area_hectares": area_info["area_hectares"],
                "area_sq_km": area_info["area_sq_km"],
                "perimeter_meters": area_info["perimeter_meters"],
            },
        }
        features.append(feature)

    # Add bounding boxes as Polygon features
    for idx, bbox in enumerate(bboxes_geo):
        polygon_coords = bbox.get("polygon_coords")
        if polygon_coords:
            feature = {
                "type": "Feature",
                "id": f"bbox_{idx + 1}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords],
                },
                "properties": {
                    "feature_type": "bounding_box",
                    "index": idx + 1,
                    "target": target,
                    "pixel_bbox": bbox.get("pixel_bbox"),
                    "geo_bbox": bbox.get("geo_bbox"),
                    "centroid": bbox.get("centroid"),
                },
            }
            features.append(feature)

    geojson_collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(geojson_collection, f, indent=2)

    return geojson_collection


def generate_evidence_json(
    target: str,
    confidence: float,
    change_detected: bool,
    bounding_boxes: List[Dict[str, Any]],
    geographic_coordinates: Dict[str, Any],
    polygons: List[Union[Polygon, MultiPolygon]],
    geojson_path: Union[str, Path],
    map_path: Union[str, Path],
    raster_metadata: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate evidence.json containing detection, geospatial coordinates, areas, and artifact links.

    Args:
        target: Target label (e.g. 'deforestation', 'vehicle', 'urban_expansion').
        confidence: Detection confidence (0.0 to 1.0).
        change_detected: Boolean indicating if change was confirmed.
        bounding_boxes: List of bounding box dictionaries with pixel and geo data.
        geographic_coordinates: Geographic coordinates dictionary (center, corners, bounds).
        polygons: List of Shapely Polygons.
        geojson_path: Path to generated GeoJSON.
        map_path: Path to generated interactive Folium map.
        raster_metadata: Optional GeoTIFF metadata dictionary.
        output_path: Path to write evidence.json.

    Returns:
        Structured dictionary matching the SatQuery-AI M5 evidence specification.
    """
    # Calculate area details
    area_summary = calculate_total_area(polygons, crs="EPSG:4326")
    individual_areas = []
    polygon_geometries = []

    for idx, poly in enumerate(polygons):
        p_area = calculate_polygon_area(poly, crs="EPSG:4326")
        individual_areas.append({
            "id": idx + 1,
            **p_area,
        })
        polygon_geometries.append(mapping(poly))

    area_payload = {
        "total_sq_meters": area_summary["total_area_sq_meters"],
        "total_hectares": area_summary["total_area_hectares"],
        "total_sq_km": area_summary["total_area_sq_km"],
        "polygon_count": len(polygons),
        "individual_polygons": individual_areas,
    }

    # Normalize relative/string paths
    geojson_str = str(Path(geojson_path).as_posix()) if geojson_path else None
    map_str = str(Path(map_path).as_posix()) if map_path else None
    evidence_str = str(Path(output_path).as_posix()) if output_path else None

    evidence: Dict[str, Any] = {
        "target": str(target),
        "confidence": float(confidence),
        "change_detected": bool(change_detected),
        "bounding_boxes": bounding_boxes,
        "geographic_coordinates": geographic_coordinates,
        "polygons": polygon_geometries,
        "area": area_payload,
        "evidence_path": evidence_str,
        "geojson_path": geojson_str,
        "map_path": map_str,
    }

    if raster_metadata:
        meta_dict = dict(raster_metadata)
        if "width" in meta_dict or "height" in meta_dict:
            meta_dict["dimensions"] = {
                "width": meta_dict.get("width"),
                "height": meta_dict.get("height"),
            }
        evidence["raster_metadata"] = meta_dict

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2)

    return evidence
