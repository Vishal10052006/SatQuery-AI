"""
Evidence and GeoJSON generation module for SatQuery-AI M5.
Produces standardized GeoJSON feature collections and evidence.json artifacts.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon, mapping

from geospatial.area import calculate_polygon_area, calculate_total_area


def generate_geojson(
    polygons: Sequence[Polygon | MultiPolygon] | None = None,
    bboxes_geo: list[dict[str, Any]] | None = None,
    target: str = "detected_change",
    confidence: float = 0.85,
    output_path: str | Path | None = None,
    georeferenced: bool = True,
) -> dict[str, Any]:
    """
    Generate standard GeoJSON FeatureCollection containing polygons and bounding boxes.

    Args:
        polygons: List of Shapely Polygons (in EPSG:4326).
        bboxes_geo: List of bbox dictionaries with 'polygon_coords' and metadata.
        target: Target label string.
        confidence: Detection confidence score.
        output_path: Optional path to write GeoJSON file.
        georeferenced: Whether output has valid geospatial reference.

    Returns:
        GeoJSON FeatureCollection dictionary.
    """
    polygons = polygons or []
    bboxes_geo = bboxes_geo or []

    features: list[dict[str, Any]] = []

    if not georeferenced:
        # Non-georeferenced: do not invent lon/lat; features have null geometry with pixel coordinates
        for idx, bbox in enumerate(bboxes_geo):
            reg_id = bbox.get("region_id", idx + 1)
            reg_target = bbox.get("target") or target
            reg_conf = float(bbox.get("confidence", confidence))
            feature = {
                "type": "Feature",
                "id": f"region_{reg_id}",
                "geometry": None,
                "properties": {
                    "feature_type": "detected_region",
                    "region_id": reg_id,
                    "target": reg_target,
                    "confidence": reg_conf,
                    "pixel_count": bbox.get("pixel_count"),
                    "pixel_bbox": bbox.get("pixel_bbox"),
                    "centroid_pixel": bbox.get("centroid_pixel"),
                    "polygon_pixel": bbox.get("polygon_pixel"),
                    "area_sq_meters": None,
                    "area_hectares": None,
                    "warning": "Images are not georeferenced; geographic coordinates and real-world area are unavailable.",
                },
            }
            features.append(feature)
    else:
        # Add change polygons (EPSG:4326)
        for idx, poly in enumerate(polygons):
            area_info = calculate_polygon_area(poly, crs="EPSG:4326")
            bbox_info = bboxes_geo[idx] if idx < len(bboxes_geo) else {}
            reg_id = bbox_info.get("region_id", idx + 1)
            reg_target = bbox_info.get("target") or target
            reg_conf = float(bbox_info.get("confidence", confidence))

            feature_props: dict[str, Any] = {
                "feature_type": "change_polygon",
                "index": idx + 1,
                "region_id": reg_id,
                "target": reg_target,
                "confidence": reg_conf,
                "area_sq_meters": area_info["area_sq_meters"],
                "area_hectares": area_info["area_hectares"],
                "area_sq_km": area_info["area_sq_km"],
                "perimeter_meters": area_info["perimeter_meters"],
            }
            if bbox_info.get("pixel_count") is not None:
                feature_props["pixel_count"] = bbox_info["pixel_count"]
            if bbox_info.get("m2_reported_area_sq_m") is not None:
                feature_props["m2_reported_area_sq_m"] = bbox_info["m2_reported_area_sq_m"]

            feature = {
                "type": "Feature",
                "id": f"polygon_{reg_id}",
                "geometry": mapping(poly),
                "properties": feature_props,
            }
            features.append(feature)

        # Add bounding boxes as Polygon features (EPSG:4326)
        for idx, bbox in enumerate(bboxes_geo):
            polygon_coords = bbox.get("polygon_coords")
            if polygon_coords:
                reg_id = bbox.get("region_id", idx + 1)
                reg_target = bbox.get("target") or target
                reg_conf = float(bbox.get("confidence", confidence))
                feature = {
                    "type": "Feature",
                    "id": f"bbox_{reg_id}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coords],
                    },
                    "properties": {
                        "feature_type": "bounding_box",
                        "index": idx + 1,
                        "region_id": reg_id,
                        "target": reg_target,
                        "confidence": reg_conf,
                        "pixel_count": bbox.get("pixel_count"),
                        "pixel_bbox": bbox.get("pixel_bbox"),
                        "geo_bbox": bbox.get("geo_bbox"),
                        "centroid": bbox.get("centroid"),
                    },
                }
                features.append(feature)

    geojson_collection: dict[str, Any] = {
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
    bounding_boxes: list[dict[str, Any]],
    geographic_coordinates: dict[str, Any] | None,
    polygons: Sequence[Polygon | MultiPolygon],
    geojson_path: str | Path,
    map_path: str | Path,
    raster_metadata: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    area: dict[str, Any] | None = None,
    georeferenced: bool = True,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
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
        area: Optional precalculated area dictionary.
        georeferenced: Whether output has valid geospatial reference.
        warnings: Optional list of warnings.

    Returns:
        Structured dictionary matching the SatQuery-AI M5 evidence specification.
    """
    warn_list = list(warnings or [])
    if not georeferenced:
        msg = "Images are not georeferenced; geographic coordinates and real-world area are unavailable."
        if msg not in warn_list:
            warn_list.append(msg)
        area_payload = None
        geo_coords_summary = None
        polygon_geometries = [mapping(p) for p in polygons]
    else:
        if area is not None:
            area_payload = area
            polygon_geometries = [mapping(p) for p in polygons]
        else:
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
        geo_coords_summary = geographic_coordinates

    # Normalize relative/string paths
    geojson_str = str(Path(geojson_path).as_posix()) if geojson_path else None
    map_str = str(Path(map_path).as_posix()) if map_path else None
    evidence_str = str(Path(output_path).as_posix()) if output_path else None

    evidence: dict[str, Any] = {
        "target": str(target),
        "confidence": float(confidence),
        "change_detected": bool(change_detected),
        "bounding_boxes": bounding_boxes,
        "geographic_coordinates": geo_coords_summary,
        "polygons": polygon_geometries,
        "area": area_payload,
        "evidence_path": evidence_str,
        "geojson_path": geojson_str,
        "map_path": map_str,
        "warnings": warn_list,
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
