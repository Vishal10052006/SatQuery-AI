"""
Main pipeline orchestrator for SatQuery-AI M5.
Coordinates GeoTIFF metadata reading, coordinate transformations, polygon extraction,
geodesic area computation, map visualization, and evidence generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from geospatial.metadata import read_geotiff_metadata
from geospatial.coordinates import pixel_bbox_to_geo_bbox, pixel_to_geo
from geospatial.polygons import mask_to_polygons
from geospatial.area import calculate_total_area
from geospatial.visualization import generate_folium_map
from geospatial.evidence import generate_geojson, generate_evidence_json


def run_geospatial_pipeline(
    reference_geotiff: Union[str, Path],
    change_mask: Optional[Union[str, Path, np.ndarray]] = None,
    bounding_boxes: Optional[List[Union[List[float], Dict[str, float]]]] = None,
    target: str = "detected_change",
    confidence: float = 0.85,
    output_dir: Union[str, Path] = "output",
) -> Dict[str, Any]:
    """
    Run the end-to-end M5 geospatial processing pipeline.

    Args:
        reference_geotiff: Path to the reference GeoTIFF raster.
        change_mask: Optional change mask (2D array, .npy file, or raster file).
        bounding_boxes: Optional list of pixel bounding boxes [xmin, ymin, xmax, ymax].
        target: Detection target category (e.g. 'deforestation', 'vehicle', 'aircraft').
        confidence: Detection confidence score (0.0 to 1.0).
        output_dir: Directory where GeoJSON, HTML map, and evidence.json will be saved.

    Returns:
        Structured dictionary containing pipeline results, file paths, and evidence summary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read GeoTIFF metadata (extracts dynamic CRS, transform, resolution, etc.)
    meta = read_geotiff_metadata(reference_geotiff)
    transform = meta["transform"]
    crs = meta["crs"]

    # 2. Process pixel bounding boxes into geographic coordinates
    bboxes_geo: List[Dict[str, Any]] = []
    if bounding_boxes:
        for bbox in bounding_boxes:
            b_info = pixel_bbox_to_geo_bbox(bbox, transform=transform, crs=crs, to_crs="EPSG:4326")
            bboxes_geo.append(b_info)

    # 3. Process change mask into EPSG:4326 polygons
    polygons = []
    if change_mask is not None:
        polygons = mask_to_polygons(
            mask=change_mask,
            transform=transform,
            crs=crs,
            to_crs="EPSG:4326",
        )

    # Determine change detected flag
    change_detected = bool(len(polygons) > 0 or len(bboxes_geo) > 0)

    # 4. Calculate geographic coordinates summary (center, bounds, centroids)
    all_lons: List[float] = []
    all_lats: List[float] = []

    for poly in polygons:
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        all_lons.extend([min_lon, max_lon])
        all_lats.extend([min_lat, max_lat])

    for b in bboxes_geo:
        g = b.get("geo_bbox", [])
        if len(g) == 4:
            all_lons.extend([g[0], g[2]])
            all_lats.extend([g[1], g[3]])

    if all_lons and all_lats:
        overall_bounds = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]
        center_lon = (min(all_lons) + max(all_lons)) / 2.0
        center_lat = (min(all_lats) + max(all_lats)) / 2.0
    else:
        # Fallback to raster center
        w, h = meta["width"], meta["height"]
        center_lon, center_lat = pixel_to_geo(w / 2.0, h / 2.0, transform, crs, to_crs="EPSG:4326")
        overall_bounds = [center_lon, center_lat, center_lon, center_lat]

    geo_coords_summary = {
        "center": [round(center_lat, 6), round(center_lon, 6)],
        "overall_bounds_4326": [round(c, 6) for c in overall_bounds],
        "polygon_centroids": [
            [round(poly.centroid.y, 6), round(poly.centroid.x, 6)] for poly in polygons
        ],
        "bbox_centroids": [
            [round(b["centroid"][1], 6), round(b["centroid"][0], 6)] for b in bboxes_geo
        ],
    }

    # 5. Output file paths
    geojson_path = out_dir / "evidence.geojson"
    map_path = out_dir / "map.html"
    evidence_path = out_dir / "evidence.json"

    # 6. Generate GeoJSON
    generate_geojson(
        polygons=polygons,
        bboxes_geo=bboxes_geo,
        target=target,
        confidence=confidence,
        output_path=geojson_path,
    )

    # 7. Generate interactive Folium map
    generate_folium_map(
        polygons=polygons,
        bboxes_geo=bboxes_geo,
        center_coords=(center_lat, center_lon),
        target=target,
        confidence=confidence,
        output_html_path=map_path,
    )

    # 8. Generate evidence.json
    evidence = generate_evidence_json(
        target=target,
        confidence=confidence,
        change_detected=change_detected,
        bounding_boxes=bboxes_geo,
        geographic_coordinates=geo_coords_summary,
        polygons=polygons,
        geojson_path=geojson_path,
        map_path=map_path,
        raster_metadata=meta,
        output_path=evidence_path,
    )

    return evidence
