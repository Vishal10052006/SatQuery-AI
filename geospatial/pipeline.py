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
    """Run the end-to-end M5 geospatial processing pipeline.

    Geographic coordinates are emitted only when the source raster contains
    both a valid CRS and a non-identity geotransform. Ordinary PNG/JPEG files
    remain valid inputs for change detection, but their pixel coordinates are
    never mislabeled as latitude/longitude.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read raster metadata.
    meta = read_geotiff_metadata(reference_geotiff)
    transform = meta.get("transform")
    crs = meta.get("crs")

    # A CRS alone is not enough: an identity transform maps pixels to pixels.
    georeferenced = bool(crs and transform)
    if not georeferenced:
        transform = None
        meta["geospatial_reference_available"] = False
        meta["geospatial_warning"] = (
            "Source raster has no valid CRS/geotransform; pixel coordinates are "
            "not exposed as geographic coordinates."
        )
    else:
        meta["geospatial_reference_available"] = True

    # 2. Convert pixel boxes only when a real georeference exists.
    bboxes_geo: List[Dict[str, Any]] = []
    if georeferenced and bounding_boxes:
        for bbox in bounding_boxes:
            bboxes_geo.append(
                pixel_bbox_to_geo_bbox(
                    bbox,
                    transform=transform,
                    crs=crs,
                    to_crs="EPSG:4326",
                )
            )

    # 3. Convert the change mask to geographic polygons only when georeferenced.
    polygons = []
    if georeferenced and change_mask is not None:
        polygons = mask_to_polygons(
            mask=change_mask,
            transform=transform,
            crs=crs,
            to_crs="EPSG:4326",
        )

    # Change is still represented by the supplied geospatial regions when
    # available. For non-georeferenced input, the image-level detector remains
    # authoritative and M5 simply reports that geographic grounding is absent.
    change_detected = bool(len(polygons) > 0 or len(bboxes_geo) > 0)

    # 4. Build geographic summary only from actual geographic coordinates.
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
        overall_bounds = None
        center_lon = None
        center_lat = None

    geo_coords_summary = {
        "center": [round(center_lat, 6), round(center_lon, 6)]
        if georeferenced and center_lat is not None and center_lon is not None
        else None,
        "overall_bounds_4326": [round(c, 6) for c in overall_bounds]
        if georeferenced and overall_bounds is not None
        else None,
        "polygon_centroids": [
            [round(poly.centroid.y, 6), round(poly.centroid.x, 6)] for poly in polygons
        ] if georeferenced else [],
        "bbox_centroids": [
            [round(b["centroid"][1], 6), round(b["centroid"][0], 6)] for b in bboxes_geo
        ] if georeferenced else [],
    }

    # 5. Output file paths.
    geojson_path = out_dir / "evidence.geojson"
    map_path = out_dir / "map.html"
    evidence_path = out_dir / "evidence.json"

    # 6. Generate GeoJSON only from real EPSG:4326 geometry.
    generate_geojson(
        polygons=polygons,
        bboxes_geo=bboxes_geo,
        target=target,
        confidence=confidence,
        output_path=geojson_path,
    )

    # 7. Generate the interactive map only when geographic grounding exists.
    if georeferenced and (all_lons and all_lats):
        generate_folium_map(
            polygons=polygons,
            bboxes_geo=bboxes_geo,
            center_coords=(center_lat, center_lon),
            target=target,
            confidence=confidence,
            output_html_path=map_path,
        )
    else:
        warning_msg = meta.get(
            "geospatial_warning",
            "No valid geographic reference is available for this raster.",
        )
        map_html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SatQuery-AI — Geospatial Reference Unavailable</title>
<style>
body {{ margin: 0; padding: 32px; font-family: Arial, sans-serif; background: #0f1418; color: #e8eef2; }}
.card {{ max-width: 720px; margin: 40px auto; padding: 28px; border: 1px solid #27323a; border-radius: 14px; background: #151c21; }}
.label {{ color: #63e6df; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }}
h2 {{ margin: 8px 0 14px; }}
p {{ color: #aebbc4; line-height: 1.6; }}
.notice {{ padding: 14px 16px; border-radius: 10px; background: #202a31; border: 1px solid #33414b; }}
code {{ color: #dbe7ed; }}
</style>
</head>
<body>
<div class="card">
<div class="label">SatQuery-AI · M5 GIS</div>
<h2>Geospatial reference unavailable</h2>
<div class="notice">{warning_msg}</div>
<p>The change detector can still operate in image/pixel space. Geographic coordinates and map overlays will appear only when the source imagery supplies valid CRS and geotransform metadata.</p>
<p>GeoJSON: <code>{geojson_path.name}</code></p>
</div>
</body>
</html>"""
        map_path.write_text(map_html_content, encoding="utf-8")

    # 8. Generate evidence.json.
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
