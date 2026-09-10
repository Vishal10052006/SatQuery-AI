"""
Integration interface for SatQuery-AI Module 5 (M5).
Provides clean entrypoints for upstream AI modules (M2: Change Detection,
M3: Object Detection / Classification, M4: Agentic Specialist Wrapper)
to invoke the geospatial processing engine.
"""

import json
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from geospatial.area import calculate_polygon_area
from geospatial.coordinates import pixel_bbox_to_geo_bbox, pixel_to_geo
from geospatial.evidence import generate_evidence_json, generate_geojson
from geospatial.pipeline import run_geospatial_pipeline
from geospatial.schema import (
    M2ChangeDetectionResult,
    M2M3Payload,
    M4SpecialistResult,
)
from geospatial.visualization import generate_folium_map


def _is_m4_payload(data: dict[str, Any]) -> bool:
    """Check if dictionary matches M4 SpecialistResult wrapper format."""
    if "mission_result" in data and isinstance(data["mission_result"], dict):
        return True
    if "evidence" in data and isinstance(data["evidence"], dict) and (
        "task" in data or "model" in data or "claim" in data
    ):
        return True
    return False


def _is_m2_detector_payload(data: dict[str, Any]) -> bool:
    """Check if dictionary matches M2 detector or M4 wrapper output format."""
    return bool(
        (data.get("task") == "change_detection")
        or ("regions" in data or "changed_pixels" in data)
        or _is_m4_payload(data)
    )


def _is_m3_payload(data: dict[str, Any]) -> bool:
    """Check if dictionary matches M3 multimodal Optical+SAR pipeline output format."""
    return bool(
        ("gis_evidence" in data and ("optical" in data or "sar" in data or "registration" in data or "fusion" in data))
        or ("optical" in data and "sar" in data and ("prediction" in data or "fusion" in data))
    )


def process_m4_result(
    payload: dict[str, Any] | M4SpecialistResult | str | Path,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """
    Process an M4 SpecialistResult wrapper payload.
    Extracts the inner M2 change detection evidence and passes it to the M5 processing layer.

    Supports:
        - M4 SpecialistResult wrapper with root 'evidence' object
        - M4 mission_result with 'results' list containing change_detection specialist results
        - Direct M2 payload fallback
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse into dictionary
    if isinstance(payload, (str, Path)):
        p_path = Path(payload)
        if p_path.is_file():
            with open(p_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = json.loads(str(payload))
    elif isinstance(payload, M4SpecialistResult):
        raw_data = payload.to_dict()
    elif isinstance(payload, dict):
        raw_data = payload
    else:
        raise TypeError(f"Expected dict or path to JSON, got {type(payload).__name__}")

    # 2. Extract M2 evidence from M4 SpecialistResult
    m4_claim = raw_data.get("claim")
    m4_model = raw_data.get("model")
    m4_status = raw_data.get("status")
    m4_artifacts = raw_data.get("artifacts", [])

    if "mission_result" in raw_data and isinstance(raw_data["mission_result"], dict):
        results = raw_data["mission_result"].get("results", [])
        extracted_ev = None
        for r in results:
            if isinstance(r, dict) and (
                r.get("task") == "change_detection"
                or "regions" in r.get("evidence", {})
                or "changed_pixels" in r.get("evidence", {})
            ):
                extracted_ev = r.get("evidence", r)
                m4_claim = r.get("claim", m4_claim)
                m4_model = r.get("model", m4_model)
                m4_status = r.get("status", m4_status)
                m4_artifacts = r.get("artifacts", m4_artifacts)
                break
        if extracted_ev is None and results and isinstance(results[0], dict):
            extracted_ev = results[0].get("evidence", results[0])
        m2_payload = extracted_ev if extracted_ev else raw_data
    elif "evidence" in raw_data and isinstance(raw_data["evidence"], dict):
        m2_payload = dict(raw_data["evidence"])
        if "target" not in m2_payload and "target" in raw_data:
            m2_payload["target"] = raw_data["target"]
        if "confidence" not in m2_payload and "confidence" in raw_data:
            m2_payload["confidence"] = raw_data["confidence"]
    else:
        m2_payload = raw_data

    # Tag M4 metadata into payload
    m2_payload["_m4_metadata"] = {
        "is_m4_wrapped": True,
        "claim": m4_claim,
        "model": m4_model,
        "status": m4_status,
        "artifacts": m4_artifacts,
    }

    return process_m2_result(m2_payload, output_dir=out_dir)


def process_m2_result(
    payload: dict[str, Any] | M2ChangeDetectionResult | M4SpecialistResult | str | Path,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """
    Process an M2 Change Detection output payload (or M4 SpecialistResult wrapper)
    and produce GIS-compliant GeoJSON, interactive Folium satellite map, and evidence.json.

    Supports:
        1. Georeferenced M2 detector outputs (with dynamic CRS, transform, polygon_geo, bbox_geo)
        2. Non-georeferenced M2 detector outputs (setting geographic fields to null, adding warning)
        3. M4 SpecialistResult wrapper (automatically unwrapped)
        4. File-based legacy M2 inputs (delegates to run_geospatial_pipeline)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse into dictionary
    if isinstance(payload, (str, Path)):
        p_path = Path(payload)
        if p_path.is_file():
            with open(p_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = json.loads(str(payload))
    elif isinstance(payload, M4SpecialistResult):
        return process_m4_result(payload, output_dir=out_dir)
    elif isinstance(payload, M2ChangeDetectionResult):
        raw_data = payload.to_dict()
    elif isinstance(payload, dict):
        raw_data = payload
    else:
        raise TypeError(f"Expected dict or path to JSON, got {type(payload).__name__}")

    # Check if legacy file-based mock (with reference_geotiff/reference_image and no regions)
    if ("reference_geotiff" in raw_data or "reference_image" in raw_data) and "regions" not in raw_data:
        ref_img = raw_data.get("reference_image") or raw_data.get("reference_geotiff")
        mask = raw_data.get("change_mask") or raw_data.get("change_mask_path")
        bboxes = raw_data.get("bounding_boxes") or raw_data.get("bounding_boxes_pixel", [])
        return run_geospatial_pipeline(
            reference_geotiff=str(ref_img),
            change_mask=mask,
            bounding_boxes=bboxes,
            target=raw_data.get("target", "deforestation"),
            confidence=float(raw_data.get("confidence", 0.94)),
            output_dir=out_dir,
        )

    # Check if payload is an M4 SpecialistResult wrapper
    if _is_m4_payload(raw_data) and "_m4_metadata" not in raw_data:
        return process_m4_result(raw_data, output_dir=out_dir)

    # 2. Extract metadata
    m4_info = raw_data.get("_m4_metadata", {})
    is_m4_wrapped = bool(m4_info.get("is_m4_wrapped", "claim" in raw_data or "model" in raw_data))
    m4_claim = m4_info.get("claim") or raw_data.get("claim")

    target = raw_data.get("target") or "newly constructed buildings"
    confidence = float(raw_data.get("confidence", 0.75))
    change_detected = bool(raw_data.get("change_detected", True))
    georef_available = bool(raw_data.get("geospatial_reference_available", False))
    native_crs = raw_data.get("crs")
    transform = raw_data.get("transform")
    regions = raw_data.get("regions", [])
    changed_area_sq_m = raw_data.get("changed_area_sq_m")
    warnings = list(raw_data.get("warnings", []))

    polygons_4326: list[Polygon] = []
    bboxes_processed: list[dict[str, Any]] = []

    # 3. Handle Georeferenced vs Non-Georeferenced outputs
    if georef_available and native_crs:
        # Dynamic CRS transformation to EPSG:4326 (do not assume EPSG:32643)
        source_crs = CRS.from_user_input(native_crs)
        target_crs = CRS.from_user_input("EPSG:4326")
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        for idx, reg in enumerate(regions):
            reg_id = reg.get("region_id", idx + 1)
            reg_target = reg.get("target") or target
            reg_conf = float(reg.get("confidence", confidence))
            pixel_count = reg.get("pixel_count")
            reg_area_sq_m = reg.get("area_sq_m")

            # Extract pixel bbox
            bbox_pixel_dict = reg.get("bbox_pixel")
            if bbox_pixel_dict:
                if isinstance(bbox_pixel_dict, dict):
                    pixel_bbox = [
                        float(bbox_pixel_dict.get("xmin", 0)),
                        float(bbox_pixel_dict.get("ymin", 0)),
                        float(bbox_pixel_dict.get("xmax", 0)),
                        float(bbox_pixel_dict.get("ymax", 0)),
                    ]
                else:
                    pixel_bbox = [float(v) for v in bbox_pixel_dict]
            else:
                pixel_bbox = []

            poly = None
            poly_geo = reg.get("polygon_geo")
            poly_pixel = reg.get("polygon_pixel")

            # Rule 8 & 10: Prefer polygon_geo when provided; validate consistent CRS
            if poly_geo and len(poly_geo) >= 3:
                pts_4326 = [list(transformer.transform(pt[0], pt[1])) for pt in poly_geo]
                poly = Polygon(pts_4326)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            elif poly_pixel and len(poly_pixel) >= 3 and transform:
                # Rule 9: Convert polygon_pixel using transform and CRS
                pts_4326 = []
                for pt in poly_pixel:
                    lon, lat = pixel_to_geo(pt[0], pt[1], transform, native_crs, to_crs="EPSG:4326")
                    pts_4326.append([lon, lat])
                poly = Polygon(pts_4326)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            elif pixel_bbox and transform:
                # Rule 9 fallback: Convert pixel bbox to geographic polygon
                b_info = pixel_bbox_to_geo_bbox(pixel_bbox, transform, native_crs, to_crs="EPSG:4326")
                poly = Polygon(b_info["polygon_coords"])

            if poly is not None:
                polygons_4326.append(poly)

            # Bounding box extraction
            bbox_geo_dict = reg.get("bbox_geo")
            if bbox_geo_dict and "top_left" in bbox_geo_dict and "bottom_right" in bbox_geo_dict:
                tl = bbox_geo_dict["top_left"]
                br = bbox_geo_dict["bottom_right"]
                tr = bbox_geo_dict.get("top_right")
                bl = bbox_geo_dict.get("bottom_left")
                tl_lon, tl_lat = transformer.transform(tl[0], tl[1])
                br_lon, br_lat = transformer.transform(br[0], br[1])
                if tr and bl:
                    tr_lon, tr_lat = transformer.transform(tr[0], tr[1])
                    bl_lon, bl_lat = transformer.transform(bl[0], bl[1])
                    lons = [tl_lon, tr_lon, br_lon, bl_lon]
                    lats = [tl_lat, tr_lat, br_lat, bl_lat]
                    polygon_coords = [
                        [tl_lon, tl_lat],
                        [tr_lon, tr_lat],
                        [br_lon, br_lat],
                        [bl_lon, bl_lat],
                        [tl_lon, tl_lat],
                    ]
                else:
                    lons = [tl_lon, br_lon]
                    lats = [tl_lat, br_lat]
                    polygon_coords = [
                        [tl_lon, tl_lat],
                        [br_lon, tl_lat],
                        [br_lon, br_lat],
                        [tl_lon, br_lat],
                        [tl_lon, tl_lat],
                    ]
                geo_bbox = [min(lons), min(lats), max(lons), max(lats)]
            elif poly is not None:
                min_lon, min_lat, max_lon, max_lat = poly.bounds
                geo_bbox = [min_lon, min_lat, max_lon, max_lat]
                polygon_coords = [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            elif pixel_bbox and transform:
                b_info = pixel_bbox_to_geo_bbox(pixel_bbox, transform, native_crs, to_crs="EPSG:4326")
                geo_bbox = b_info["geo_bbox"]
                polygon_coords = b_info["polygon_coords"]
            else:
                geo_bbox = []
                polygon_coords = []

            # Centroid
            centroid_geo = reg.get("centroid_geo")
            if centroid_geo and "x" in centroid_geo:
                c_lon, c_lat = transformer.transform(centroid_geo["x"], centroid_geo["y"])
                centroid = (round(c_lon, 6), round(c_lat, 6))
            elif poly is not None:
                centroid = (round(poly.centroid.x, 6), round(poly.centroid.y, 6))
            else:
                centroid = (0.0, 0.0)

            # Area calculation & validation
            if poly is not None:
                poly_area = calculate_polygon_area(poly, crs="EPSG:4326")
                area_m2 = poly_area["area_sq_meters"]
                area_ha = poly_area["area_hectares"]
            else:
                area_m2 = reg_area_sq_m
                area_ha = reg_area_sq_m / 10000.0 if reg_area_sq_m else None

            bboxes_processed.append({
                "region_id": reg_id,
                "target": reg_target,
                "confidence": reg_conf,
                "pixel_count": pixel_count,
                "pixel_bbox": pixel_bbox,
                "geo_bbox": geo_bbox,
                "polygon_coords": polygon_coords,
                "centroid": centroid,
                "area_sq_m": area_m2,
                "area_hectares": area_ha,
                "m2_reported_area_sq_m": reg_area_sq_m,
            })

        # Calculate overall bounds & center (EPSG:4326)
        all_lons: list[float] = []
        all_lats: list[float] = []
        for p in polygons_4326:
            min_lon, min_lat, max_lon, max_lat = p.bounds
            all_lons.extend([min_lon, max_lon])
            all_lats.extend([min_lat, max_lat])

        if all_lons and all_lats:
            overall_bounds = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]
            center_lon = (min(all_lons) + max(all_lons)) / 2.0
            center_lat = (min(all_lats) + max(all_lats)) / 2.0
        else:
            overall_bounds = [0.0, 0.0, 0.0, 0.0]
            center_lon, center_lat = 0.0, 0.0

        geo_coords_summary: dict[str, Any] | None = {
            "center": [round(center_lat, 6), round(center_lon, 6)],
            "overall_bounds_4326": [round(c, 6) for c in overall_bounds],
            "polygon_centroids": [
                [round(p.centroid.y, 6), round(p.centroid.x, 6)] for p in polygons_4326
            ],
            "bbox_centroids": [
                [round(b["centroid"][1], 6), round(b["centroid"][0], 6)] for b in bboxes_processed
            ],
        }

    else:
        # CASE 2: Non-georeferenced output
        # Rule 7: Do NOT invent lat/lon, preserve pixel coords, set geo coords & area to null, add warning
        unref_warning = "Images are not georeferenced; geographic coordinates and real-world area are unavailable."
        if unref_warning not in warnings:
            warnings.append(unref_warning)

        geo_coords_summary = None
        all_lons, all_lats = [], []
        center_lon, center_lat = 0.0, 0.0

        for idx, reg in enumerate(regions):
            reg_id = reg.get("region_id", idx + 1)
            reg_target = reg.get("target") or target
            reg_conf = float(reg.get("confidence", confidence))
            pixel_count = reg.get("pixel_count")

            poly_pixel = reg.get("polygon_pixel")
            if poly_pixel and len(poly_pixel) >= 3:
                poly = Polygon(poly_pixel)
                polygons_4326.append(poly)

            bbox_pixel_dict = reg.get("bbox_pixel")
            if bbox_pixel_dict:
                if isinstance(bbox_pixel_dict, dict):
                    pixel_bbox = [
                        float(bbox_pixel_dict.get("xmin", 0)),
                        float(bbox_pixel_dict.get("ymin", 0)),
                        float(bbox_pixel_dict.get("xmax", 0)),
                        float(bbox_pixel_dict.get("ymax", 0)),
                    ]
                else:
                    pixel_bbox = [float(v) for v in bbox_pixel_dict]
            else:
                pixel_bbox = []

            centroid_pixel = reg.get("centroid_pixel")
            if centroid_pixel and "x" in centroid_pixel:
                c_px = (float(centroid_pixel["x"]), float(centroid_pixel["y"]))
            elif pixel_bbox:
                c_px = ((pixel_bbox[0] + pixel_bbox[2]) / 2.0, (pixel_bbox[1] + pixel_bbox[3]) / 2.0)
            else:
                c_px = (0.0, 0.0)

            bboxes_processed.append({
                "region_id": reg_id,
                "target": reg_target,
                "confidence": reg_conf,
                "pixel_count": pixel_count,
                "pixel_bbox": pixel_bbox,
                "centroid_pixel": centroid_pixel,
                "polygon_pixel": poly_pixel,
                "geo_bbox": None,
                "polygon_coords": None,
                "centroid": c_px,  # pixel centroid
                "area_sq_m": None,
                "area_hectares": None,
            })

    # 4. Output file paths
    geojson_path = out_dir / "evidence.geojson"
    map_path = out_dir / "map.html"
    evidence_path = out_dir / "evidence.json"

    # 5. Generate GeoJSON
    generate_geojson(
        polygons=polygons_4326 if georef_available else None,
        bboxes_geo=bboxes_processed,
        target=target,
        confidence=confidence,
        output_path=geojson_path,
        georeferenced=georef_available,
    )

    # 6. Generate Folium map or fallback HTML
    if georef_available and (all_lons and all_lats):
        generate_folium_map(
            polygons=polygons_4326,
            bboxes_geo=bboxes_processed,
            center_coords=(center_lat, center_lon),
            target=target,
            confidence=confidence,
            output_html_path=map_path,
        )
    else:
        warning_msg = (
            warnings[0]
            if warnings
            else "Images are not georeferenced; geographic coordinates and real-world area are unavailable."
        )
        map_html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>M5 Visualization (Non-Georeferenced)</title>
<style>body {{ font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa; }}
.card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 600px; }}
.alert {{ background: #fff3cd; color: #856404; padding: 12px; border-radius: 4px; margin-bottom: 15px; }}
</style></head>
<body><div class="card">
<h2>SatQuery-AI — Non-Georeferenced Detection</h2>
<div class="alert"><b>Notice:</b> {warning_msg}</div>
<p><b>Target:</b> {target}</p>
<p><b>Confidence:</b> {confidence * 100:.1f}%</p>
<p><b>Detected Regions:</b> {len(regions)}</p>
<p>GeoJSON available at <code>{geojson_path.name}</code></p>
</div></body></html>"""
        map_path.write_text(map_html_content, encoding="utf-8")

    # 7. Generate evidence.json
    raster_meta = {
        "crs": native_crs,
        "transform": transform,
        "geospatial_reference_available": georef_available,
        "geographic_bbox": raw_data.get("geographic_bbox"),
        "changed_pixels": raw_data.get("changed_pixels"),
        "change_fraction": raw_data.get("change_fraction"),
    }

    evidence = generate_evidence_json(
        target=target,
        confidence=confidence,
        change_detected=change_detected,
        bounding_boxes=bboxes_processed,
        geographic_coordinates=geo_coords_summary,
        polygons=polygons_4326,
        geojson_path=geojson_path,
        map_path=map_path,
        raster_metadata=raster_meta,
        output_path=evidence_path,
        georeferenced=georef_available,
        warnings=warnings,
    )

    # Attach detector / M4 metadata
    evidence["detector_metadata"] = {
        "detector": raw_data.get("detector"),
        "detector_type": raw_data.get("detector_type"),
        "quality": raw_data.get("quality", {}),
        "warnings": warnings,
        "is_m4_wrapped": is_m4_wrapped,
        "m4_claim": m4_claim,
    }

    if georef_available and changed_area_sq_m is not None and evidence.get("area"):
        evidence["area"]["m2_reported_area_sq_m"] = changed_area_sq_m

    # Re-save with updated extra metadata
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return evidence


# Backwards-compatible alias
process_m2_detector_output = process_m2_result


def process_m3_pipeline_payload(
    payload: dict[str, Any] | str | Path,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """
    Process an M3 Multimodal (Optical + SAR) pipeline output payload and produce
    GIS-compliant GeoJSON, interactive Folium satellite map, and evidence.json.

    Handles:
        1. Native projected CRS (e.g. EPSG:32632) and coordinates
        2. Bounds dict or GeoJSON polygon footprint from gis_evidence
        3. Multimodal target classification (e.g. "Vegetation", "aircraft_hangar")
        4. Registration and fusion validation metadata
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load payload from file, dict, or pipeline result object
    if isinstance(payload, (str, Path)):
        p_path = Path(payload)
        if p_path.is_file():
            with open(p_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = json.loads(str(payload))
    elif isinstance(payload, dict):
        raw_data = dict(payload)
    elif hasattr(payload, "to_dict") and callable(payload.to_dict):
        raw_data = payload.to_dict()
        if hasattr(payload, "to_gis_evidence") and callable(payload.to_gis_evidence):
            gis_ev_obj = payload.to_gis_evidence()
            if "gis_evidence" not in raw_data:
                raw_data["gis_evidence"] = gis_ev_obj
            elif isinstance(raw_data["gis_evidence"], dict):
                raw_data["gis_evidence"].update(gis_ev_obj)
    else:
        raise TypeError(f"Expected dict, M3 result object, or path to JSON, got {type(payload).__name__}")

    # 2. Extract classification and confidence
    pred = raw_data.get("prediction", {})
    target = (
        pred.get("predicted_class_name")
        or pred.get("predicted_class")
        or "multimodal_detection"
    )
    if isinstance(target, int):
        target = pred.get("predicted_class_name") or f"class_{target}"
    target = str(target)

    conf_data = raw_data.get("confidence", {})
    if isinstance(conf_data, dict):
        confidence = float(conf_data.get("score", pred.get("model_confidence", 0.85)))
    else:
        confidence = float(conf_data)

    # 3. Extract GIS metadata
    gis_ev = raw_data.get("gis_evidence") or raw_data.get("metadata", {})
    reg = raw_data.get("registration", {})
    native_crs = gis_ev.get("crs") or reg.get("crs") or "EPSG:32632"
    res_meters = (
        gis_ev.get("resolution_meters")
        or gis_ev.get("resolution")
        or reg.get("resolution")
        or [10.0, 10.0]
    )
    pixel_dims = (
        gis_ev.get("pixel_dimensions")
        or gis_ev.get("spatial_shape")
        or raw_data.get("optical", {}).get("shape", [4, 512, 512])[-2:]
    )

    # 4. Extract Footprint Polygon & Bounds
    poly_geojson = gis_ev.get("polygon_geojson")
    bounds_obj = gis_ev.get("bounds")

    # Reproject from native CRS to standard WGS84 (EPSG:4326)
    source_crs = CRS.from_user_input(native_crs)
    target_crs = CRS.from_user_input("EPSG:4326")
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    polygons_4326: list[Polygon] = []
    if poly_geojson and "coordinates" in poly_geojson:
        coords_raw = poly_geojson["coordinates"][0]
        pts_4326 = [list(transformer.transform(pt[0], pt[1])) for pt in coords_raw]
        poly = Polygon(pts_4326)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons_4326.append(poly)
    elif bounds_obj:
        if isinstance(bounds_obj, dict):
            min_x, min_y = float(bounds_obj["min_x"]), float(bounds_obj["min_y"])
            max_x, max_y = float(bounds_obj["max_x"]), float(bounds_obj["max_y"])
        else:
            min_x, min_y, max_x, max_y = float(bounds_obj[0]), float(bounds_obj[1]), float(bounds_obj[2]), float(bounds_obj[3])
        pts_raw = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ]
        pts_4326 = [list(transformer.transform(pt[0], pt[1])) for pt in pts_raw]
        poly = Polygon(pts_4326)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons_4326.append(poly)

    # Calculate geographic bounds & center
    all_lons: list[float] = []
    all_lats: list[float] = []
    for p in polygons_4326:
        min_lon, min_lat, max_lon, max_lat = p.bounds
        all_lons.extend([min_lon, max_lon])
        all_lats.extend([min_lat, max_lat])

    if all_lons and all_lats:
        overall_bounds = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]
        center_lon = (min(all_lons) + max(all_lons)) / 2.0
        center_lat = (min(all_lats) + max(all_lats)) / 2.0
    else:
        overall_bounds = [0.0, 0.0, 0.0, 0.0]
        center_lat, center_lon = 0.0, 0.0

    geo_coords_summary = {
        "center": [round(center_lat, 6), round(center_lon, 6)],
        "overall_bounds_4326": [round(c, 6) for c in overall_bounds],
        "polygon_centroids": [
            [round(p.centroid.y, 6), round(p.centroid.x, 6)] for p in polygons_4326
        ],
        "bbox_centroids": [[round(center_lat, 6), round(center_lon, 6)]],
    }

    # Synthesize bounding box entry
    h_px, w_px = float(pixel_dims[0]), float(pixel_dims[1])
    bbox_entry = {
        "pixel_bbox": [0.0, 0.0, w_px, h_px],
        "geo_bbox": [round(c, 6) for c in overall_bounds],
        "centroid": [round(center_lon, 6), round(center_lat, 6)],
        "polygon_coords": [
            [overall_bounds[0], overall_bounds[1]],
            [overall_bounds[2], overall_bounds[1]],
            [overall_bounds[2], overall_bounds[3]],
            [overall_bounds[0], overall_bounds[3]],
            [overall_bounds[0], overall_bounds[1]],
        ],
        "target": target,
        "confidence": confidence,
    }

    # 5. Output file paths
    geojson_path = out_dir / "evidence.geojson"
    map_path = out_dir / "map.html"
    evidence_path = out_dir / "evidence.json"

    # 6. Generate GeoJSON
    generate_geojson(
        polygons=polygons_4326,
        bboxes_geo=[bbox_entry],
        target=target,
        confidence=confidence,
        output_path=geojson_path,
    )

    # 7. Generate interactive Folium map
    if all_lons and all_lats:
        generate_folium_map(
            polygons=polygons_4326,
            bboxes_geo=[bbox_entry],
            center_coords=(center_lat, center_lon),
            target=target,
            confidence=confidence,
            output_html_path=map_path,
        )

    # 8. Generate evidence.json
    raster_meta = {
        "crs": native_crs,
        "resolution": res_meters,
        "pixel_dimensions": pixel_dims,
        "geospatial_reference_available": True,
        "bounds": bounds_obj,
    }

    evidence = generate_evidence_json(
        target=target,
        confidence=confidence,
        change_detected=True,
        bounding_boxes=[bbox_entry],
        geographic_coordinates=geo_coords_summary,
        polygons=polygons_4326,
        geojson_path=geojson_path,
        map_path=map_path,
        raster_metadata=raster_meta,
        output_path=evidence_path,
    )

    # Attach M3 Multimodal AI and Fusion Metadata
    evidence["m3_multimodal_metadata"] = {
        "status": raw_data.get("status"),
        "optical": raw_data.get("optical", {}),
        "sar": raw_data.get("sar", {}),
        "registration": reg,
        "fusion": raw_data.get("fusion", {}),
        "prediction": pred,
        "confidence": conf_data,
        "registration_passed": reg.get("passed", False),
        "registration_score": reg.get("registration_score", 0.0),
    }

    # Re-save updated evidence.json
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return evidence


def process_m3_result(
    payload: Any,
    output_dir: str | Path = "output",
    export_layers: bool = True,
) -> dict[str, Any]:
    """
    Ingest a raw M3 Optical+SAR pipeline result (object, dict, or JSON path) and run M5.
    If in-memory raster layers (optical_data, registered_sar_data) are present and export_layers=True,
    multimodal layers are exported to GIS-ready GeoTIFFs.
    """
    if hasattr(payload, "optical_data") and getattr(payload, "optical_data", None) is not None and export_layers:
        from geospatial.m3_adapter import process_m3_result as adapter_process_m3
        return adapter_process_m3(payload, output_dir=output_dir, export_layers=export_layers)
    return process_m3_pipeline_payload(payload, output_dir=output_dir)


def process_m2_m3_result(
    payload: dict[str, Any] | M2M3Payload | str | Path,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """
    Unified entrypoint for processing upstream M2 / M3 / M4 outputs.

    Automatically detects payload format:
        A) M2 Change Detector format (with 'regions', 'crs', 'transform', etc.)
        B) M4 SpecialistResult wrapper (with inner 'evidence' object)
        C) M3 Multimodal (Optical + SAR) analysis output (with 'gis_evidence', 'fusion', etc.)
        D) Standard file-based payload ('reference_image', 'change_mask', 'bounding_boxes')
    """
    # Parse into dict if file or JSON string
    if isinstance(payload, (str, Path)):
        p_path = Path(payload)
        if p_path.is_file():
            with open(p_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(payload))
    elif isinstance(payload, M2M3Payload):
        data = payload.to_dict()
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError(
            f"Unsupported payload type '{type(payload).__name__}'. "
            "Expected dict, M2M3Payload, or path to JSON file."
        )

    # If payload is an M4 wrapper, route to process_m4_result
    if _is_m4_payload(data):
        return process_m4_result(data, output_dir=output_dir)

    # If payload is an M2 detector output, route to process_m2_result
    if _is_m2_detector_payload(data):
        return process_m2_result(data, output_dir=output_dir)

    # If payload is an M3 multimodal pipeline output, route to process_m3_pipeline_payload
    if _is_m3_payload(data):
        return process_m3_pipeline_payload(data, output_dir=output_dir)

    # Otherwise route to standard file-based M5 pipeline
    parsed_payload = M2M3Payload.from_dict(data)
    if not parsed_payload.reference_image:
        raise ValueError(
            "Payload must provide 'reference_image' pointing to the reference GeoTIFF, "
            "or an M2 detector dictionary with 'regions'."
        )

    ref_path = Path(parsed_payload.reference_image)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference GeoTIFF not found at: {ref_path}")

    evidence_dict = run_geospatial_pipeline(
        reference_geotiff=str(ref_path),
        change_mask=parsed_payload.change_mask,
        bounding_boxes=parsed_payload.bounding_boxes,
        target=parsed_payload.target,
        confidence=parsed_payload.confidence,
        output_dir=output_dir,
    )

    return evidence_dict
