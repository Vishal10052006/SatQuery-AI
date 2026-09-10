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

from geospatial.evidence import generate_evidence_json, generate_geojson
from geospatial.pipeline import run_geospatial_pipeline
from geospatial.schema import M2M3Payload
from geospatial.visualization import generate_folium_map


def _is_m2_detector_payload(data: dict[str, Any]) -> bool:
    """Check if dictionary matches M2 detector or M4 wrapper output format."""
    return bool(
        (data.get("task") == "change_detection")
        or ("regions" in data or "changed_pixels" in data)
        or (
            "evidence" in data
            and isinstance(data["evidence"], dict)
            and ("regions" in data["evidence"] or "changed_pixels" in data["evidence"])
        )
    )


def _is_m3_payload(data: dict[str, Any]) -> bool:
    """Check if dictionary matches M3 multimodal Optical+SAR pipeline output format."""
    return bool(
        ("gis_evidence" in data and ("optical" in data or "sar" in data or "registration" in data or "fusion" in data))
        or ("optical" in data and "sar" in data and ("prediction" in data or "fusion" in data))
    )


def process_m2_detector_output(
    payload: dict[str, Any] | str | Path,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """
    Process an M2 Change Detection output payload (or M4 SpecialistResult wrapper)
    and produce GIS-compliant GeoJSON, interactive Folium satellite map, and evidence.json.

    Supports:
        1. Georeferenced M2 detector outputs (with native CRS, e.g. EPSG:32643, and polygon_geo)
        2. Non-georeferenced M2 detector outputs (plain PNG/JPG with pixel coordinate fallback)
        3. M4 SpecialistResult wrapper with inner 'evidence' object
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load payload from file or dict
    if isinstance(payload, (str, Path)):
        p_path = Path(payload)
        if p_path.is_file():
            with open(p_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = json.loads(str(payload))
    elif isinstance(payload, dict):
        raw_data = payload
    else:
        raise TypeError(f"Expected dict or path to JSON, got {type(payload).__name__}")

    # 2. Unwrap M4 SpecialistResult wrapper if present
    is_m4_wrapped = False
    m4_claim = None
    if "evidence" in raw_data and isinstance(raw_data["evidence"], dict):
        is_m4_wrapped = True
        m4_claim = raw_data.get("claim")
        detector_body = raw_data["evidence"]
        confidence = float(raw_data.get("confidence", detector_body.get("confidence", 0.75)))
        target = raw_data.get("target") or detector_body.get("target") or "newly constructed buildings"
    else:
        detector_body = raw_data
        confidence = float(detector_body.get("confidence", 0.75))
        target = detector_body.get("target") or "newly constructed buildings"

    change_detected = bool(detector_body.get("change_detected", True))
    georef_available = bool(detector_body.get("geospatial_reference_available", False))
    native_crs = detector_body.get("crs")
    transform = detector_body.get("transform")
    regions = detector_body.get("regions", [])
    changed_area_sq_m = detector_body.get("changed_area_sq_m")

    polygons_4326: list[Polygon] = []
    bboxes_processed: list[dict[str, Any]] = []

    # 3. Process regions based on georeferencing availability
    if georef_available and native_crs:
        # Reproject from native CRS to standard WGS84 (EPSG:4326)
        source_crs = CRS.from_user_input(native_crs)
        target_crs = CRS.from_user_input("EPSG:4326")
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        for reg in regions:
            reg_target = reg.get("target") or target
            reg_confidence = float(reg.get("confidence", confidence))

            # Polygon extraction
            poly = None
            poly_geo = reg.get("polygon_geo")
            if poly_geo and len(poly_geo) >= 3:
                # Reproject points [x, y] in native CRS to [lon, lat] in EPSG:4326
                pts_4326 = [list(transformer.transform(pt[0], pt[1])) for pt in poly_geo]
                poly = Polygon(pts_4326)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polygons_4326.append(poly)

            # Bounding box extraction
            bbox_pixel_dict = reg.get("bbox_pixel")
            if bbox_pixel_dict:
                pixel_bbox = [
                    float(bbox_pixel_dict["xmin"]),
                    float(bbox_pixel_dict["ymin"]),
                    float(bbox_pixel_dict["xmax"]),
                    float(bbox_pixel_dict["ymax"]),
                ]
            else:
                pixel_bbox = []

            bbox_geo_dict = reg.get("bbox_geo")
            if bbox_geo_dict and "top_left" in bbox_geo_dict and "bottom_right" in bbox_geo_dict:
                tl = bbox_geo_dict["top_left"]
                br = bbox_geo_dict["bottom_right"]
                tl_lon, tl_lat = transformer.transform(tl[0], tl[1])
                br_lon, br_lat = transformer.transform(br[0], br[1])
                geo_bbox = [min(tl_lon, br_lon), min(tl_lat, br_lat), max(tl_lon, br_lon), max(tl_lat, br_lat)]
                polygon_coords = [
                    [tl_lon, tl_lat],
                    [br_lon, tl_lat],
                    [br_lon, br_lat],
                    [tl_lon, br_lat],
                    [tl_lon, tl_lat],
                ]
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
            else:
                geo_bbox = []
                polygon_coords = []

            centroid_geo = reg.get("centroid_geo")
            if centroid_geo and "x" in centroid_geo:
                c_lon, c_lat = transformer.transform(centroid_geo["x"], centroid_geo["y"])
                centroid = (round(c_lon, 6), round(c_lat, 6))
            elif polygons_4326:
                centroid = (round(polygons_4326[-1].centroid.x, 6), round(polygons_4326[-1].centroid.y, 6))
            else:
                centroid = (0.0, 0.0)

            bboxes_processed.append({
                "region_id": reg.get("region_id"),
                "target": reg_target,
                "confidence": reg_confidence,
                "pixel_bbox": pixel_bbox,
                "geo_bbox": geo_bbox,
                "polygon_coords": polygon_coords,
                "centroid": centroid,
            })

    else:
        # Non-georeferenced fallback (pixel space)
        for reg in regions:
            reg_target = reg.get("target") or target
            reg_confidence = float(reg.get("confidence", confidence))
            poly_pixel = reg.get("polygon_pixel")
            if poly_pixel and len(poly_pixel) >= 3:
                poly = Polygon(poly_pixel)
                polygons_4326.append(poly)

            bbox_pixel_dict = reg.get("bbox_pixel", {})
            pixel_bbox = [
                float(bbox_pixel_dict.get("xmin", 0)),
                float(bbox_pixel_dict.get("ymin", 0)),
                float(bbox_pixel_dict.get("xmax", 0)),
                float(bbox_pixel_dict.get("ymax", 0)),
            ]
            centroid_pixel = reg.get("centroid_pixel", {})
            c_x = float(centroid_pixel.get("x", (pixel_bbox[0] + pixel_bbox[2]) / 2.0))
            c_y = float(centroid_pixel.get("y", (pixel_bbox[1] + pixel_bbox[3]) / 2.0))

            bboxes_processed.append({
                "region_id": reg.get("region_id"),
                "target": reg_target,
                "confidence": reg_confidence,
                "pixel_bbox": pixel_bbox,
                "geo_bbox": None,
                "polygon_coords": poly_pixel,
                "centroid": (c_x, c_y),
            })

    # 4. Calculate geographic coordinates summary
    all_lons: list[float] = []
    all_lats: list[float] = []
    if georef_available:
        for poly in polygons_4326:
            min_lon, min_lat, max_lon, max_lat = poly.bounds
            all_lons.extend([min_lon, max_lon])
            all_lats.extend([min_lat, max_lat])

    if all_lons and all_lats:
        overall_bounds = [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]
        center_lon = (min(all_lons) + max(all_lons)) / 2.0
        center_lat = (min(all_lats) + max(all_lats)) / 2.0
    else:
        overall_bounds = [0.0, 0.0, 0.0, 0.0]
        center_lon, center_lat = 0.0, 0.0

    geo_coords_summary = {
        "center": [round(center_lat, 6), round(center_lon, 6)] if georef_available else None,
        "overall_bounds_4326": [round(c, 6) for c in overall_bounds] if georef_available else None,
        "polygon_centroids": [
            [round(poly.centroid.y, 6), round(poly.centroid.x, 6)] for poly in polygons_4326
        ] if georef_available else [
            [round(poly.centroid.x, 1), round(poly.centroid.y, 1)] for poly in polygons_4326
        ],
        "bbox_centroids": [
            [round(b["centroid"][1], 6), round(b["centroid"][0], 6)] for b in bboxes_processed
        ] if georef_available else [
            list(b["centroid"]) for b in bboxes_processed
        ],
    }

    # 5. Output file paths
    geojson_path = out_dir / "evidence.geojson"
    map_path = out_dir / "map.html"
    evidence_path = out_dir / "evidence.json"

    # 6. Generate GeoJSON
    generate_geojson(
        polygons=polygons_4326,
        bboxes_geo=bboxes_processed,
        target=target,
        confidence=confidence,
        output_path=geojson_path,
    )

    # 7. Generate interactive Folium map (if georeferenced)
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
        # Generate non-georeferenced placeholder map HTML with quality warning
        warning_msg = (
            detector_body.get("warnings", [
                "Images are not georeferenced; pixel coordinate grid used."
            ])[0] if detector_body.get("warnings") else "Non-georeferenced input"
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

    # 8. Generate evidence.json
    raster_meta = {
        "crs": native_crs,
        "transform": transform,
        "geospatial_reference_available": georef_available,
        "geographic_bbox": detector_body.get("geographic_bbox"),
        "changed_pixels": detector_body.get("changed_pixels"),
        "change_fraction": detector_body.get("change_fraction"),
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
    )

    # Attach quality / detector metadata
    evidence["detector_metadata"] = {
        "detector": detector_body.get("detector"),
        "detector_type": detector_body.get("detector_type"),
        "quality": detector_body.get("quality", {}),
        "warnings": detector_body.get("warnings", []),
        "is_m4_wrapped": is_m4_wrapped,
        "m4_claim": m4_claim,
    }

    # If M2 reported changed_area_sq_m, include it
    if changed_area_sq_m is not None:
        evidence["area"]["m2_reported_area_sq_m"] = changed_area_sq_m

    # Re-save with updated extra metadata
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return evidence


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

    # 1. Load payload from file or dict
    if isinstance(payload, (str, Path)):
        p_path = Path(payload)
        if p_path.is_file():
            with open(p_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            raw_data = json.loads(str(payload))
    elif isinstance(payload, dict):
        raw_data = payload
    else:
        raise TypeError(f"Expected dict or path to JSON, got {type(payload).__name__}")

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
    gis_ev = raw_data.get("gis_evidence", {})
    reg = raw_data.get("registration", {})
    native_crs = gis_ev.get("crs") or reg.get("crs") or "EPSG:32632"
    res_meters = gis_ev.get("resolution_meters") or reg.get("resolution") or [10.0, 10.0]
    pixel_dims = gis_ev.get("pixel_dimensions") or raw_data.get("optical", {}).get("shape", [4, 512, 512])[-2:]

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

    # If payload is an M2 detector output or M4 wrapper, route to process_m2_detector_output
    if _is_m2_detector_payload(data):
        return process_m2_detector_output(data, output_dir=output_dir)

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
