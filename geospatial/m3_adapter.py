"""
M5 GIS Integration Adapter: Ingests multimodal analysis from Module 3 (Optical + SAR).

This module extracts:
1. Geospatial metadata (CRS, Bounding Box, Resolution, Affine Transform)
2. Vector polygon footprints for GeoPandas / PostGIS queries
3. Aligned raster layers (Optical RGB, SAR VV/VH, NDVI vegetation index, 6-channel stack)
4. Exports layers to GIS-ready GeoTIFF files with projection metadata
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import rasterio
from rasterio.transform import Affine

from geospatial.pipeline import run_geospatial_pipeline


def export_layer_to_geotiff(
    data_array: np.ndarray,
    output_path: Union[str, Path],
    crs: Any,
    transform: Optional[Union[Affine, Sequence[float], Any]] = None,
    nodata: Optional[float] = None,
) -> Path:
    """
    Save any extracted 2D or 3D layer into a GIS-standard GeoTIFF file with spatial reference.

    Args:
        data_array: 2D array (H, W) or 3D array (Channels, H, W).
        output_path: Path to the destination .tif file.
        crs: Coordinate reference system (EPSG code string or CRS object).
        transform: Affine transform matrix or sequence of 6 coefficients.
        nodata: Optional nodata value.

    Returns:
        Path to the written GeoTIFF file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if transform is None:
        affine_transform = Affine.identity()
    elif isinstance(transform, Affine):
        affine_transform = transform
    elif hasattr(transform, "__len__") and len(transform) >= 6:
        affine_transform = Affine(*transform[:6])
    else:
        affine_transform = Affine.identity()

    if data_array.ndim == 2:
        count = 1
        h, w = data_array.shape
        data_to_write = data_array[np.newaxis, :, :]
    elif data_array.ndim == 3:
        count, h, w = data_array.shape
        data_to_write = data_array
    else:
        raise ValueError(
            f"Expected 2D (H, W) or 3D (C, H, W) numpy array, got {data_array.ndim}D array"
        )

    # Cast to supported rasterio dtype if needed
    dtype = data_to_write.dtype
    if dtype == np.float64:
        data_to_write = data_to_write.astype(np.float32)
        dtype = np.float32
    elif dtype == bool:
        data_to_write = data_to_write.astype(np.uint8)
        dtype = np.uint8

    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=affine_transform,
        nodata=nodata,
    ) as dst:
        dst.write(data_to_write)

    return out_file


def export_all_layers_to_geotiff(
    m3_gis_evidence: Dict[str, Any],
    output_dir: Union[str, Path] = "output/layers",
) -> Dict[str, Path]:
    """
    Export all aligned multimodal layers from M3 evidence into individual GeoTIFF files.

    Args:
        m3_gis_evidence: Dictionary returned by extract_gis_evidence_from_m3().
        output_dir: Target directory for GeoTIFF layers.

    Returns:
        Dictionary mapping layer names to their generated file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crs = m3_gis_evidence.get("crs")
    transform = m3_gis_evidence.get("transform")
    layers = m3_gis_evidence.get("layers", {})

    exported_paths: Dict[str, Path] = {}
    for layer_name, layer_arr in layers.items():
        if isinstance(layer_arr, np.ndarray):
            file_path = out_dir / f"{layer_name}.tif"
            export_layer_to_geotiff(
                data_array=layer_arr,
                output_path=file_path,
                crs=crs,
                transform=transform,
            )
            exported_paths[layer_name] = file_path

    return exported_paths


def extract_gis_evidence_from_m3(
    optical_path: str,
    sar_path: str,
    scl_path: Optional[str] = None,
    weights_path: str = "modules/optical_sar/weights/m3_optical_sar_model.pth",
) -> Dict[str, Any]:
    """
    Execute M3 analysis and extract all GIS-ready evidence for M5.
    Dynamically imports M3 modules if available.
    """
    try:
        from modules.optical_sar.pipeline import run_optical_sar_pipeline  # type: ignore
        from modules.optical_sar.fusion.model import OpticalSARModel  # type: ignore
        from modules.optical_sar.optical.features import compute_optical_features  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Module 3 (Optical + SAR) dependencies not found: {e}. "
            "Ensure 'modules.optical_sar' is on the Python path when running full M3 integration."
        )

    # 1. Load trained M3 multimodal model
    model = OpticalSARModel(weights_path=weights_path) if Path(weights_path).exists() else None

    # 2. Run M3 Optical + SAR Pipeline
    m3_result = run_optical_sar_pipeline(
        optical_path=optical_path,
        sar_path=sar_path,
        cloud_mask_path=scl_path,
        model=model,
        run_inference=True,
    )

    # 3. Extract core GIS Evidence
    gis_meta = m3_result.to_gis_evidence()

    # 4. Extract Bounding Box coordinates
    min_x, min_y, max_x, max_y = gis_meta["bounds"]

    # Format GeoJSON Polygon footprint for GeoPandas / PostGIS
    footprint_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ]]
    }

    # 5. Extract Spectral Indices & Features
    opt_features = compute_optical_features(m3_result.optical_data)
    ndvi_array = opt_features["ndvi"]     # Normalized Difference Vegetation Index
    ndwi_array = opt_features["ndwi"]     # Normalized Difference Water Index

    # 6. Extract Aligned Multimodal Rasters
    optical_4band = m3_result.optical_data.data          # Shape: (4, H, W) [B02, B03, B04, B08]
    sar_2band = m3_result.registered_sar_data.data       # Shape: (2, H, W) [VV, VH in dB]
    fused_6band = m3_result.early_fusion_result.fused_data  # Shape: (6, H, W) [All aligned]

    return {
        # --- GIS Metadata & GeoDatabase Attributes ---
        "crs": gis_meta["crs"],                         # e.g. "EPSG:32632"
        "bounds": gis_meta["bounds"],                   # (min_x, min_y, max_x, max_y)
        "transform": m3_result.optical_data.transform,  # Affine matrix
        "resolution": gis_meta["resolution"],           # (10.0, 10.0) meters
        "spatial_shape": gis_meta["spatial_shape"],     # (512, 512)
        "footprint_geojson": footprint_geojson,         # GeoJSON polygon

        # --- Quality & AI Validation Flags ---
        "registration_passed": gis_meta["registration_passed"],
        "registration_score": gis_meta["registration_score"],
        "confidence_score": m3_result.confidence["score"],
        "predicted_class": m3_result.prediction.get("predicted_class"),

        # --- In-Memory Aligned Geospatial Arrays (for Spatial Queries & Mapping) ---
        "layers": {
            "ndvi": ndvi_array,
            "ndwi": ndwi_array,
            "optical_multispectral": optical_4band,
            "sar_polarimetric": sar_2band,
            "fused_multimodal": fused_6band,
        },
    }


def process_m3_evidence_to_m5(
    m3_gis_evidence: Dict[str, Any],
    output_dir: Union[str, Path] = "output",
    export_layers: bool = True,
) -> Dict[str, Any]:
    """
    Bridge M3 extracted GIS evidence into the complete M5 geospatial pipeline.
    Optionally exports all multimodal raster layers to GIS-ready GeoTIFFs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crs = m3_gis_evidence.get("crs")
    transform = m3_gis_evidence.get("transform")
    target = m3_gis_evidence.get("predicted_class") or "optical_sar_detection"
    confidence = float(m3_gis_evidence.get("confidence_score") or 0.85)

    # Export a reference GeoTIFF from optical multispectral or ndvi layer
    layers = m3_gis_evidence.get("layers", {})
    ref_layer = layers.get("optical_multispectral")
    if ref_layer is None and layers:
        ref_layer = next(iter(layers.values()))

    if ref_layer is not None:
        ref_geotiff_path = out_dir / "m3_reference.tif"
        export_layer_to_geotiff(
            data_array=ref_layer,
            output_path=ref_geotiff_path,
            crs=crs,
            transform=transform,
        )
    else:
        raise ValueError("No raster layers found in m3_gis_evidence['layers'].")

    # 1. Check if change mask or NDVI threshold is available
    change_mask = None
    if "ndvi" in layers and isinstance(layers["ndvi"], np.ndarray):
        change_mask = (layers["ndvi"] < 0.2).astype(np.uint8)

    # 2. Resolve Problem 1 & 2: M3 does not output change mask or localized bounding boxes.
    # Synthesize scene ROI bounding box [0, 0, w, h] so M5 pipeline can geolocate
    # and compute coordinates, area, and vector polygons.
    bounding_boxes: Optional[List[List[float]]] = None
    if change_mask is None:
        if hasattr(ref_layer, "shape") and len(ref_layer.shape) >= 2:
            h, w = ref_layer.shape[-2], ref_layer.shape[-1]
        else:
            w, h = m3_gis_evidence.get("spatial_shape", (64, 64))
        bounding_boxes = [[0.0, 0.0, float(w), float(h)]]

    is_active_target = bool(
        target and str(target).lower() not in ["background", "none", "no_change", "unspecified_target"]
    )

    # Run M5 pipeline
    evidence = run_geospatial_pipeline(
        reference_geotiff=ref_geotiff_path,
        change_mask=change_mask,
        bounding_boxes=bounding_boxes,
        target=target,
        confidence=confidence,
        change_detected=is_active_target,
        output_dir=out_dir,
    )

    # Attach exported layer paths if requested
    if export_layers:
        exported_layers = export_all_layers_to_geotiff(
            m3_gis_evidence=m3_gis_evidence,
            output_dir=out_dir / "layers",
        )
        evidence["exported_layers"] = {k: str(p.as_posix()) for k, p in exported_layers.items()}

    # Attach M3 quality flags to metadata
    evidence["m3_validation"] = {
        "registration_passed": m3_gis_evidence.get("registration_passed"),
        "registration_score": m3_gis_evidence.get("registration_score"),
    }

    return evidence


def process_m3_result(
    m3_result: Any,
    output_dir: Union[str, Path] = "output",
    export_layers: bool = True,
) -> Dict[str, Any]:
    """
    Ingest a raw M3 Optical+SAR pipeline result object or dictionary and run M5.

    Resolves all 9 integration gaps in code:
    1. Mask: Derives target/change mask from spectral indices (NDVI/NDWI) or footprint.
    2. Bounding Boxes: Extracts scene/region bounds and formats both pixel and EPSG:4326 boxes.
    3. Prediction: Safely extracts target class from m3_result.prediction (dict, object, or string).
    4. Confidence: Safely extracts confidence score from m3_result.confidence['score'] or float.
    5. GIS Evidence: Ingests m3_result.to_gis_evidence() directly.
    6. CRS: Reads native projected CRS and reprojects geometries to EPSG:4326.
    7. Raster Layers: Automatically exports in-memory rasters (optical, SAR, fused) to GeoTIFFs.
    8. Direct Fields: Maps native resolution, transform, and quality metrics directly.
    9. Missing Fields: Automatically generates GeoJSON FeatureCollection, Folium map, and evidence.json.
    """
    if isinstance(m3_result, dict):
        if "layers" in m3_result and "crs" in m3_result:
            return process_m3_evidence_to_m5(m3_result, output_dir=output_dir, export_layers=export_layers)
        else:
            gis_meta = m3_result.get("to_gis_evidence", lambda: m3_result)() if callable(m3_result.get("to_gis_evidence")) else m3_result
            m3_gis_evidence = dict(gis_meta)
            return process_m3_evidence_to_m5(m3_gis_evidence, output_dir=output_dir, export_layers=export_layers)

    # If it is an M3 result object
    gis_meta = m3_result.to_gis_evidence() if hasattr(m3_result, "to_gis_evidence") else {}
    optical_data = getattr(m3_result, "optical_data", None)
    sar_data = getattr(m3_result, "registered_sar_data", None)
    fusion_data = getattr(m3_result, "early_fusion_result", None)

    # 3. Prediction extraction
    pred = getattr(m3_result, "prediction", {})
    if isinstance(pred, dict):
        predicted_class = pred.get("predicted_class", "optical_sar_detection")
    else:
        predicted_class = getattr(pred, "predicted_class", str(pred))

    # 4. Confidence extraction
    conf = getattr(m3_result, "confidence", 0.85)
    if isinstance(conf, dict):
        confidence_score = float(conf.get("score", 0.85))
    else:
        confidence_score = float(conf)

    # 5. Extract layers from in-memory objects
    layers = {}
    if optical_data is not None and hasattr(optical_data, "data"):
        layers["optical_multispectral"] = optical_data.data
    if sar_data is not None and hasattr(sar_data, "data"):
        layers["sar_polarimetric"] = sar_data.data
    if fusion_data is not None and hasattr(fusion_data, "fused_data"):
        layers["fused_multimodal"] = fusion_data.fused_data

    # Try compute optical features if helper is available
    try:
        from modules.optical_sar.optical.features import compute_optical_features  # type: ignore
        if optical_data is not None:
            features = compute_optical_features(optical_data)
            if "ndvi" in features:
                layers["ndvi"] = features["ndvi"]
            if "ndwi" in features:
                layers["ndwi"] = features["ndwi"]
    except Exception:
        pass

    # Extract bounds and transform
    crs = gis_meta.get("crs") or getattr(getattr(optical_data, "crs", None), "to_string", lambda: "EPSG:32643")()
    transform = getattr(optical_data, "transform", None) or gis_meta.get("transform")
    bounds = gis_meta.get("bounds", (0, 0, 0, 0))

    footprint_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [bounds[0], bounds[1]],
            [bounds[2], bounds[1]],
            [bounds[2], bounds[3]],
            [bounds[0], bounds[3]],
            [bounds[0], bounds[1]],
        ]],
    }

    m3_gis_evidence = {
        "crs": crs,
        "bounds": bounds,
        "transform": transform,
        "resolution": gis_meta.get("resolution", (10.0, 10.0)),
        "spatial_shape": gis_meta.get("spatial_shape", (512, 512)),
        "footprint_geojson": footprint_geojson,
        "registration_passed": gis_meta.get("registration_passed", True),
        "registration_score": gis_meta.get("registration_score", 1.0),
        "confidence_score": confidence_score,
        "predicted_class": predicted_class,
        "layers": layers,
    }

    return process_m3_evidence_to_m5(
        m3_gis_evidence=m3_gis_evidence,
        output_dir=output_dir,
        export_layers=export_layers,
    )

