"""Comprehensive unit and integration tests for the M2 Change Analysis Subsystem.

Verifies:
1. Identical images -> no change
2. Synthetic changed region -> change detected
3. Threshold behavior
4. Minimum region size filtering
5. Different dimensions handling (resampling)
6. Invalid / unsupported input handling
7. Missing file handling
8. Visualization generation (composite, overlay, mask)
9. Connected-component region extraction (region_id, centroid, bbox, polygon)
10. Geospatial metadata extraction
11. Pixel-to-geographic conversion
12. Area calculation when valid (projected CRS)
13. Non-georeferenced image behavior (geospatial_reference_available=False)
14. RCD adapter status when model unavailable (honest reporting)
15. Grounding adapter status when model unavailable
16. Complete end-to-end M2 pipeline (run_m2)
17. M2 result schema and serialization (to_dict, to_specialist_result)
18. Backward compatibility with detect_changes and run_change_detection
"""
from pathlib import Path
import numpy as np
from PIL import Image

from core.contracts import SpecialistResult
from geospatial.geometry import calculate_ground_area, calculate_pixel_area_m2, pixel_to_geo
from geospatial.raster import (
    bbox_pixel_to_geo,
    inspect_raster,
    is_projected_crs,
    point_pixel_to_geo,
    polygon_pixel_to_geo,
)
from models.change.adapter import run_change_detection, run_grounding
from models.change.align import align_images
from models.change.baseline import detect_changes
from models.change.grounding import GroundingAdapter
from models.change.mask_processing import extract_change_regions
from models.change.pipeline import M2Result, run_m2
from models.change.preprocess import load_and_preprocess
from models.change.rcd import RCDAdapter
from models.change.validation import validate_bitemporal_inputs


def _create_test_image(
    path: Path,
    size: tuple[int, int] = (64, 64),
    fill: int = 120,
    box: tuple[int, int, int, int] | None = None,
    box_fill: int = 240,
) -> Path:
    """Helper to generate deterministic synthetic test rasters."""
    img = Image.new("L", size, fill)
    if box:
        x0, y0, x1, y1 = box
        for x in range(x0, x1):
            for y in range(y0, y1):
                img.putpixel((x, y), box_fill)
    img.save(path)
    return path


# 1. Identical images -> no change
def test_identical_images_produce_no_change(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=100)
    p2 = _create_test_image(tmp_path / "after.png", fill=100)

    result = run_m2(p1, p2, config={"threshold": 0.10})
    assert result.status == "success"
    assert result.change_detected is False
    assert result.changed_pixels == 0
    assert result.change_fraction == 0.0
    assert len(result.regions) == 0


# 2. Synthetic changed region -> change detected
def test_synthetic_change_detected_accurately(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=50)
    # Changed box of 10x10 = 100 pixels in [20, 20, 30, 30]
    p2 = _create_test_image(tmp_path / "after.png", fill=50, box=(20, 20, 30, 30), box_fill=230)

    result = run_m2(p1, p2, config={"threshold": 0.15, "min_pixels": 8})
    assert result.status == "success"
    assert result.change_detected is True
    assert result.changed_pixels == 100
    assert len(result.regions) == 1
    reg = result.regions[0]
    assert reg["pixel_count"] == 100
    assert reg["bbox_pixel"] == {"xmin": 20, "ymin": 20, "xmax": 29, "ymax": 29}


# 3. Threshold behavior
def test_threshold_sensitivity(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=100)
    # Difference is 40/255 ~= 0.157
    p2 = _create_test_image(tmp_path / "after.png", fill=100, box=(10, 10, 20, 20), box_fill=140)

    # Threshold 0.10 should catch it
    res_low = run_m2(p1, p2, config={"threshold": 0.10})
    assert res_low.changed_pixels == 100

    # Threshold 0.25 should ignore it
    res_high = run_m2(p1, p2, config={"threshold": 0.25})
    assert res_high.changed_pixels == 0


# 4. Minimum region size filtering
def test_minimum_region_size_filtering(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=50)
    # Small 2x2 = 4 pixel change
    p2 = _create_test_image(tmp_path / "after.png", fill=50, box=(10, 10, 12, 12), box_fill=250)

    # min_pixels=8 should filter out the 4-pixel speckle
    result = run_m2(p1, p2, config={"threshold": 0.15, "min_pixels": 8})
    assert len(result.regions) == 0

    # min_pixels=2 should retain it
    result_small = run_m2(p1, p2, config={"threshold": 0.15, "min_pixels": 2})
    assert len(result_small.regions) == 1
    assert result_small.regions[0]["pixel_count"] == 4


# 5. Different dimensions (safe resampling)
def test_different_image_dimensions_resampling(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", size=(64, 64), fill=50)
    p2 = _create_test_image(tmp_path / "after.png", size=(32, 32), fill=50)

    result = run_m2(p1, p2)
    assert result.status == "success"
    assert result.quality["alignment_status"] == "resampled"
    assert any("Resampling" in w for w in result.warnings)


# 6. Invalid input handling
def test_invalid_input_corrupt_or_unsupported(tmp_path: Path):
    p1 = tmp_path / "invalid.xyz"
    p1.write_text("not an image")
    p2 = _create_test_image(tmp_path / "after.png")

    result = run_m2(p1, p2)
    assert result.status == "failed"
    assert result.error is not None
    assert "Unsupported" in result.error or "Cannot read" in result.error


# 7. Missing file handling
def test_missing_input_file(tmp_path: Path):
    p1 = tmp_path / "non_existent.png"
    p2 = _create_test_image(tmp_path / "after.png")

    result = run_m2(p1, p2)
    assert result.status == "failed"
    assert "does not exist" in (result.error or "")


# 8. Visualization generation
def test_visualization_generation(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=50)
    p2 = _create_test_image(tmp_path / "after.png", fill=50, box=(15, 15, 25, 25), box_fill=220)
    out_dir = tmp_path / "outputs"

    result = run_m2(p1, p2, output_dir=out_dir)
    assert result.status == "success"
    assert len(result.artifacts) >= 1
    assert (out_dir / result.artifacts[0]).exists()
    assert result.composite_path is not None and Path(result.composite_path).exists()
    assert result.overlay_path is not None and Path(result.overlay_path).exists()
    assert result.mask_path is not None and Path(result.mask_path).exists()


# 9. Connected-component region extraction
def test_connected_component_extraction(tmp_path: Path):
    raw_mask = np.zeros((30, 30), dtype=bool)
    diff_map = np.zeros((30, 30), dtype=np.float32)

    # Region 1: 5x5 = 25 pixels
    raw_mask[2:7, 2:7] = True
    diff_map[2:7, 2:7] = 0.8

    # Region 2: 3x3 = 9 pixels
    raw_mask[20:23, 20:23] = True
    diff_map[20:23, 20:23] = 0.5

    clean_mask, regions = extract_change_regions(raw_mask, diff_map, min_pixels=5)
    assert len(regions) == 2
    assert regions[0].pixel_count == 25
    assert regions[0].region_id == 1
    assert regions[0].centroid_pixel["x"] == 4.0
    assert regions[0].centroid_pixel["y"] == 4.0
    assert regions[0].bbox_pixel == {"xmin": 2, "ymin": 2, "xmax": 6, "ymax": 6}
    assert len(regions[0].polygon_pixel) >= 4


# 10. Geospatial metadata inspection
def test_geospatial_metadata_inspection(tmp_path: Path):
    p = _create_test_image(tmp_path / "test.png", size=(40, 20))
    meta = inspect_raster(str(p))
    assert meta["width"] == 40
    assert meta["height"] == 20
    assert meta["georeferenced"] is False
    assert meta["crs"] is None


# 11. Pixel to geographic coordinates
def test_pixel_to_geo_transformations():
    # Affine transform: x_geo = 10*x + 500000, y_geo = -10*y + 3000000 (e.g. UTM 10m res)
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)

    # Single point
    gx, gy = point_pixel_to_geo(5, 5, transform)
    assert gx == 500050.0
    assert gy == 2999950.0

    # Bounding box
    bbox_geo = bbox_pixel_to_geo({"xmin": 0, "ymin": 0, "xmax": 9, "ymax": 9}, transform)
    assert bbox_geo["top_left"] == (500000.0, 3000000.0)
    assert bbox_geo["bottom_right"] == (500100.0, 2999900.0)

    # Polygon
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    poly_geo = polygon_pixel_to_geo(poly, transform)
    assert len(poly_geo) == 4
    assert poly_geo[0] == (500000.0, 3000000.0)


# 12. Area calculation when projected vs geographic
def test_ground_area_calculation():
    # 10m x 10m UTM projected transform -> pixel area is 100 m²
    transform_utm = (10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
    crs_utm = "EPSG:32643"  # UTM Zone 43N

    area_100_px = calculate_ground_area(100, transform_utm, crs_utm)
    assert area_100_px == 10000.0  # 100 px * 100 m²/px

    # Geographic degrees -> area should return None (not blindly calculated as m²)
    transform_geo = (0.0001, 0.0, 77.0, 0.0, -0.0001, 28.0)
    crs_geo = "EPSG:4326"
    assert calculate_ground_area(100, transform_geo, crs_geo) is None

    # Projected CRS check
    assert is_projected_crs("EPSG:32643") is True
    assert is_projected_crs("UTM zone 32N") is True
    assert is_projected_crs("EPSG:4326") is False
    assert is_projected_crs("WGS 84") is False


# 13. Non-georeferenced image behavior
def test_non_georeferenced_image_handling(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "b.png")
    p2 = _create_test_image(tmp_path / "a.png")

    result = run_m2(p1, p2)
    assert result.geospatial_reference_available is False
    assert result.crs is None
    assert result.changed_area_sq_m is None
    assert result.geographic_bbox is None


# 14. RCD adapter status when model unavailable
def test_rcd_adapter_status_reporting(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=50)
    p2 = _create_test_image(tmp_path / "after.png", fill=50, box=(5, 5, 15, 15), box_fill=200)

    # Calling run_m2 with a target query
    target_query = "newly constructed buildings"
    result = run_m2(p1, p2, target=target_query)

    assert result.target == target_query
    # Must honestly report fallback_baseline, not fake neural model
    assert result.detector_type == "fallback_baseline"
    assert result.status == "fallback_baseline"
    assert "rcd" in result.detector.lower() or "fallback" in result.detector.lower()
    assert result.change_detected is True


def test_rcd_adapter_awaiting_model_when_fallback_disabled(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=50)
    p2 = _create_test_image(tmp_path / "after.png", fill=50, box=(5, 5, 15, 15), box_fill=200)

    # With allow_fallback=False, it must report awaiting_model rather than fallback
    result = run_m2(p1, p2, target="newly constructed buildings", config={"allow_fallback": False})
    assert result.status == "awaiting_model"
    assert result.detector_type == "adapter"
    assert "rcd" in result.detector.lower()


# 15. Grounding adapter status when model unavailable
def test_grounding_adapter_status():
    adapter = GroundingAdapter()
    res = adapter.ground_target(None, "water body that increased")
    assert res.status == "awaiting_model"
    assert res.confidence == 0.0
    assert "awaiting_model" in res.to_specialist_result().status


# 16. Complete end-to-end M2 pipeline
def test_end_to_end_m2_pipeline_comprehensive(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "b.png", size=(48, 48), fill=40)
    p2 = _create_test_image(tmp_path / "a.png", size=(48, 48), fill=40, box=(10, 10, 20, 20), box_fill=240)
    out = tmp_path / "viz"

    result = run_m2(
        before_path=p1,
        after_path=p2,
        target="vegetation loss",
        output_dir=out,
        config={"threshold": 0.15, "min_pixels": 10},
    )

    assert isinstance(result, M2Result)
    assert result.status in {"success", "fallback_baseline"}
    assert result.change_detected is True
    assert result.changed_pixels == 100
    assert result.confidence > 0.0
    assert len(result.regions) == 1
    assert result.regions[0]["target"] == "vegetation loss"
    assert result.composite_path is not None


# 17. M2 Result Schema & Serialization
def test_m2_result_schema_and_serialization(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "b.png")
    p2 = _create_test_image(tmp_path / "a.png")

    result = run_m2(p1, p2)
    payload = result.to_dict()

    expected_keys = {
        "task", "status", "target", "detector", "detector_type",
        "change_detected", "confidence", "changed_pixels", "change_fraction",
        "mean_difference", "regions", "number_of_regions", "region_sizes",
        "artifacts", "mask_path", "overlay_path", "composite_path",
        "geospatial_reference_available", "crs", "transform",
        "geographic_bbox", "changed_area_sq_m", "quality", "warnings", "metadata", "error",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["number_of_regions"] == 0
    assert payload["region_sizes"] == []

    # Verify to_specialist_result conversion for M4
    spec_res = result.to_specialist_result()
    assert isinstance(spec_res, SpecialistResult)
    assert spec_res.task == "change_detection"
    assert "evidence" in spec_res.to_dict()
    assert "number_of_regions" in spec_res.evidence
    assert "region_sizes" in spec_res.evidence


# 18. Backward compatibility with detect_changes & run_change_detection
def test_backward_compatibility_existing_interfaces(tmp_path: Path):
    p1 = _create_test_image(tmp_path / "before.png", fill=60)
    p2 = _create_test_image(tmp_path / "after.png", fill=60, box=(10, 10, 20, 20), box_fill=220)

    # Test baseline.detect_changes
    base_res = detect_changes(str(p1), str(p2), threshold=0.15, min_pixels=8)
    assert base_res["status"] == "success"
    assert base_res["model"] == "pixel-difference-baseline"
    assert base_res["changed_pixels"] == 100
    assert base_res["regions"][0]["pixel_count"] == 100
    assert "bbox_pixel" in base_res["regions"][0]

    # Test adapter.run_change_detection
    spec_res = run_change_detection(str(p1), str(p2), target="roads", output_dir=tmp_path / "arts")
    assert isinstance(spec_res, SpecialistResult)
    assert spec_res.status == "success"
    assert spec_res.evidence["changed_pixels"] == 100
    assert len(spec_res.artifacts) == 1
    assert (tmp_path / "arts" / spec_res.artifacts[0]).exists()

    # Test adapter.run_grounding
    ground_res = run_grounding(str(p1), "roads")
    assert isinstance(ground_res, SpecialistResult)
    assert ground_res.status == "awaiting_model"
