"""
Unit and Integration Test Suite for SatQuery-AI Module 5 (M5):
GIS, Geospatial Processing and Evidence Generation.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import shapely.geometry

from geospatial.area import calculate_polygon_area, calculate_total_area
from geospatial.coordinates import pixel_bbox_to_geo_bbox, pixel_to_geo
from geospatial.evidence import generate_evidence_json, generate_geojson
from geospatial.integration import (
    process_m2_m3_result,
    process_m2_result,
    process_m3_result,
    process_m4_result,
)
from geospatial.m3_adapter import (
    export_layer_to_geotiff,
    process_m3_evidence_to_m5,
)
from geospatial.metadata import read_geotiff_metadata
from geospatial.pipeline import run_geospatial_pipeline
from geospatial.polygons import mask_to_polygons
from geospatial.schema import (
    M2ChangeDetectionResult,
    M2M3Payload,
    M2Region,
    M4SpecialistResult,
)
from geospatial.visualization import generate_folium_map

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_MOCK_DIR = BASE_DIR / "data" / "mock"
SAMPLE_GEOTIFF = DATA_MOCK_DIR / "sample.tif"
SAMPLE_MASK = DATA_MOCK_DIR / "change_mask.npy"
M2_RESULT_JSON = DATA_MOCK_DIR / "m2_result.json"
M2_GEOREF_JSON = DATA_MOCK_DIR / "m2_georef_output.json"
M2_NON_GEOREF_JSON = DATA_MOCK_DIR / "m2_non_georef_output.json"
M2_REAL_RESULT_JSON = DATA_MOCK_DIR / "m2_real_result.json"
M2_NON_GEOREFERENCED_JSON = DATA_MOCK_DIR / "m2_non_georeferenced.json"
M4_SPECIALIST_JSON = DATA_MOCK_DIR / "m4_specialist_result.json"
M4_M2_RESULT_JSON = DATA_MOCK_DIR / "m4_m2_result.json"
M3_PIPELINE_JSON = DATA_MOCK_DIR / "m3_pipeline_output.json"
OUTPUT_TEST_DIR = BASE_DIR / "output" / "test_run"


@pytest.fixture(scope="session", autouse=True)
def setup_mock_data():
    """Ensure mock data is generated before tests run."""
    if not SAMPLE_GEOTIFF.exists() or not SAMPLE_MASK.exists() or not M2_RESULT_JSON.exists():
        from data.mock.generate_mock import create_sample_mock_data
        create_sample_mock_data(DATA_MOCK_DIR)
    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)


class TestM5Geospatial:
    """Test suite covering all M5 requirements."""

    def test_01_read_geotiff_metadata(self):
        """Verify reading GeoTIFF metadata via rasterio."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)

        assert "crs" in meta
        assert meta["crs"] is not None
        assert "width" in meta and meta["width"] == 256
        assert "height" in meta and meta["height"] == 256
        assert "bounds" in meta
        assert "resolution" in meta and meta["resolution"] == (10.0, 10.0)
        assert "transform" in meta and len(meta["transform"]) >= 6
        assert "count" in meta and meta["count"] == 3
        assert meta["is_projected"] is True

        # Non-existent file error handling
        with pytest.raises(FileNotFoundError):
            read_geotiff_metadata("non_existent_file.tif")

    def test_02_pixel_to_geo(self):
        """Verify converting pixel coordinates to geographic coordinates (EPSG:4326)."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        lon, lat = pixel_to_geo(
            col=128,
            row=128,
            transform=meta["transform"],
            crs=meta["crs"],
            to_crs="EPSG:4326",
        )

        # Coordinate sanity checks for UTM Zone 43N (India/Asia region: lon ~68-78°E, lat ~25-35°N)
        assert 60.0 <= lon <= 85.0
        assert 20.0 <= lat <= 40.0

    def test_03_pixel_bbox_to_geo_bbox(self):
        """Verify converting pixel bounding boxes to latitude/longitude."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        pixel_bbox = [40, 50, 90, 100]

        result = pixel_bbox_to_geo_bbox(
            bbox_pixel=pixel_bbox,
            transform=meta["transform"],
            crs=meta["crs"],
            to_crs="EPSG:4326",
        )

        assert "pixel_bbox" in result and result["pixel_bbox"] == pixel_bbox
        assert "geo_bbox" in result and len(result["geo_bbox"]) == 4
        min_lon, min_lat, max_lon, max_lat = result["geo_bbox"]
        assert min_lon < max_lon
        assert min_lat < max_lat

        assert "corners_geo" in result and len(result["corners_geo"]) == 4
        assert "polygon_coords" in result and len(result["polygon_coords"]) == 5
        assert "centroid" in result and len(result["centroid"]) == 2

    def test_04_mask_to_polygons(self):
        """Verify converting change mask into EPSG:4326 Shapely polygons."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        polygons = mask_to_polygons(
            mask=SAMPLE_MASK,
            transform=meta["transform"],
            crs=meta["crs"],
            to_crs="EPSG:4326",
        )

        assert len(polygons) == 2
        for poly in polygons:
            assert isinstance(poly, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))
            assert poly.is_valid
            assert not poly.is_empty
            min_lon, min_lat, _, _ = poly.bounds
            assert 60.0 <= min_lon <= 85.0
            assert 20.0 <= min_lat <= 40.0

        # Empty mask returns empty list
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        assert mask_to_polygons(empty_mask, meta["transform"], meta["crs"]) == []

    def test_05_calculate_polygon_area(self):
        """Verify polygon area calculation in square meters and hectares."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        polygons = mask_to_polygons(
            mask=SAMPLE_MASK,
            transform=meta["transform"],
            crs=meta["crs"],
            to_crs="EPSG:4326",
        )

        # Patch 1 is 50x50 pixels = 500m x 500m = 250,000 m² = 25 hectares
        area_info = calculate_polygon_area(polygons[0], crs="EPSG:4326")
        assert "area_sq_meters" in area_info
        assert "area_hectares" in area_info
        assert "area_sq_km" in area_info

        # Geodesic area should match expected 250,000 m² within 1.5%
        assert 240000.0 <= area_info["area_sq_meters"] <= 260000.0
        assert 24.0 <= area_info["area_hectares"] <= 26.0

        total_info = calculate_total_area(polygons, crs="EPSG:4326")
        assert total_info["polygon_count"] == 2
        assert total_info["total_area_hectares"] > area_info["area_hectares"]

    def test_06_generate_geojson(self):
        """Verify GeoJSON FeatureCollection generation."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        polygons = mask_to_polygons(SAMPLE_MASK, meta["transform"], meta["crs"])
        bbox_info = pixel_bbox_to_geo_bbox([40, 50, 90, 100], meta["transform"], meta["crs"])

        geojson_path = OUTPUT_TEST_DIR / "test_evidence.geojson"
        geojson_data = generate_geojson(
            polygons=polygons,
            bboxes_geo=[bbox_info],
            target="deforestation",
            confidence=0.92,
            output_path=geojson_path,
        )

        assert geojson_data["type"] == "FeatureCollection"
        # 2 change polygons + 1 bounding box = 3 features
        assert len(geojson_data["features"]) == 3
        assert geojson_path.exists()

        # Validate saved file matches
        with open(geojson_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            assert loaded["type"] == "FeatureCollection"

    def test_07_generate_folium_map(self):
        """Verify Folium interactive map generation."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        polygons = mask_to_polygons(SAMPLE_MASK, meta["transform"], meta["crs"])
        bbox_info = pixel_bbox_to_geo_bbox([40, 50, 90, 100], meta["transform"], meta["crs"])

        map_path = OUTPUT_TEST_DIR / "test_map.html"
        folium_map = generate_folium_map(
            polygons=polygons,
            bboxes_geo=[bbox_info],
            target="deforestation",
            confidence=0.94,
            output_html_path=map_path,
        )

        assert folium_map is not None
        assert map_path.exists()
        content = map_path.read_text(encoding="utf-8")
        assert "leaflet" in content.lower()
        assert "deforestation" in content.lower()

    def test_08_generate_evidence_json(self):
        """Verify evidence.json conforms to all project requirements."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        polygons = mask_to_polygons(SAMPLE_MASK, meta["transform"], meta["crs"])
        bbox_info = pixel_bbox_to_geo_bbox([40, 50, 90, 100], meta["transform"], meta["crs"])

        evidence_path = OUTPUT_TEST_DIR / "test_evidence.json"
        evidence = generate_evidence_json(
            target="deforestation",
            confidence=0.94,
            change_detected=True,
            bounding_boxes=[bbox_info],
            geographic_coordinates={"center": [28.0, 77.0]},
            polygons=polygons,
            geojson_path="output/evidence.geojson",
            map_path="output/map.html",
            raster_metadata=meta,
            output_path=evidence_path,
        )

        # Assert all 9 required keys are present
        assert evidence["target"] == "deforestation"
        assert evidence["confidence"] == 0.94
        assert evidence["change_detected"] is True
        assert "bounding_boxes" in evidence and len(evidence["bounding_boxes"]) == 1
        assert "geographic_coordinates" in evidence
        assert "polygons" in evidence and len(evidence["polygons"]) == 2
        assert "area" in evidence
        assert "total_sq_meters" in evidence["area"]
        assert "total_hectares" in evidence["area"]
        assert evidence["geojson_path"] == "output/evidence.geojson"
        assert evidence["map_path"] == "output/map.html"
        assert evidence["evidence_path"] == str(Path(evidence_path).as_posix())

        assert evidence_path.exists()

    def test_09_run_geospatial_pipeline_end_to_end(self):
        """Verify full end-to-end execution of run_geospatial_pipeline producing required output files."""
        result = run_geospatial_pipeline(
            reference_geotiff=SAMPLE_GEOTIFF,
            change_mask=SAMPLE_MASK,
            bounding_boxes=[[40, 50, 90, 100], [140, 160, 200, 185]],
            target="deforestation",
            confidence=0.95,
            output_dir="output",
        )

        assert result["change_detected"] is True
        assert result["target"] == "deforestation"
        assert len(result["bounding_boxes"]) == 2
        assert len(result["polygons"]) == 2
        assert result["area"]["total_hectares"] > 0

        # Verify all 3 required files are generated in output/
        evidence_json_path = Path("output/evidence.json")
        geojson_path = Path("output/evidence.geojson")
        map_html_path = Path("output/map.html")

        assert evidence_json_path.exists(), "output/evidence.json must exist"
        assert geojson_path.exists(), "output/evidence.geojson must exist"
        assert map_html_path.exists(), "output/map.html must exist"

        # Verify exact path strings in result payload
        assert result["evidence_path"] == "output/evidence.json"
        assert result["geojson_path"] == "output/evidence.geojson"
        assert result["map_path"] == "output/map.html"

        # Verify saved evidence.json contains the correct paths internally
        with open(evidence_json_path, "r", encoding="utf-8") as f:
            evidence_file_data = json.load(f)
            assert evidence_file_data["evidence_path"] == "output/evidence.json"
            assert evidence_file_data["geojson_path"] == "output/evidence.geojson"
            assert evidence_file_data["map_path"] == "output/map.html"

    def test_10_pipeline_with_mock_m2_payload(self):
        """Verify pipeline consumes mock upstream M2 output correctly."""
        with open(M2_RESULT_JSON, "r", encoding="utf-8") as f:
            m2_data = json.load(f)

        out_dir = OUTPUT_TEST_DIR / "from_m2_payload"
        result = run_geospatial_pipeline(
            reference_geotiff=m2_data["reference_geotiff"],
            change_mask=m2_data["change_mask_path"],
            bounding_boxes=m2_data["bounding_boxes_pixel"],
            target=m2_data["target"],
            confidence=m2_data["confidence"],
            output_dir=out_dir,
        )

        assert result["target"] == "deforestation"
        assert result["confidence"] == 0.94
        assert result["change_detected"] is True
        assert (out_dir / "evidence.json").exists()

    def test_11_process_m2_m3_result_standard_payload(self):
        """Verify process_m2_m3_result accepts standardized team payload."""
        standard_payload = {
            "target": "new building",
            "confidence": 0.91,
            "change_detected": True,
            "reference_image": str(SAMPLE_GEOTIFF),
            "change_mask": str(SAMPLE_MASK),
            "bounding_boxes": [
                [50, 60, 100, 110]
            ],
        }

        out_dir = OUTPUT_TEST_DIR / "integration_standard"
        evidence = process_m2_m3_result(standard_payload, output_dir=out_dir)

        assert evidence["target"] == "new building"
        assert evidence["confidence"] == 0.91
        assert evidence["change_detected"] is True
        assert len(evidence["bounding_boxes"]) == 1
        assert len(evidence["polygons"]) == 2
        assert evidence["area"]["total_hectares"] > 0

        # Verify all 3 required artifacts are produced
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

    def test_12_process_m2_m3_result_dataclass_and_json_file(self):
        """Verify process_m2_m3_result accepts M2M3Payload dataclass and JSON path."""
        # 1. Test from JSON file path directly
        out_dir_file = OUTPUT_TEST_DIR / "integration_from_json_file"
        res_file = process_m2_m3_result(M2_RESULT_JSON, output_dir=out_dir_file)
        assert res_file["target"] == "deforestation"
        assert (out_dir_file / "evidence.json").exists()

        # 2. Test from dataclass instance
        payload_obj = M2M3Payload(
            target="unauthorized construction",
            confidence=0.88,
            change_detected=True,
            reference_image=str(SAMPLE_GEOTIFF),
            change_mask=str(SAMPLE_MASK),
            bounding_boxes=[[30, 30, 80, 80]],
        )
        out_dir_obj = OUTPUT_TEST_DIR / "integration_dataclass"
        res_obj = process_m2_m3_result(payload_obj, output_dir=out_dir_obj)
        assert res_obj["target"] == "unauthorized construction"
        assert (out_dir_obj / "evidence.json").exists()
        assert (out_dir_obj / "map.html").exists()

    def test_13_process_m2_m3_result_validation(self):
        """Verify proper error handling for invalid or missing payload data."""
        # Missing reference image
        with pytest.raises(ValueError):
            process_m2_m3_result({"target": "test", "confidence": 0.9})

        # Non-existent reference image file
        with pytest.raises(FileNotFoundError):
            process_m2_m3_result({
                "reference_image": "non_existent_sat_image.tif",
                "target": "test",
            })

        # Invalid type
        with pytest.raises(TypeError):
            process_m2_m3_result(12345)  # type: ignore[arg-type]

    def test_14_export_layer_to_geotiff_2d_and_3d(self):
        """Verify exporting 2D (NDVI) and 3D (Multispectral/SAR) arrays to GeoTIFF."""
        import rasterio

        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        transform = meta["transform"]
        crs = meta["crs"]

        # 1. Test 2D layer (e.g. NDVI, float32)
        ndvi_arr = np.random.uniform(-0.2, 0.8, size=(64, 64)).astype(np.float32)
        out_2d = OUTPUT_TEST_DIR / "m3_export" / "ndvi_test.tif"
        written_2d = export_layer_to_geotiff(ndvi_arr, out_2d, crs, transform)

        assert written_2d.exists()
        with rasterio.open(written_2d) as src:
            assert src.count == 1
            assert src.width == 64
            assert src.height == 64
            assert src.crs.to_string() == crs

        # 2. Test 3D layer (e.g. 4-band optical RGBN, uint8)
        optical_arr = np.random.randint(0, 255, size=(4, 64, 64), dtype=np.uint8)
        out_3d = OUTPUT_TEST_DIR / "m3_export" / "optical_4band.tif"
        written_3d = export_layer_to_geotiff(optical_arr, out_3d, crs, transform)

        assert written_3d.exists()
        with rasterio.open(written_3d) as src:
            assert src.count == 4
            assert src.width == 64
            assert src.height == 64
            assert src.crs.to_string() == crs

    def test_15_process_m3_evidence_to_m5(self):
        """Verify bridging M3 multimodal analysis into complete M5 pipeline and GeoTIFF layer exports."""
        meta = read_geotiff_metadata(SAMPLE_GEOTIFF)
        transform = meta["transform"]
        crs = meta["crs"]

        # Simulated M3 multimodal GIS output
        simulated_m3_evidence = {
            "crs": crs,
            "bounds": [700000.0, 3097440.0, 702560.0, 3100000.0],
            "transform": transform,
            "resolution": (10.0, 10.0),
            "spatial_shape": (64, 64),
            "footprint_geojson": {
                "type": "Polygon",
                "coordinates": [[[700000.0, 3097440.0], [702560.0, 3097440.0], [702560.0, 3100000.0], [700000.0, 3100000.0], [700000.0, 3097440.0]]]
            },
            "registration_passed": True,
            "registration_score": 0.98,
            "confidence_score": 0.93,
            "predicted_class": "industrial_facility",
            "layers": {
                "optical_multispectral": np.random.randint(10, 200, size=(4, 64, 64), dtype=np.uint8),
                "sar_polarimetric": np.random.uniform(-25.0, 0.0, size=(2, 64, 64)).astype(np.float32),
                "ndvi": np.random.uniform(-0.1, 0.7, size=(64, 64)).astype(np.float32),
                "ndwi": np.random.uniform(-0.3, 0.5, size=(64, 64)).astype(np.float32),
            },
        }

        out_dir = OUTPUT_TEST_DIR / "m3_bridge_test"
        evidence = process_m3_evidence_to_m5(
            m3_gis_evidence=simulated_m3_evidence,
            output_dir=out_dir,
            export_layers=True,
        )

        assert evidence["target"] == "industrial_facility"
        assert evidence["confidence"] == 0.93
        assert evidence["change_detected"] is True
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        # Check that individual layers were exported to GeoTIFF
        assert (out_dir / "layers" / "optical_multispectral.tif").exists()
        assert (out_dir / "layers" / "sar_polarimetric.tif").exists()
        assert (out_dir / "layers" / "ndvi.tif").exists()
        assert (out_dir / "layers" / "ndwi.tif").exists()
        assert "exported_layers" in evidence

    def test_16_process_real_m2_georeferenced_detector_output(self):
        """Verify processing real M2 georeferenced detector output format."""
        out_dir = OUTPUT_TEST_DIR / "m2_georef_test"
        evidence = process_m2_m3_result(M2_GEOREF_JSON, output_dir=out_dir)

        assert evidence["target"] == "newly constructed buildings"
        assert evidence["confidence"] == 0.75
        assert evidence["change_detected"] is True
        assert len(evidence["bounding_boxes"]) == 2
        assert len(evidence["polygons"]) == 2
        assert evidence["area"]["total_hectares"] > 0
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        # Check that coordinates are in EPSG:4326 (WGS84 lon ~75, lat ~27)
        first_bbox = evidence["bounding_boxes"][0]["geo_bbox"]
        assert 65.0 <= first_bbox[0] <= 85.0
        assert 20.0 <= first_bbox[1] <= 35.0

    def test_17_process_real_m2_non_georeferenced_output(self):
        """Verify graceful fallback for non-georeferenced M2 outputs (plain PNG/JPG)."""
        out_dir = OUTPUT_TEST_DIR / "m2_non_georef_test"
        evidence = process_m2_m3_result(M2_NON_GEOREF_JSON, output_dir=out_dir)

        assert evidence["change_detected"] is True
        assert len(evidence["bounding_boxes"]) == 1
        assert len(evidence["polygons"]) == 1
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()
        assert evidence["raster_metadata"]["geospatial_reference_available"] is False

    def test_18_process_m4_specialist_result_wrapper(self):
        """Verify processing M2 output when received wrapped in M4 SpecialistResult."""
        out_dir = OUTPUT_TEST_DIR / "m4_wrapper_test"
        evidence = process_m2_m3_result(M4_SPECIALIST_JSON, output_dir=out_dir)

        assert evidence["confidence"] == 0.75
        assert evidence["change_detected"] is True
        assert len(evidence["bounding_boxes"]) == 2
        assert len(evidence["polygons"]) == 2
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()
        assert evidence["detector_metadata"]["is_m4_wrapped"] is True
        assert "Detected 2 changed region(s)" in evidence["detector_metadata"]["m4_claim"]

    def test_19_process_m3_result_object_solving_9_problems(self):
        """Verify process_m3_result ingests raw m3_result object solving all 9 integration challenges."""
        from rasterio.transform import Affine

        class MockData:
            def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform

        class MockM3Result:
            def __init__(self):
                self.optical_data = MockData(
                    data=np.random.randint(10, 200, size=(4, 64, 64), dtype=np.uint8),
                    transform=Affine(10.0, 0.0, 700000.0, 0.0, -10.0, 3100000.0),
                )
                self.registered_sar_data = MockData(
                    data=np.random.uniform(-25.0, 0.0, size=(2, 64, 64)).astype(np.float32)
                )

                class EarlyFusion:
                    fused_data = np.random.uniform(0.0, 1.0, size=(6, 64, 64)).astype(np.float32)

                self.early_fusion_result = EarlyFusion()
                self.confidence = {"score": 0.94}
                self.prediction = {"predicted_class": "aircraft_hangar"}

            def to_gis_evidence(self):
                return {
                    "crs": "EPSG:32643",
                    "bounds": [700000.0, 3097440.0, 702560.0, 3100000.0],
                    "resolution": (10.0, 10.0),
                    "spatial_shape": (64, 64),
                    "registration_passed": True,
                    "registration_score": 0.99,
                }

        mock_obj = MockM3Result()
        out_dir = OUTPUT_TEST_DIR / "m3_raw_obj_test"
        evidence = process_m3_result(mock_obj, output_dir=out_dir)

        assert evidence["target"] == "aircraft_hangar"
        assert evidence["confidence"] == 0.94
        assert evidence["change_detected"] is True
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()
        assert (out_dir / "layers" / "optical_multispectral.tif").exists()
        assert (out_dir / "layers" / "sar_polarimetric.tif").exists()
        assert (out_dir / "layers" / "fused_multimodal.tif").exists()
        assert evidence["m3_validation"]["registration_passed"] is True

    def test_20_process_m3_pipeline_json_payload(self):
        """Verify process_m2_m3_result processes official M3 pipeline output payload."""
        out_dir = OUTPUT_TEST_DIR / "m3_official_pipeline_payload"
        evidence = process_m2_m3_result(M3_PIPELINE_JSON, output_dir=out_dir)

        assert evidence["target"] == "Vegetation"
        assert evidence["confidence"] == 0.878
        assert evidence["change_detected"] is True
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        assert evidence["area"]["total_sq_meters"] > 0
        assert evidence["area"]["total_hectares"] > 0
        assert evidence["raster_metadata"]["crs"] == "EPSG:32632"
        assert evidence["m3_multimodal_metadata"]["registration_score"] == 0.4005
        assert evidence["m3_multimodal_metadata"]["registration_passed"] is False
        assert evidence["m3_multimodal_metadata"]["prediction"]["predicted_class_name"] == "Vegetation"

    def test_m2_georeferenced_integration(self):
        """Verify processing real georeferenced M2 payload with dynamic CRS and EPSG:4326 reprojection."""
        out_dir = OUTPUT_TEST_DIR / "m2_real_georef_test"
        evidence = process_m2_result(M2_REAL_RESULT_JSON, output_dir=out_dir)

        # Basic fields
        assert evidence["target"] == "newly constructed buildings"
        assert evidence["confidence"] == 0.75
        assert evidence["change_detected"] is True
        assert evidence["raster_metadata"]["crs"] == "EPSG:32643"
        assert evidence["raster_metadata"]["geospatial_reference_available"] is True

        # Region metadata preservation
        assert len(evidence["bounding_boxes"]) == 1
        bbox = evidence["bounding_boxes"][0]
        assert bbox["region_id"] == 1
        assert bbox["pixel_count"] == 350
        assert bbox["confidence"] == 0.88
        assert bbox["target"] == "newly constructed buildings"
        assert bbox["pixel_bbox"] == [20.0, 30.0, 55.0, 65.0]

        # Coordinates converted to EPSG:4326 (WGS84 degrees, lon ~75°E, lat ~27°N)
        assert bbox["geo_bbox"] is not None
        min_lon, min_lat, max_lon, max_lat = bbox["geo_bbox"]
        assert 60.0 <= min_lon <= 85.0
        assert 20.0 <= min_lat <= 40.0
        assert min_lon < max_lon
        assert min_lat < max_lat

        # Area validation
        assert evidence["area"] is not None
        assert evidence["area"]["total_sq_meters"] > 0
        assert evidence["area"]["total_hectares"] > 0
        assert evidence["area"]["m2_reported_area_sq_m"] == 45000.0

        # Artifacts generation
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        # Verify GeoJSON uses EPSG:4326 coordinates and preserves properties
        with open(out_dir / "evidence.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
            assert geojson_data["type"] == "FeatureCollection"
            assert len(geojson_data["features"]) >= 1
            feat = geojson_data["features"][0]
            assert feat["geometry"]["type"] == "Polygon"
            coords = feat["geometry"]["coordinates"][0]
            for pt in coords:
                assert 60.0 <= pt[0] <= 85.0
                assert 20.0 <= pt[1] <= 40.0
            assert feat["properties"]["region_id"] == 1
            assert feat["properties"]["pixel_count"] == 350

    def test_m2_non_georeferenced_integration(self):
        """Verify non-georeferenced M2 output sets geographic coordinates and area to null and adds warning."""
        out_dir = OUTPUT_TEST_DIR / "m2_real_non_georef_test"
        evidence = process_m2_result(M2_NON_GEOREFERENCED_JSON, output_dir=out_dir)

        assert evidence["change_detected"] is True
        assert evidence["raster_metadata"]["geospatial_reference_available"] is False
        assert evidence["raster_metadata"]["crs"] is None
        assert evidence["raster_metadata"]["transform"] is None

        # Requirement 7: Geographic coordinates and area must be null
        assert evidence["geographic_coordinates"] is None
        assert evidence["area"] is None

        # Requirement 7: Warning must be present
        expected_warning = "Images are not georeferenced; geographic coordinates and real-world area are unavailable."
        assert expected_warning in evidence["warnings"]

        # Pixel coordinates preserved
        assert len(evidence["bounding_boxes"]) == 2
        bbox1 = evidence["bounding_boxes"][0]
        assert bbox1["region_id"] == 1
        assert bbox1["pixel_count"] == 350
        assert bbox1["pixel_bbox"] == [20.0, 30.0, 55.0, 65.0]
        assert bbox1["geo_bbox"] is None
        assert bbox1["area_sq_m"] is None

        # Artifacts generation
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        # Verify GeoJSON does NOT invent latitude/longitude
        with open(out_dir / "evidence.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
            assert geojson_data["type"] == "FeatureCollection"
            for feat in geojson_data["features"]:
                assert feat["geometry"] is None
                assert expected_warning in feat["properties"]["warning"]

    def test_m4_wrapped_m2_integration(self):
        """Verify processing M4 SpecialistResult wrapper extracting inner M2 evidence."""
        out_dir = OUTPUT_TEST_DIR / "m4_wrapped_m2_test"
        evidence = process_m4_result(M4_M2_RESULT_JSON, output_dir=out_dir)

        assert evidence["target"] == "newly constructed buildings"
        assert evidence["confidence"] == 0.75
        assert evidence["change_detected"] is True
        assert evidence["detector_metadata"]["is_m4_wrapped"] is True
        assert "Detected 2 changed region(s)" in evidence["detector_metadata"]["m4_claim"]
        assert evidence["raster_metadata"]["crs"] == "EPSG:32643"
        assert evidence["area"]["total_hectares"] > 0

        # Artifacts generation
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        # Also test passing M4 wrapper to process_m2_result directly (auto-detection)
        auto_dir = OUTPUT_TEST_DIR / "m4_auto_detect_test"
        auto_evidence = process_m2_result(M4_M2_RESULT_JSON, output_dir=auto_dir)
        assert auto_evidence["detector_metadata"]["is_m4_wrapped"] is True
        assert (auto_dir / "evidence.json").exists()

    def test_21_real_m3_pipeline_to_m5_integration(self, tmp_path: Path):
        """
        Verify real end-to-end integration from M3 run_optical_sar_pipeline()
        to M5 process_m3_result() without fake or mock runtime.
        """
        import affine
        import rasterio
        from rasterio.crs import CRS as RioCRS
        from modules.optical_sar.pipeline import run_optical_sar_pipeline
        from modules.optical_sar.config import OpticalSARConfig

        # 1. Create real test GeoTIFF rasters with projected CRS EPSG:32632
        pixel_size = 10.0
        transform = affine.Affine(pixel_size, 0.0, 684000.0, 0.0, -pixel_size, 5339120.0)
        opt_path = tmp_path / "real_s2.tif"
        sar_path = tmp_path / "real_s1.tif"

        # Optical 4-band synthetic raster (64x64)
        y, x = np.mgrid[0:64, 0:64]
        base_opt = ((x + y) * 2.0 + 10.0).astype(np.float32)
        opt_data = np.stack([base_opt * (i + 1) for i in range(4)])
        with rasterio.open(
            opt_path, "w", driver="GTiff", height=64, width=64, count=4,
            dtype="float32", crs=RioCRS.from_string("EPSG:32632"), transform=transform,
        ) as dst:
            dst.write(opt_data)

        # SAR 2-band synthetic raster (64x64)
        base_sar = np.random.RandomState(42).normal(loc=0.2, scale=0.05, size=(2, 64, 64)).astype(np.float32)
        with rasterio.open(
            sar_path, "w", driver="GTiff", height=64, width=64, count=2,
            dtype="float32", crs=RioCRS.from_string("EPSG:32632"), transform=transform,
        ) as dst:
            dst.write(base_sar)

        # 2. Run REAL M3 execution pipeline
        config = OpticalSARConfig()
        m3_result = run_optical_sar_pipeline(
            optical_path=opt_path,
            sar_path=sar_path,
            config=config,
            run_inference=True,
        )

        assert m3_result is not None
        assert hasattr(m3_result, "to_dict")
        assert hasattr(m3_result, "to_gis_evidence")

        # 3. Direct handoff to M5 GIS layer (In-memory, no manual JSON copying)
        out_dir = OUTPUT_TEST_DIR / "real_m3_pipeline_test"
        m5_evidence = process_m3_result(m3_result, output_dir=out_dir)

        # 4. Verify M5 output and GIS preservation
        assert m5_evidence["change_detected"] is True
        assert (out_dir / "evidence.json").exists()
        assert (out_dir / "evidence.geojson").exists()
        assert (out_dir / "map.html").exists()

        # Dynamic CRS handling (EPSG:32632 preserved in metadata)
        assert m5_evidence["raster_metadata"]["crs"] == "EPSG:32632"
        assert m5_evidence["geographic_coordinates"] is not None

        # GeoJSON features reprojected to EPSG:4326 (WGS84 degrees)
        with open(out_dir / "evidence.geojson", "r", encoding="utf-8") as f:
            geojson = json.load(f)
            assert geojson["type"] == "FeatureCollection"
            assert len(geojson["features"]) >= 1
            poly_coords = geojson["features"][0]["geometry"]["coordinates"][0]
            for lon, lat in poly_coords:
                # Lon ~ 11°E, Lat ~ 48°N in Bavaria/UTM zone 32N
                assert 10.0 <= lon <= 13.0
                assert 47.0 <= lat <= 49.5

        # Area calculation in square meters and hectares (geodesic ellipsoid)
        assert m5_evidence["area"]["total_sq_meters"] > 0
        assert m5_evidence["area"]["total_hectares"] > 0

        # Registration quality & passed status faithfully preserved
        m3_meta = m5_evidence["m3_multimodal_metadata"]
        assert "registration_passed" in m3_meta
        assert m3_meta["registration_passed"] == m3_result.registration["passed"]
        assert m3_meta["registration_score"] == m3_result.registration["registration_score"]

        # Optical & SAR quality preserved
        assert "optical" in m3_meta
        assert "sar" in m3_meta
        assert "confidence" in m3_meta






