"""
Unit and Integration Test Suite for SatQuery-AI Module 5 (M5):
GIS, Geospatial Processing and Evidence Generation.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import shapely.geometry
from shapely.geometry import Polygon

from geospatial.metadata import read_geotiff_metadata
from geospatial.coordinates import pixel_to_geo, pixel_bbox_to_geo_bbox
from geospatial.polygons import mask_to_polygons, bbox_to_polygon, load_mask
from geospatial.area import calculate_polygon_area, calculate_total_area
from geospatial.visualization import generate_folium_map
from geospatial.evidence import generate_geojson, generate_evidence_json
from geospatial.pipeline import run_geospatial_pipeline
from geospatial.schema import M2M3Payload, EvidenceOutput
from geospatial.integration import process_m2_m3_result, process_m2_detector_output
from geospatial.m3_adapter import (
    export_layer_to_geotiff,
    export_all_layers_to_geotiff,
    process_m3_evidence_to_m5,
    process_m3_result,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_MOCK_DIR = BASE_DIR / "data" / "mock"
SAMPLE_GEOTIFF = DATA_MOCK_DIR / "sample.tif"
SAMPLE_MASK = DATA_MOCK_DIR / "change_mask.npy"
M2_RESULT_JSON = DATA_MOCK_DIR / "m2_result.json"
M2_GEOREF_JSON = DATA_MOCK_DIR / "m2_georef_output.json"
M2_NON_GEOREF_JSON = DATA_MOCK_DIR / "m2_non_georef_output.json"
M4_SPECIALIST_JSON = DATA_MOCK_DIR / "m4_specialist_result.json"
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
            min_lon, min_lat, max_lon, max_lat = poly.bounds
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
            process_m2_m3_result(12345)

    def test_14_export_layer_to_geotiff_2d_and_3d(self):
        """Verify exporting 2D (NDVI) and 3D (Multispectral/SAR) arrays to GeoTIFF."""
        from rasterio.transform import from_origin
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



