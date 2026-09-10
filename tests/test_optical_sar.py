"""Unit and integration test suite for M3 Optical + SAR module."""

from pathlib import Path
import affine
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
import torch

from modules.optical_sar.config import OpticalSARConfig, OpticalData, SARData
from modules.optical_sar.optical.loader import load_optical
from modules.optical_sar.optical.cloud_mask import cloud_mask_optical
from modules.optical_sar.optical.normalization import (
    normalize_min_max,
    normalize_percentile,
    normalize_optical,
)
from modules.optical_sar.optical.features import (
    compute_ndvi,
    compute_ndwi,
    compose_rgb,
    compute_optical_features,
)
from modules.optical_sar.sar.loader import load_sar
from modules.optical_sar.sar.calibration import linear_to_db, db_to_linear, calibrate_sar
from modules.optical_sar.sar.speckle import lee_filter, apply_speckle_filter
from modules.optical_sar.sar.terrain import apply_terrain_correction
from modules.optical_sar.sar.normalization import normalize_sar, normalize_sar_array
from modules.optical_sar.registration.reprojection import reproject_to_reference
from modules.optical_sar.registration.alignment import align_modalities, phase_correlation_shift
from modules.optical_sar.registration.validation import (
    validate_registration,
    compute_normalized_mutual_information,
    compute_edge_structural_consistency,
)
from modules.optical_sar.fusion.early_fusion import fuse_early
from modules.optical_sar.fusion.feature_fusion import FeatureFusionNetwork
from modules.optical_sar.fusion.model import OpticalSARModel
from modules.optical_sar.confidence.confidence import (
    calculate_multimodal_confidence,
    ConfidenceAssessment,
)
from modules.optical_sar.pipeline import run_optical_sar_pipeline


def _create_test_raster(
    file_path: Path,
    num_bands: int = 4,
    height: int = 64,
    width: int = 64,
    crs: str = "EPSG:32632",
    pixel_size: float = 10.0,
    nodata_val: float = -9999.0,
) -> Path:
    """Helper to generate self-contained GeoTIFF rasters with known spatial geometry."""
    transform = affine.Affine(pixel_size, 0.0, 500000.0, 0.0, -pixel_size, 5200000.0)

    # Generate synthetic spatial gradient pattern
    y, x = np.mgrid[0:height, 0:width]
    base = (x + y).astype(np.float32)

    data = np.empty((num_bands, height, width), dtype=np.float32)
    for b in range(num_bands):
        data[b] = base * (b + 1) + 10.0

    # Inject some nodata pixels in corners
    data[:, 0:2, 0:2] = nodata_val

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype="float32",
        crs=CRS.from_string(crs),
        transform=transform,
        nodata=nodata_val,
    ) as dst:
        dst.write(data)

    return file_path


# =====================================================================
# 1. OPTICAL TESTS
# =====================================================================

def test_optical_loader(tmp_path: Path):
    """Test loading optical GeoTIFF with band mapping and metadata extraction."""
    raster_path = _create_test_raster(tmp_path / "optical.tif", num_bands=4)
    band_mapping = {"B02": 1, "B03": 2, "B04": 3, "B08": 4}

    optical_data = load_optical(raster_path, band_mapping=band_mapping)

    assert optical_data.channels == 4
    assert optical_data.height == 64
    assert optical_data.width == 64
    assert optical_data.band_names == ["B02", "B03", "B04", "B08"]
    assert optical_data.crs is not None
    assert optical_data.valid_fraction > 0.90
    # Check that nodata pixels became NaN
    assert np.isnan(optical_data.data[0, 0, 0])


def test_optical_loader_missing_file():
    """Test that missing optical file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_optical("non_existent_file_path.tif")


def test_optical_cloud_masking(tmp_path: Path):
    """Test optical cloud mask application and status reporting."""
    raster_path = _create_test_raster(tmp_path / "optical.tif", num_bands=4)
    optical_data = load_optical(raster_path)

    # 1. When no mask is supplied: reports unavailable
    res_none = cloud_mask_optical(optical_data, cloud_mask=None)
    assert res_none.status == "unavailable"
    assert res_none.cloud_mask is None
    assert res_none.cloud_fraction is None

    # 2. When external numpy mask is supplied
    mask_arr = np.zeros((64, 64), dtype=bool)
    mask_arr[10:20, 10:20] = True  # 100 cloudy pixels

    res_mask = cloud_mask_optical(optical_data, cloud_mask=mask_arr)
    assert res_mask.status == "applied_external"
    assert res_mask.cloud_fraction == pytest.approx(100 / (64 * 64), rel=1e-3)
    assert np.isnan(res_mask.optical_data.data[0, 15, 15])


def test_optical_normalization():
    """Test min-max and percentile normalization routines."""
    arr = np.array([[[np.nan, 10.0], [20.0, 100.0]]], dtype=np.float32)

    # Min-max
    norm_mm = normalize_min_max(arr, clip=True)
    assert np.isnan(norm_mm[0, 0, 0])
    assert norm_mm[0, 0, 1] == 0.0
    assert norm_mm[0, 1, 1] == 1.0

    # Percentile
    norm_pct = normalize_percentile(arr, lower_percentile=10.0, upper_percentile=90.0)
    assert np.isnan(norm_pct[0, 0, 0])
    assert 0.0 <= np.nanmin(norm_pct) <= np.nanmax(norm_pct) <= 1.0


def test_optical_ndvi_and_features():
    """Test NDVI calculation, zero-division safety, and feature extraction."""
    red = np.array([[0.1, 0.2], [0.0, np.nan]], dtype=np.float32)
    nir = np.array([[0.5, 0.2], [0.0, 0.8]], dtype=np.float32)

    ndvi = compute_ndvi(nir, red)
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    assert ndvi[0, 0] == pytest.approx(0.6667, abs=1e-3)
    # (0.2 - 0.2) / 0.4 = 0.0
    assert ndvi[0, 1] == pytest.approx(0.0, abs=1e-3)
    # Zero division safe: (0 - 0) / (0 + 0 + eps) -> 0.0
    assert ndvi[1, 0] == pytest.approx(0.0, abs=1e-3)
    # NaN preservation
    assert np.isnan(ndvi[1, 1])


# =====================================================================
# 2. SAR TESTS
# =====================================================================

def test_sar_loader(tmp_path: Path):
    """Test SAR GeoTIFF loader with dual polarization VV/VH."""
    raster_path = _create_test_raster(tmp_path / "sar.tif", num_bands=2)
    sar_data = load_sar(raster_path, polarizations=["VV", "VH"])

    assert sar_data.channels == 2
    assert sar_data.polarizations == ["VV", "VH"]
    assert sar_data.valid_fraction > 0.90


def test_sar_calibration_and_db():
    """Test linear to dB backscatter conversion and inversion."""
    linear = np.array([[0.001, 0.01], [0.1, 1.0]], dtype=np.float32)

    # 10 * log10(0.01) = -20 dB, 10 * log10(1.0) = 0 dB
    db = linear_to_db(linear, min_db=-40.0, max_db=10.0)
    assert db[0, 1] == pytest.approx(-20.0, abs=1e-3)
    assert db[1, 1] == pytest.approx(0.0, abs=1e-3)

    # Roundtrip check
    reconstructed = db_to_linear(db)
    assert np.allclose(reconstructed, linear, atol=1e-4)


def test_sar_speckle_filtering():
    """Test Lee speckle filtering variance reduction."""
    rng = np.random.default_rng(42)
    clean = np.full((50, 50), 10.0, dtype=np.float32)
    # Add multiplicative speckle noise
    speckled = clean * rng.exponential(scale=1.0, size=(50, 50)).astype(np.float32)

    filtered = lee_filter(speckled, kernel_size=5)

    assert filtered.shape == speckled.shape
    # Speckle variance should be significantly reduced
    assert np.var(filtered) < np.var(speckled)


def test_sar_terrain_correction_adapter():
    """Test that terrain correction reports actual status when DEM is omitted."""
    data = np.ones((2, 32, 32), dtype=np.float32)
    sar = SARData(
        data=data,
        crs="EPSG:32632",
        transform=affine.Affine.identity(),
        resolution=(10.0, 10.0),
        bounds=(0.0, 0.0, 320.0, 320.0),
        polarizations=["VV", "VH"],
    )

    res = apply_terrain_correction(sar, dem_path=None)
    assert res.applied is False
    assert res.status == "dem_not_provided"
    assert sar.terrain_corrected is False


# =====================================================================
# 3. REGISTRATION TESTS
# =====================================================================

def test_geospatial_reprojection(tmp_path: Path):
    """Test reprojecting SAR raster with different resolution and dimensions onto optical grid."""
    opt_path = _create_test_raster(tmp_path / "opt.tif", num_bands=4, height=64, width=64, pixel_size=10.0)
    # SAR at coarser 20m resolution (32x32 pixels) covering same extent
    sar_path = _create_test_raster(tmp_path / "sar.tif", num_bands=2, height=32, width=32, pixel_size=20.0)

    optical = load_optical(opt_path)
    sar = load_sar(sar_path, polarizations=["VV", "VH"])

    sar_reproj = reproject_to_reference(sar, optical, resampling_method="bilinear")

    assert sar_reproj.shape == (2, 64, 64)
    assert sar_reproj.crs == optical.crs
    assert sar_reproj.transform == optical.transform
    assert sar_reproj.resolution == optical.resolution


def test_registration_validation(tmp_path: Path):
    """Test registration validation metrics and scoring."""
    opt_path = _create_test_raster(tmp_path / "opt.tif", num_bands=4, height=64, width=64)
    sar_path = _create_test_raster(tmp_path / "sar.tif", num_bands=2, height=64, width=64)

    optical = load_optical(opt_path)
    sar = load_sar(sar_path, polarizations=["VV", "VH"])

    validation = validate_registration(optical, sar)

    assert 0.0 <= validation.overlap_ratio <= 1.0
    assert 0.0 <= validation.mutual_information <= 1.0
    assert 0.0 <= validation.registration_score <= 1.0
    assert isinstance(validation.passed, bool)
    assert validation.overlap_ratio > 0.90


# =====================================================================
# 4. FUSION TESTS
# =====================================================================

def test_early_fusion(tmp_path: Path):
    """Test early channel stacking with variable channel compositions."""
    opt_path = _create_test_raster(tmp_path / "opt.tif", num_bands=4, height=64, width=64)
    sar_path = _create_test_raster(tmp_path / "sar.tif", num_bands=2, height=64, width=64)

    optical = load_optical(opt_path, band_mapping={"B02": 1, "B03": 2, "B04": 3, "B08": 4})
    sar = load_sar(sar_path, polarizations=["VV", "VH"])

    # 1. Full 6-channel fusion
    fused_6ch = fuse_early(optical, sar)
    assert fused_6ch.shape == (6, 64, 64)
    assert fused_6ch.channel_names == ["B02", "B03", "B04", "B08", "VV", "VH"]

    # 2. Dynamic 5-channel fusion (single VV polarization)
    fused_5ch = fuse_early(optical, sar, sar_polarizations=["VV"])
    assert fused_5ch.shape == (5, 64, 64)
    assert fused_5ch.channel_names == ["B02", "B03", "B04", "B08", "VV"]


def test_early_fusion_dimension_mismatch(tmp_path: Path):
    """Test that early fusion raises ValueError on spatial dimension mismatch."""
    opt_path = _create_test_raster(tmp_path / "opt.tif", num_bands=4, height=64, width=64)
    sar_path = _create_test_raster(tmp_path / "sar.tif", num_bands=2, height=32, width=32)

    optical = load_optical(opt_path)
    sar = load_sar(sar_path)

    with pytest.raises(ValueError, match="Spatial dimension mismatch"):
        fuse_early(optical, sar)


def test_feature_fusion_pytorch_model():
    """Test PyTorch two-stream model forward pass and shape integrity."""
    model = OpticalSARModel(
        fusion_type="feature",
        optical_channels=4,
        sar_channels=2,
        feature_dim=64,
        num_classes=3,
    )

    optical_tensor = torch.rand(2, 4, 32, 32)
    sar_tensor = torch.rand(2, 2, 32, 32)

    logits = model(optical_tensor, sar_tensor)
    assert logits.shape == (2, 3)

    inference_result = model.predict(optical_tensor[0], sar_tensor[0])
    assert inference_result.predicted_class in range(3)
    assert len(inference_result.probabilities) == 3
    assert 0.0 <= inference_result.model_confidence <= 1.0
    assert inference_result.status == "untrained_baseline"


# =====================================================================
# 5. CONFIDENCE TESTS
# =====================================================================

def test_confidence_scoring():
    """Test confidence bounds, threshold levels, and missing modality penalization."""
    # Complete modalities
    conf_high = calculate_multimodal_confidence(
        optical_data=OpticalData(
            data=np.ones((4, 10, 10)), crs=None, transform=None, resolution=(10, 10),
            bounds=(0, 0, 100, 100), valid_fraction=1.0, cloud_fraction=0.0
        ),
        sar_data=SARData(
            data=np.ones((2, 10, 10)), crs=None, transform=None, resolution=(10, 10),
            bounds=(0, 0, 100, 100), valid_fraction=1.0
        ),
        registration_result=type("MockReg", (), {"registration_score": 0.95})(),
        model_result=type("MockModel", (), {"model_confidence": 0.92})(),
    )

    assert 0.0 <= conf_high.score <= 1.0
    assert conf_high.level == "high"
    assert len(conf_high.missing_modalities) == 0

    # Missing SAR modality
    conf_missing_sar = calculate_multimodal_confidence(
        optical_data=OpticalData(
            data=np.ones((4, 10, 10)), crs=None, transform=None, resolution=(10, 10),
            bounds=(0, 0, 100, 100), valid_fraction=1.0, cloud_fraction=0.0
        ),
        sar_data=None,
        registration_result=None,
    )

    assert "sar" in conf_missing_sar.missing_modalities
    assert conf_missing_sar.score < conf_high.score
    assert "missing modalities" in conf_missing_sar.notes


# =====================================================================
# 6. END-TO-END PIPELINE TEST
# =====================================================================

def test_end_to_end_pipeline(tmp_path: Path):
    """Test running complete multimodal pipeline on synthetic satellite rasters."""
    opt_path = _create_test_raster(tmp_path / "s2_synthetic.tif", num_bands=4, height=64, width=64, pixel_size=10.0)
    sar_path = _create_test_raster(tmp_path / "s1_synthetic.tif", num_bands=2, height=32, width=32, pixel_size=20.0)

    config = OpticalSARConfig()
    result = run_optical_sar_pipeline(
        optical_path=opt_path,
        sar_path=sar_path,
        config=config,
        run_inference=True,
    )

    assert result.status in ("success", "partial")
    assert result.optical["bands"] is not None
    assert result.sar["polarizations"] is not None
    assert result.registration["crs"] is not None
    assert result.fusion["shape"][0] == 6  # 4 optical + 2 SAR
    assert 0.0 <= result.confidence["score"] <= 1.0

    # Check JSON export for M4
    dict_out = result.to_dict()
    assert "status" in dict_out
    assert "confidence" in dict_out

    # Check GIS evidence export for M5
    gis_evidence = result.to_gis_evidence()
    assert "crs" in gis_evidence
    assert "bounds" in gis_evidence
