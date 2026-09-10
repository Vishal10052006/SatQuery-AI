"""High-level multimodal processing pipeline orchestrating M3 Optical + SAR analysis."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from modules.optical_sar.config import (
    OpticalData,
    OpticalSARConfig,
    SARData,
)
from modules.optical_sar.confidence.confidence import (
    ConfidenceAssessment,
    calculate_multimodal_confidence,
)
from modules.optical_sar.fusion.early_fusion import EarlyFusionResult, fuse_early
from modules.optical_sar.fusion.model import ModelInferenceResult, OpticalSARModel
from modules.optical_sar.optical.cloud_mask import CloudMaskResult, cloud_mask_optical
from modules.optical_sar.optical.features import compute_optical_features
from modules.optical_sar.optical.loader import load_optical
from modules.optical_sar.optical.normalization import normalize_optical
from modules.optical_sar.registration.alignment import AlignmentResult, align_modalities
from modules.optical_sar.registration.reprojection import reproject_to_reference
from modules.optical_sar.registration.validation import (
    RegistrationValidationResult,
    validate_registration,
)
from modules.optical_sar.sar.calibration import calibrate_sar
from modules.optical_sar.sar.loader import load_sar
from modules.optical_sar.sar.normalization import normalize_sar
from modules.optical_sar.sar.speckle import apply_speckle_filter
from modules.optical_sar.sar.terrain import TerrainCorrectionResult, apply_terrain_correction

logger = logging.getLogger(__name__)


@dataclass
class OpticalSARPipelineResult:
    """Structured, production-grade output of the complete M3 Optical + SAR analysis pipeline.

    Designed for seamless downstream handoff:
    - M2 (Change & Grounding): Access registered rasters, features, and fusion arrays.
    - M4 (Agent): High-level classification, confidence scores, and categorical status.
    - M5 (GIS & Evidence): Geospatial bounds, CRS, resolution, and affine transform.
    """
    status: str
    optical: Dict[str, Any]
    sar: Dict[str, Any]
    registration: Dict[str, Any]
    fusion: Dict[str, Any]
    features: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence: Dict[str, Any]
    metadata: Dict[str, Any]

    # In-memory artifact references for programmatic consumption
    optical_data: Optional[OpticalData] = None
    registered_sar_data: Optional[SARData] = None
    early_fusion_result: Optional[EarlyFusionResult] = None
    registration_validation: Optional[RegistrationValidationResult] = None
    confidence_assessment: Optional[ConfidenceAssessment] = None
    model_inference: Optional[ModelInferenceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline output to clean, JSON-serializable dictionary for M4 agent and M6 API."""
        return {
            "status": self.status,
            "optical": self.optical,
            "sar": self.sar,
            "registration": self.registration,
            "fusion": self.fusion,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_gis_evidence(self) -> Dict[str, Any]:
        """Extract geospatial evidence and coordinate metadata for M5 GIS Module."""
        return {
            "crs": self.metadata.get("crs"),
            "bounds": self.metadata.get("bounds"),
            "transform": self.metadata.get("transform"),
            "resolution": self.metadata.get("resolution"),
            "spatial_shape": self.metadata.get("spatial_shape"),
            "registration_score": self.registration.get("registration_score"),
            "registration_passed": self.registration.get("passed"),
        }


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_optical_sar_pipeline(
    optical_path: Union[str, Path],
    sar_path: Union[str, Path],
    cloud_mask_path: Optional[Union[str, Path]] = None,
    dem_path: Optional[Union[str, Path]] = None,
    config: Optional[OpticalSARConfig] = None,
    model: Optional[OpticalSARModel] = None,
    run_inference: bool = True,
) -> OpticalSARPipelineResult:
    """Execute the end-to-end multimodal Optical + SAR analysis pipeline.

    Workflow:
        1. Load Optical multi-band GeoTIFF.
        2. Load SAR polarimetric GeoTIFF.
        3. Preprocess Optical: Cloud masking & percentile normalization.
        4. Preprocess SAR: Calibration, speckle filtering, terrain correction adapter, normalization.
        5. Geospatial Reprojection: Resample SAR directly onto Optical reference grid.
        6. Optional Fine Alignment: Edge-based phase correlation.
        7. Registration Quality Validation: Overlap, NMI, and structural consistency.
        8. Feature Extraction: NDVI, RGB composition, and SAR backscatter ratios.
        9. Early Multimodal Fusion: Concatenate aligned channels.
        10. Model Inference: PyTorch multimodal feature fusion forward pass.
        11. Multimodal Confidence Assessment: Weighted data and model quality scoring.
        12. Return structured OpticalSARPipelineResult.

    Args:
        optical_path: Filepath to optical GeoTIFF.
        sar_path: Filepath to SAR GeoTIFF.
        cloud_mask_path: Optional filepath to cloud/SCL mask GeoTIFF.
        dem_path: Optional filepath to Digital Elevation Model (DEM) for SAR terrain correction.
        config: OpticalSARConfig specifying processing parameters. Defaults to standard configuration.
        model: Optional pre-instantiated OpticalSARModel. If None and run_inference is True, a default model is instantiated.
        run_inference: Whether to execute deep learning model inference.

    Returns:
        OpticalSARPipelineResult dataclass containing structured metrics and in-memory tensors.

    Raises:
        FileNotFoundError: If optical or SAR files are missing.
        ValueError: If CRS is missing or unrecoverable geospatial errors occur.
    """
    cfg = config if config is not None else OpticalSARConfig()
    set_seed(cfg.random_seed)

    logger.info(f"Initiating M3 Optical+SAR pipeline with optical={optical_path}, sar={sar_path}")

    # Step 1: Load Optical Data
    optical_raw = load_optical(
        path=optical_path,
        band_mapping=cfg.optical.band_mapping if hasattr(cfg.optical, "band_mapping") else None,
    )

    # Step 2: Load SAR Data
    sar_raw = load_sar(
        path=sar_path,
        polarizations=cfg.sar.polarizations,
    )

    # Step 3: Optical Preprocessing
    # 3a. Cloud masking
    cloud_result: CloudMaskResult = cloud_mask_optical(
        optical_data=optical_raw,
        cloud_mask=cloud_mask_path,
    )
    optical_masked = cloud_result.optical_data

    # 3b. Optical normalization
    optical_norm = normalize_optical(
        optical_data=optical_masked,
        method=cfg.optical.normalization_method,
        percentile_bounds=cfg.optical.percentile_bounds,
        clip=cfg.optical.clip_output,
    )

    # Step 4: SAR Preprocessing
    # 4a. Calibration & dB conversion
    sar_cal = calibrate_sar(
        sar_data=sar_raw,
        is_already_calibrated=cfg.sar.is_already_calibrated,
        calibration_type=cfg.sar.calibration_type,
        to_db=cfg.sar.to_db,
        min_db=cfg.sar.min_db,
        max_db=cfg.sar.max_db,
    )

    # 4b. Speckle filtering
    sar_filtered_data = apply_speckle_filter(
        image=sar_cal.data,
        method=cfg.sar.speckle_filter_method,
        kernel_size=cfg.sar.speckle_kernel_size,
    )
    sar_filtered = SARData(
        data=sar_filtered_data,
        crs=sar_cal.crs,
        transform=sar_cal.transform,
        resolution=sar_cal.resolution,
        bounds=sar_cal.bounds,
        nodata=sar_cal.nodata,
        band_names=list(sar_cal.band_names),
        polarizations=list(sar_cal.polarizations),
        metadata=dict(sar_cal.metadata),
        valid_fraction=sar_cal.valid_fraction,
        is_calibrated=sar_cal.is_calibrated,
        is_db=sar_cal.is_db,
        terrain_corrected=sar_cal.terrain_corrected,
    )

    # 4c. Terrain correction adapter
    terrain_result: TerrainCorrectionResult = apply_terrain_correction(
        sar_data=sar_filtered,
        dem_path=dem_path,
    )
    sar_terrain = terrain_result.sar_data

    # 4d. SAR normalization
    sar_norm = normalize_sar(
        sar_data=sar_terrain,
        method=cfg.sar.normalization_method,
        percentile_bounds=cfg.sar.percentile_bounds,
    )

    # Step 5: Geospatial Reprojection (SAR onto Optical Grid)
    sar_reprojected = reproject_to_reference(
        source=sar_norm,
        reference=optical_norm,
        resampling_method=cfg.registration.resampling_method,
    )

    # Step 6: Fine Cross-Modal Alignment (Optional)
    align_result: AlignmentResult = align_modalities(
        optical=optical_norm,
        sar=sar_reprojected,
        method=cfg.registration.fine_alignment_method,
        max_displacement=cfg.registration.max_displacement,
    )
    sar_aligned = align_result.aligned_sar

    # Step 7: Registration Quality Validation
    reg_validation: RegistrationValidationResult = validate_registration(
        optical_data=optical_norm,
        sar_data=sar_aligned,
        validation_threshold=cfg.registration.validation_threshold,
        min_overlap_ratio=cfg.registration.min_overlap_ratio,
    )

    # Step 8: Feature Extraction
    opt_features = compute_optical_features(optical_norm)
    sar_features: Dict[str, np.ndarray] = {}
    if sar_aligned.has_band("VV") and sar_aligned.has_band("VH"):
        vv = sar_aligned.get_band("VV")
        vh = sar_aligned.get_band("VH")
        # Polarization cross-ratio VH / (VV + eps)
        sar_features["cross_ratio"] = vh / (vv + 1e-6)

    # Step 9: Early Multimodal Fusion
    early_fusion: EarlyFusionResult = fuse_early(
        optical_data=optical_norm,
        sar_data=sar_aligned,
    )

    # Step 10: Multimodal Deep Learning Model Inference
    model_inference: Optional[ModelInferenceResult] = None
    if run_inference:
        try:
            if model is None:
                weights_path = Path(__file__).resolve().parent / "weights" / "m3_optical_sar_model.pth"

                model = OpticalSARModel(
                    fusion_type=cfg.fusion.method,
                    optical_channels=optical_norm.channels,
                    sar_channels=sar_aligned.channels,
                    feature_dim=cfg.fusion.feature_dim,
                    num_classes=cfg.fusion.num_classes,
                    weights_path=str(weights_path) if weights_path.exists() else None,
                )
            model_inference = model.predict(optical_norm.data, sar_aligned.data)
        except Exception as e:
            logger.warning(f"Multimodal model inference encountered error: {e}")
            model_inference = ModelInferenceResult(
                predicted_class=None,
                probabilities=None,
                logits=None,
                model_confidence=None,
                fusion_type=cfg.fusion.method,
                status="failed",
                notes=f"Inference error: {str(e)}",
            )
    else:
        model_inference = ModelInferenceResult(
            predicted_class=None,
            probabilities=None,
            logits=None,
            model_confidence=None,
            fusion_type=cfg.fusion.method,
            status="not_available",
            notes="Model inference was explicitly disabled (run_inference=False).",
        )

    # Step 11: Multimodal Confidence Assessment
    confidence_assessment: ConfidenceAssessment = calculate_multimodal_confidence(
        optical_data=optical_norm,
        sar_data=sar_aligned,
        registration_result=reg_validation,
        model_result=model_inference,
        weights=cfg.confidence.weights,
        thresholds=cfg.confidence.thresholds,
    )

    # Step 12: Assemble Structured Result
    pipeline_status = "success" if reg_validation.passed else "partial"

    result = OpticalSARPipelineResult(
        status=pipeline_status,
        optical={
            "bands": optical_norm.band_names,
            "cloud_fraction": optical_norm.cloud_fraction,
            "valid_fraction": optical_norm.valid_fraction,
            "cloud_mask_status": cloud_result.status,
            "shape": list(optical_norm.shape),
        },
        sar={
            "polarizations": sar_aligned.polarizations,
            "valid_fraction": sar_aligned.valid_fraction,
            "is_calibrated": sar_aligned.is_calibrated,
            "is_db": sar_aligned.is_db,
            "terrain_corrected": sar_aligned.terrain_corrected,
            "terrain_correction_status": terrain_result.status,
            "shape": list(sar_aligned.shape),
        },
        registration={
            "crs": str(optical_norm.crs),
            "resolution": list(optical_norm.resolution),
            "overlap_ratio": round(reg_validation.overlap_ratio, 4),
            "mutual_information": round(reg_validation.mutual_information, 4),
            "structural_score": round(reg_validation.structural_score, 4),
            "alignment_error": round(reg_validation.alignment_error, 2),
            "registration_score": round(reg_validation.registration_score, 4),
            "passed": reg_validation.passed,
            "fine_alignment_applied": align_result.status,
        },
        fusion={
            "method": "early_channel_stacking",
            "channels": early_fusion.channel_names,
            "shape": list(early_fusion.shape),
        },
        features={
            "optical_features": list(opt_features.keys()),
            "sar_features": list(sar_features.keys()),
        },
        prediction=model_inference.to_dict(),
        confidence=confidence_assessment.to_dict(),
        metadata={
            "crs": str(optical_norm.crs),
            "bounds": list(optical_norm.bounds),
            "transform": [float(x) for x in list(optical_norm.transform)[:6]],
            "resolution": list(optical_norm.resolution),
            "spatial_shape": [optical_norm.height, optical_norm.width],
        },
        optical_data=optical_norm,
        registered_sar_data=sar_aligned,
        early_fusion_result=early_fusion,
        registration_validation=reg_validation,
        confidence_assessment=confidence_assessment,
        model_inference=model_inference,
    )

    logger.info(
        f"Pipeline completed: status={pipeline_status}, reg_score={reg_validation.registration_score:.3f}, "
        f"confidence={confidence_assessment.score:.3f} ({confidence_assessment.level})"
    )

    return result
