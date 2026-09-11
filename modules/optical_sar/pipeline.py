"""High-level multimodal processing pipeline orchestrating M3 Optical + SAR analysis."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch

from modules.optical_sar.config import OpticalData, OpticalSARConfig, SARData
from modules.optical_sar.confidence.confidence import ConfidenceAssessment, calculate_multimodal_confidence
from modules.optical_sar.fusion.early_fusion import EarlyFusionResult, fuse_early
from modules.optical_sar.fusion.model import ModelInferenceResult, OpticalSARModel
from modules.optical_sar.optical.cloud_mask import CloudMaskResult, cloud_mask_optical
from modules.optical_sar.optical.features import compute_optical_features
from modules.optical_sar.optical.loader import load_optical
from modules.optical_sar.optical.normalization import normalize_optical
from modules.optical_sar.registration.alignment import AlignmentResult, align_modalities
from modules.optical_sar.registration.reprojection import reproject_to_reference
from modules.optical_sar.registration.validation import RegistrationValidationResult, validate_registration
from modules.optical_sar.sar.calibration import calibrate_sar
from modules.optical_sar.sar.loader import load_sar
from modules.optical_sar.sar.normalization import normalize_sar
from modules.optical_sar.sar.speckle import apply_speckle_filter
from modules.optical_sar.sar.terrain import TerrainCorrectionResult, apply_terrain_correction

logger = logging.getLogger(__name__)


@dataclass
class OpticalSARPipelineResult:
    """Structured output for M4/M5/M6 handoff."""
    status: str
    optical: Dict[str, Any]
    sar: Dict[str, Any]
    registration: Dict[str, Any]
    fusion: Dict[str, Any]
    features: Dict[str, Any]
    prediction: Dict[str, Any]
    confidence: Dict[str, Any]
    metadata: Dict[str, Any]
    optical_data: Optional[OpticalData] = None
    registered_sar_data: Optional[SARData] = None
    early_fusion_result: Optional[EarlyFusionResult] = None
    registration_validation: Optional[RegistrationValidationResult] = None
    confidence_assessment: Optional[ConfidenceAssessment] = None
    model_inference: Optional[ModelInferenceResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "optical": self.optical,
            "sar": self.sar,
            "registration": self.registration,
            "fusion": self.fusion,
            "features": self.features,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def to_gis_evidence(self) -> Dict[str, Any]:
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
    """Execute the end-to-end Optical + SAR pipeline.

    SAR preprocessing keeps Lee filtering in linear intensity. Radiometric conversion
    to dB is performed before normalization, and production model inference requires
    a real weights file; an absent checkpoint is never treated as a scientific result.
    """
    cfg = config if config is not None else OpticalSARConfig()
    set_seed(cfg.random_seed)

    optical_raw = load_optical(
        path=optical_path,
        band_mapping=cfg.optical.band_mapping if hasattr(cfg.optical, "band_mapping") else None,
    )
    sar_raw = load_sar(path=sar_path, polarizations=cfg.sar.polarizations)

    cloud_result: CloudMaskResult = cloud_mask_optical(optical_data=optical_raw, cloud_mask=cloud_mask_path)
    optical_norm = normalize_optical(
        optical_data=cloud_result.optical_data,
        method=cfg.optical.normalization_method,
        percentile_bounds=cfg.optical.percentile_bounds,
        clip=cfg.optical.clip_output,
    )

    sar_cal = calibrate_sar(
        sar_data=sar_raw,
        is_already_calibrated=cfg.sar.is_already_calibrated,
        calibration_type=cfg.sar.calibration_type,
        to_db=cfg.sar.to_db,
        min_db=cfg.sar.min_db,
        max_db=cfg.sar.max_db,
    )

    sar_filtered_data = apply_speckle_filter(
        image=sar_cal.data,
        method=cfg.sar.speckle_filter_method,
        kernel_size=cfg.sar.speckle_kernel_size,
        is_db=sar_cal.is_db,
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

    terrain_result: TerrainCorrectionResult = apply_terrain_correction(sar_data=sar_filtered, dem_path=dem_path)
    sar_norm = normalize_sar(
        sar_data=terrain_result.sar_data,
        method=cfg.sar.normalization_method,
        percentile_bounds=cfg.sar.percentile_bounds,
    )

    sar_reprojected = reproject_to_reference(
        source=sar_norm,
        reference=optical_norm,
        resampling_method=cfg.registration.resampling_method,
    )
    align_result: AlignmentResult = align_modalities(
        optical=optical_norm,
        sar=sar_reprojected,
        method=cfg.registration.fine_alignment_method,
        max_displacement=cfg.registration.max_displacement,
    )
    sar_aligned = align_result.aligned_sar

    reg_validation = validate_registration(
        optical_data=optical_norm,
        sar_data=sar_aligned,
        validation_threshold=cfg.registration.validation_threshold,
        min_overlap_ratio=cfg.registration.min_overlap_ratio,
    )

    opt_features = compute_optical_features(optical_norm)
    sar_features: Dict[str, np.ndarray] = {}
    if sar_aligned.has_band("VV") and sar_aligned.has_band("VH"):
        vv = sar_aligned.get_band("VV")
        vh = sar_aligned.get_band("VH")
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_ratio = vh / (vv + 1e-6)
        cross_ratio[~np.isfinite(cross_ratio)] = np.nan
        sar_features["cross_ratio"] = cross_ratio

    early_fusion = fuse_early(optical_data=optical_norm, sar_data=sar_aligned)

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
                    weights_path=str(weights_path),
                )
            model_inference = model.predict(optical_norm.data, sar_aligned.data)
        except Exception as e:
            logger.warning("Multimodal model inference encountered error: %s", e)
            model_inference = ModelInferenceResult(
                predicted_class=None, probabilities=None, logits=None,
                model_confidence=None, fusion_type=cfg.fusion.method,
                status="failed", notes=f"Inference error: {e}",
            )
    else:
        model_inference = ModelInferenceResult(
            predicted_class=None, probabilities=None, logits=None,
            model_confidence=None, fusion_type=cfg.fusion.method,
            status="not_available", notes="Model inference was explicitly disabled (run_inference=False).",
        )

    confidence_assessment = calculate_multimodal_confidence(
        optical_data=optical_norm,
        sar_data=sar_aligned,
        registration_result=reg_validation,
        model_result=model_inference,
        weights=cfg.confidence.weights,
        thresholds=cfg.confidence.thresholds,
    )

    if not reg_validation.passed:
        pipeline_status = "partial"
    elif model_inference.status == "success":
        pipeline_status = "success"
    else:
        pipeline_status = "partial"

    return OpticalSARPipelineResult(
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
