"""M3 Optical + SAR Module for Sat Query.

Provides production-grade multimodal optical and SAR preprocessing, geospatial
reprojection, cross-modal alignment, registration validation, early & feature fusion,
deep learning models, and confidence scoring.
"""

from modules.optical_sar.config import (
    ConfidenceConfig,
    FusionConfig,
    OpticalConfig,
    OpticalData,
    OpticalSARConfig,
    RasterData,
    RegistrationConfig,
    SARConfig,
    SARData,
)
from modules.optical_sar.confidence.confidence import (
    ConfidenceAssessment,
    calculate_multimodal_confidence,
)
from modules.optical_sar.fusion.early_fusion import EarlyFusionResult, fuse_early
from modules.optical_sar.fusion.feature_fusion import (
    ClassificationHead,
    FeatureFusionNetwork,
    MultimodalFusionModule,
    OpticalEncoder,
    SAREncoder,
)
from modules.optical_sar.fusion.model import ModelInferenceResult, OpticalSARModel
from modules.optical_sar.optical.cloud_mask import CloudMaskResult, cloud_mask_optical
from modules.optical_sar.optical.features import (
    compose_rgb,
    compute_ndvi,
    compute_ndwi,
    compute_optical_features,
)
from modules.optical_sar.optical.loader import load_optical
from modules.optical_sar.optical.normalization import (
    normalize_min_max,
    normalize_optical,
    normalize_percentile,
)
from modules.optical_sar.pipeline import (
    OpticalSARPipelineResult,
    run_optical_sar_pipeline,
    set_seed,
)
from modules.optical_sar.registration.alignment import AlignmentResult, align_modalities
from modules.optical_sar.registration.reprojection import reproject_to_reference
from modules.optical_sar.registration.validation import (
    RegistrationValidationResult,
    validate_registration,
)
from modules.optical_sar.sar.calibration import (
    calibrate_sar,
    db_to_linear,
    linear_to_db,
)
from modules.optical_sar.sar.loader import load_sar
from modules.optical_sar.sar.normalization import normalize_sar
from modules.optical_sar.sar.speckle import apply_speckle_filter, lee_filter
from modules.optical_sar.sar.terrain import TerrainCorrectionResult, apply_terrain_correction

__all__ = [
    # Top-level pipeline
    "run_optical_sar_pipeline",
    "OpticalSARPipelineResult",
    "set_seed",
    # Config & data structures
    "RasterData",
    "OpticalData",
    "SARData",
    "OpticalSARConfig",
    "OpticalConfig",
    "SARConfig",
    "RegistrationConfig",
    "FusionConfig",
    "ConfidenceConfig",
    # Optical
    "load_optical",
    "cloud_mask_optical",
    "CloudMaskResult",
    "normalize_optical",
    "normalize_min_max",
    "normalize_percentile",
    "compute_optical_features",
    "compute_ndvi",
    "compute_ndwi",
    "compose_rgb",
    # SAR
    "load_sar",
    "calibrate_sar",
    "linear_to_db",
    "db_to_linear",
    "apply_speckle_filter",
    "lee_filter",
    "apply_terrain_correction",
    "TerrainCorrectionResult",
    "normalize_sar",
    # Registration
    "reproject_to_reference",
    "align_modalities",
    "AlignmentResult",
    "validate_registration",
    "RegistrationValidationResult",
    # Fusion
    "fuse_early",
    "EarlyFusionResult",
    "OpticalSARModel",
    "ModelInferenceResult",
    "OpticalEncoder",
    "SAREncoder",
    "MultimodalFusionModule",
    "ClassificationHead",
    "FeatureFusionNetwork",
    # Confidence
    "calculate_multimodal_confidence",
    "ConfidenceAssessment",
]
