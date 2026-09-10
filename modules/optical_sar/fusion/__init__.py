"""Multimodal Optical + SAR Fusion package."""

from modules.optical_sar.fusion.early_fusion import fuse_early, EarlyFusionResult
from modules.optical_sar.fusion.feature_fusion import (
    OpticalEncoder,
    SAREncoder,
    MultimodalFusionModule,
    ClassificationHead,
    FeatureFusionNetwork,
)
from modules.optical_sar.fusion.model import OpticalSARModel, ModelInferenceResult

__all__ = [
    "fuse_early",
    "EarlyFusionResult",
    "OpticalEncoder",
    "SAREncoder",
    "MultimodalFusionModule",
    "ClassificationHead",
    "FeatureFusionNetwork",
    "OpticalSARModel",
    "ModelInferenceResult",
]
