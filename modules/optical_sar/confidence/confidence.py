"""Multimodal confidence evaluation combining optical, SAR, registration, and model metrics."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from modules.optical_sar.config import OpticalData, SARData
from modules.optical_sar.fusion.model import ModelInferenceResult
from modules.optical_sar.registration.validation import RegistrationValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceAssessment:
    """Structured container for multimodal confidence scoring.

    Scientific Disclaimer:
        This confidence score is an engineering heuristic weighting data validity,
        cross-modal alignment quality, and neural prediction margins. It represents
        an operational quality baseline rather than formal calibrated Bayesian uncertainty.

    Attributes:
        score: Composite confidence score normalized between 0.0 and 1.0.
        level: Categorical quality descriptor ('high', 'medium', 'low').
        components: Dictionary of individual quality components.
        missing_modalities: List of any missing modalities or omitted inference steps.
        weights_used: Dictionary of effective weights applied to available components.
        notes: Contextual explanation of the confidence computation.
    """
    score: float
    level: str
    components: Dict[str, Optional[float]]
    missing_modalities: List[str]
    weights_used: Dict[str, float]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "components": {
                k: round(v, 4) if v is not None else None
                for k, v in self.components.items()
            },
            "missing_modalities": self.missing_modalities,
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "notes": self.notes,
        }


def compute_optical_quality(optical_data: OpticalData) -> float:
    """Compute data quality score for optical imagery based on validity and cloud cover."""
    valid_frac = optical_data.valid_fraction if optical_data.valid_fraction is not None else 1.0
    cloud_frac = optical_data.cloud_fraction if optical_data.cloud_fraction is not None else 0.0

    # Optical quality decreases with clouds and nodata
    clear_sky_frac = max(0.0, 1.0 - cloud_frac)
    quality = float(np.clip(valid_frac * clear_sky_frac, 0.0, 1.0))
    return quality


def compute_sar_quality(sar_data: SARData) -> float:
    """Compute data quality score for SAR imagery based on valid pixel coverage."""
    valid_frac = sar_data.valid_fraction if sar_data.valid_fraction is not None else 1.0
    return float(np.clip(valid_frac, 0.0, 1.0))


def calculate_multimodal_confidence(
    optical_data: Optional[OpticalData] = None,
    sar_data: Optional[SARData] = None,
    registration_result: Optional[RegistrationValidationResult] = None,
    model_result: Optional[ModelInferenceResult] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> ConfidenceAssessment:
    """Calculate composite multimodal confidence score.

    Default Weights:
        optical_quality:      0.25
        sar_quality:          0.20
        registration_quality: 0.20
        model_confidence:     0.35

    Default Thresholds:
        high:   >= 0.80
        medium: >= 0.50
        low:    <  0.50

    Args:
        optical_data: Processed OpticalData container.
        sar_data: Processed SARData container.
        registration_result: Result from registration validation.
        model_result: Result from model prediction.
        weights: Custom weighting dictionary.
        thresholds: Custom dictionary defining 'high' and 'medium' thresholds.

    Returns:
        ConfidenceAssessment containing final score, rating level, and component breakdown.
    """
    default_weights = {
        "optical_quality": 0.25,
        "sar_quality": 0.20,
        "registration_quality": 0.20,
        "model_confidence": 0.35,
    }
    applied_weights = dict(weights) if weights is not None else default_weights

    default_thresholds = {
        "high": 0.80,
        "medium": 0.50,
    }
    thresh = dict(thresholds) if thresholds is not None else default_thresholds

    components: Dict[str, Optional[float]] = {}
    missing: List[str] = []

    # 1. Optical Quality
    if optical_data is not None:
        components["optical_quality"] = compute_optical_quality(optical_data)
    else:
        components["optical_quality"] = None
        missing.append("optical")

    # 2. SAR Quality
    if sar_data is not None:
        components["sar_quality"] = compute_sar_quality(sar_data)
    else:
        components["sar_quality"] = None
        missing.append("sar")

    # 3. Registration Quality
    if registration_result is not None:
        components["registration_quality"] = registration_result.registration_score
    else:
        components["registration_quality"] = None
        missing.append("registration")

    # 4. Model Confidence
    if model_result is not None and model_result.model_confidence is not None:
        components["model_confidence"] = model_result.model_confidence
    else:
        components["model_confidence"] = None
        missing.append("model_inference")

    # Dynamic Weight Normalization across available components
    available_keys = [k for k, v in components.items() if v is not None]
    total_avail_weight = sum(applied_weights.get(k, 0.0) for k in available_keys)

    notes_list: List[str] = []

    if not available_keys or total_avail_weight <= 0.0:
        return ConfidenceAssessment(
            score=0.0,
            level="low",
            components=components,
            missing_modalities=missing,
            weights_used={},
            notes="No valid data or components available for confidence calculation.",
        )

    # Compute normalized weighted sum
    normalized_weights: Dict[str, float] = {}
    weighted_sum = 0.0
    for k in available_keys:
        w_norm = applied_weights.get(k, 0.0) / total_avail_weight
        normalized_weights[k] = w_norm
        val = components[k]
        if val is not None:
            weighted_sum += w_norm * val

    # Apply penalty if primary modalities (optical or SAR) are missing
    if "optical" in missing or "sar" in missing:
        penalty_factor = 0.50
        weighted_sum *= penalty_factor
        notes_list.append(
            f"Multimodal confidence could not be fully calculated due to missing modalities ({missing}). "
            f"Applied 50% single-modality confidence reduction."
        )

    if "model_inference" in missing:
        notes_list.append("Model inference was omitted; confidence reflects data and registration quality only.")

    final_score = float(np.clip(weighted_sum, 0.0, 1.0))

    # Categorize into High / Medium / Low
    if final_score >= thresh["high"]:
        level = "high"
    elif final_score >= thresh["medium"]:
        level = "medium"
    else:
        level = "low"

    full_notes = " ".join(notes_list) if notes_list else "Full multimodal confidence calculated successfully."

    logger.info(f"Calculated confidence: score={final_score:.3f}, level='{level}', missing={missing}")

    return ConfidenceAssessment(
        score=final_score,
        level=level,
        components=components,
        missing_modalities=missing,
        weights_used=normalized_weights,
        notes=full_notes,
    )
