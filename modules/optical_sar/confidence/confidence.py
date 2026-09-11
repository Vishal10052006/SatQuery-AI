"""Multimodal confidence evaluation combining optical, SAR, registration, and model metrics."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from modules.optical_sar.config import OpticalData, SARData
from modules.optical_sar.fusion.model import ModelInferenceResult
from modules.optical_sar.registration.validation import RegistrationValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceAssessment:
    """Operational confidence heuristic, not calibrated probabilistic uncertainty."""
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
            "components": {k: round(v, 4) if v is not None else None for k, v in self.components.items()},
            "missing_modalities": self.missing_modalities,
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "notes": self.notes,
        }


def compute_optical_quality(optical_data: OpticalData) -> float:
    """Score optical data quality conservatively when cloud information is unavailable."""
    valid_frac = float(np.clip(optical_data.valid_fraction if optical_data.valid_fraction is not None else 0.0, 0.0, 1.0))
    if optical_data.cloud_fraction is None:
        # Absence of a cloud mask is uncertainty, not evidence of a clear scene.
        return 0.75 * valid_frac
    cloud_frac = float(np.clip(optical_data.cloud_fraction, 0.0, 1.0))
    return float(np.clip(valid_frac * (1.0 - cloud_frac), 0.0, 1.0))


def compute_sar_quality(sar_data: SARData) -> float:
    """Compute SAR quality from valid pixel coverage."""
    valid_frac = sar_data.valid_fraction if sar_data.valid_fraction is not None else 0.0
    return float(np.clip(valid_frac, 0.0, 1.0))


def calculate_multimodal_confidence(
    optical_data: Optional[OpticalData] = None,
    sar_data: Optional[SARData] = None,
    registration_result: Optional[RegistrationValidationResult] = None,
    model_result: Optional[ModelInferenceResult] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> ConfidenceAssessment:
    """Calculate an operational quality heuristic from available evidence."""
    default_weights = {
        "optical_quality": 0.25,
        "sar_quality": 0.20,
        "registration_quality": 0.20,
        "model_confidence": 0.35,
    }
    applied_weights = dict(weights) if weights is not None else default_weights
    thresh = dict(thresholds) if thresholds is not None else {"high": 0.80, "medium": 0.50}

    components: Dict[str, Optional[float]] = {}
    missing: List[str] = []

    if optical_data is not None:
        components["optical_quality"] = compute_optical_quality(optical_data)
        if optical_data.cloud_fraction is None:
            missing.append("cloud_mask")
    else:
        components["optical_quality"] = None
        missing.append("optical")

    if sar_data is not None:
        components["sar_quality"] = compute_sar_quality(sar_data)
    else:
        components["sar_quality"] = None
        missing.append("sar")

    if registration_result is not None:
        components["registration_quality"] = float(np.clip(registration_result.registration_score, 0.0, 1.0))
    else:
        components["registration_quality"] = None
        missing.append("registration")

    if model_result is not None and model_result.model_confidence is not None:
        components["model_confidence"] = float(np.clip(model_result.model_confidence, 0.0, 1.0))
    else:
        components["model_confidence"] = None
        missing.append("model_inference")

    available_keys = [k for k, v in components.items() if v is not None]
    total_avail_weight = sum(max(0.0, applied_weights.get(k, 0.0)) for k in available_keys)

    if not available_keys or total_avail_weight <= 0.0:
        return ConfidenceAssessment(
            score=0.0, level="low", components=components, missing_modalities=missing,
            weights_used={}, notes="No valid evidence components available for confidence calculation.",
        )

    normalized_weights = {k: max(0.0, applied_weights.get(k, 0.0)) / total_avail_weight for k in available_keys}
    weighted_sum = sum(normalized_weights[k] * float(components[k]) for k in available_keys if components[k] is not None)

    notes_list: List[str] = []
    if "cloud_mask" in missing:
        notes_list.append("Cloud mask unavailable; optical quality was conservatively discounted.")
    if "model_inference" in missing:
        notes_list.append("Trained model inference unavailable; score reflects data and registration quality only.")
    if "optical" in missing or "sar" in missing:
        weighted_sum *= 0.50
        notes_list.append("A primary modality is missing; applied a 50% multimodal evidence penalty.")

    final_score = float(np.clip(weighted_sum, 0.0, 1.0))
    if final_score >= thresh["high"]:
        level = "high"
    elif final_score >= thresh["medium"]:
        level = "medium"
    else:
        level = "low"

    return ConfidenceAssessment(
        score=final_score,
        level=level,
        components=components,
        missing_modalities=missing,
        weights_used=normalized_weights,
        notes=" ".join(notes_list) if notes_list else "Operational multimodal confidence calculated from available evidence.",
    )
