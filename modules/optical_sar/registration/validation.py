"""Registration validation metrics and composite quality assessment."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import sobel

from modules.optical_sar.config import OpticalData, SARData
from modules.optical_sar.registration.alignment import phase_correlation_shift

logger = logging.getLogger(__name__)


@dataclass
class RegistrationValidationResult:
    """Structured result of optical-to-SAR registration quality validation.

    Attributes:
        overlap_ratio: Proportion of shared valid pixels between modalities [0, 1].
        mutual_information: Normalized Mutual Information (NMI) [0, 1].
        structural_score: Cross-modal edge gradient correlation [0, 1].
        alignment_error: Estimated residual spatial displacement error in pixels.
        registration_score: Documented composite registration quality index [0, 1].
        passed: Boolean indicating whether alignment satisfies required thresholds.
        metrics: Detailed dictionary of computed metrics.
    """
    overlap_ratio: float
    mutual_information: float
    structural_score: float
    alignment_error: float
    registration_score: float
    passed: bool
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overlap_ratio": round(self.overlap_ratio, 4),
            "mutual_information": round(self.mutual_information, 4),
            "structural_score": round(self.structural_score, 4),
            "alignment_error": round(self.alignment_error, 2),
            "registration_score": round(self.registration_score, 4),
            "passed": self.passed,
            "metrics": self.metrics,
        }


def compute_normalized_mutual_information(
    img1: np.ndarray,
    img2: np.ndarray,
    valid_mask: np.ndarray,
    bins: int = 32,
    eps: float = 1e-12,
) -> float:
    """Calculate Normalized Mutual Information (NMI) over valid pixel overlap.

    Formula:
        NMI(X, Y) = 2 * I(X, Y) / (H(X) + H(Y))
        where I(X, Y) = H(X) + H(Y) - H(X, Y)

    Returns:
        NMI score in range [0, 1].
    """
    if not np.any(valid_mask):
        return 0.0

    v1 = img1[valid_mask]
    v2 = img2[valid_mask]

    # Compute 2D joint histogram
    hist_2d, _, _ = np.histogram2d(v1, v2, bins=bins)
    p_xy = hist_2d / np.sum(hist_2d)

    # Marginal distributions
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)

    # Shannon Entropies
    h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0] + eps))
    h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0] + eps))
    h_xy = -np.sum(p_xy[p_xy > 0] * np.log2(p_xy[p_xy > 0] + eps))

    if (h_x + h_y) < eps:
        return 1.0 if np.allclose(v1, v2) else 0.0

    nmi = 2.0 * (h_x + h_y - h_xy) / (h_x + h_y)
    return float(np.clip(nmi, 0.0, 1.0))


def compute_edge_structural_consistency(
    img1: np.ndarray,
    img2: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute structural edge consistency using Pearson correlation of Sobel gradient magnitudes.

    Cross-modal note:
        While raw radiometry differs fundamentally between optical reflection and radar backscatter,
        geometrical boundaries (coastlines, roads, field boundaries, urban perimeters) create
        coincident gradient edges in both modalities.

    Returns:
        Correlation score in range [0, 1].
    """
    if not np.any(valid_mask):
        return 0.0

    c1 = np.where(valid_mask, img1, 0.0)
    c2 = np.where(valid_mask, img2, 0.0)

    g1 = np.hypot(sobel(c1, axis=1), sobel(c1, axis=0))
    g2 = np.hypot(sobel(c2, axis=1), sobel(c2, axis=0))

    e1 = g1[valid_mask]
    e2 = g2[valid_mask]

    std1 = np.std(e1)
    std2 = np.std(e2)

    if std1 < eps or std2 < eps:
        return 0.0

    cov = np.cov(e1, e2)[0, 1]
    corr = cov / (std1 * std2 + eps)
    return float(max(0.0, min(1.0, corr)))


def validate_registration(
    optical_data: OpticalData,
    sar_data: SARData,
    validation_threshold: float = 0.70,
    min_overlap_ratio: float = 0.80,
) -> RegistrationValidationResult:
    """Validate spatial registration between Optical and SAR rasters.

    Exact Composite Registration Score Formula:
        score = 0.40 * overlap_ratio + 0.35 * mutual_information + 0.25 * structural_score - displacement_penalty
        where displacement_penalty = min(1.0, alignment_error / 10.0) * 0.20

    Args:
        optical_data: Preprocessed optical raster.
        sar_data: Reprojected/aligned SAR raster.
        validation_threshold: Minimum composite score required to pass (default 0.70).
        min_overlap_ratio: Minimum valid pixel overlap ratio required (default 0.80).

    Returns:
        RegistrationValidationResult containing quantitative metrics and pass/fail verdict.
    """
    if optical_data.shape[1:] != sar_data.shape[1:]:
        raise ValueError(
            f"Shape mismatch: Optical {optical_data.shape[1:]} vs SAR {sar_data.shape[1:]}. "
            f"SAR must be reprojected onto optical grid before validation."
        )

    # Valid pixel masks
    opt_valid = ~np.isnan(optical_data.data[0])
    sar_valid = ~np.isnan(sar_data.data[0])
    overlap_mask = opt_valid & sar_valid
    union_mask = opt_valid | sar_valid

    # 1. Overlap Ratio (Intersection over Union of valid pixels)
    total_union = int(np.count_nonzero(union_mask))
    overlap_pixels = int(np.count_nonzero(overlap_mask))
    overlap_ratio = float(overlap_pixels / total_union) if total_union > 0 else 0.0

    # Representative bands for cross-modal comparison
    ref_opt = optical_data.get_band("B08") if optical_data.has_band("B08") else optical_data.data[0]
    tgt_sar = sar_data.get_band("VV") if sar_data.has_band("VV") else sar_data.data[0]

    # 2. Normalized Mutual Information
    nmi = compute_normalized_mutual_information(ref_opt, tgt_sar, overlap_mask)

    # 3. Structural Edge Correlation
    struct_score = compute_edge_structural_consistency(ref_opt, tgt_sar, overlap_mask)

    # 4. Residual Alignment Displacement Error (via phase correlation)
    dx, dy, _ = phase_correlation_shift(ref_opt, tgt_sar, max_displacement=15)
    displacement_error = float(np.hypot(dx, dy))

    # 5. Composite Score Calculation
    disp_penalty = min(1.0, displacement_error / 10.0) * 0.20
    raw_score = 0.40 * overlap_ratio + 0.35 * nmi + 0.25 * struct_score - disp_penalty
    registration_score = float(np.clip(raw_score, 0.0, 1.0))

    passed = bool(registration_score >= validation_threshold and overlap_ratio >= min_overlap_ratio)

    metrics = {
        "overlap_pixels": overlap_pixels,
        "total_union_pixels": total_union,
        "residual_dx_px": dx,
        "residual_dy_px": dy,
        "displacement_penalty": round(disp_penalty, 4),
        "validation_threshold": validation_threshold,
        "min_overlap_ratio": min_overlap_ratio,
    }

    logger.info(
        f"Registration validation: score={registration_score:.3f} (passed={passed}), "
        f"overlap={overlap_ratio:.3f}, NMI={nmi:.3f}, struct={struct_score:.3f}, "
        f"disp_err={displacement_error:.2f}px"
    )

    return RegistrationValidationResult(
        overlap_ratio=overlap_ratio,
        mutual_information=nmi,
        structural_score=struct_score,
        alignment_error=displacement_error,
        registration_score=registration_score,
        passed=passed,
        metrics=metrics,
    )
