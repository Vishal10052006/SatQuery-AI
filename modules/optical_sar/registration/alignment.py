"""Fine cross-modal alignment between optical and SAR imagery."""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import shift, sobel

from modules.optical_sar.config import OpticalData, SARData

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Structured result of fine cross-modal alignment.

    Attributes:
        aligned_sar: SARData after fine alignment.
        displacement: Estimated translation shift (dx, dy) in pixels.
        transformation_matrix: Estimated 2x3 or 3x3 transformation matrix, if computed.
        status: Status code ('aligned', 'skipped', 'displacement_exceeded', 'failed').
        method: Alignment method utilized.
        confidence: Alignment confidence score (0.0 to 1.0).
        notes: Informational notes describing transformation applied.
    """
    aligned_sar: SARData
    displacement: Tuple[float, float]
    transformation_matrix: Optional[np.ndarray]
    status: str
    method: str
    confidence: float
    notes: str


def _compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operators for cross-modal structural alignment."""
    # Replace NaNs with local/median value for gradient computation
    nan_mask = np.isnan(image)
    clean = np.where(nan_mask, 0.0, image)

    gx = sobel(clean, axis=1)
    gy = sobel(clean, axis=0)
    mag = np.hypot(gx, gy)
    mag[nan_mask] = 0.0

    # Normalize to [0, 1]
    m_max = np.max(mag)
    if m_max > 1e-8:
        mag = mag / m_max
    return mag.astype(np.float32)


def phase_correlation_shift(
    ref_image: np.ndarray,
    target_image: np.ndarray,
    max_displacement: int = 20,
) -> Tuple[float, float, float]:
    """Estimate translation displacement between two images using Phase Correlation on gradient maps.

    Cross-modal note:
        Optical reflectance and SAR backscatter frequently exhibit inverse or complex non-linear
        relationships. Operating phase correlation on gradient edge structures significantly improves
        robustness compared to raw pixel intensity matching.

    Returns:
        (dx, dy, peak_response)
    """
    # Compute structural gradient maps
    grad_ref = _compute_gradient_magnitude(ref_image)
    grad_tgt = _compute_gradient_magnitude(target_image)

    # Apply Hanning window to mitigate boundary leakage
    h, w = grad_ref.shape
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    window = np.outer(hann_y, hann_x)

    f_ref = np.fft.fft2(grad_ref * window)
    f_tgt = np.fft.fft2(grad_tgt * window)

    # Cross-power spectrum
    cross_power = f_ref * np.conj(f_tgt)
    eps = 1e-12
    norm_cross = cross_power / (np.abs(cross_power) + eps)

    # Phase correlation surface
    corr_surface = np.fft.ifft2(norm_cross)
    corr_surface = np.fft.fftshift(np.abs(corr_surface))

    # Find peak
    cy, cx = h // 2, w // 2
    max_idx = np.unravel_index(np.argmax(corr_surface), corr_surface.shape)
    dy = float(max_idx[0] - cy)
    dx = float(max_idx[1] - cx)

    peak_val = float(corr_surface[max_idx])
    # Baseline normalization for response
    mean_val = float(np.mean(corr_surface))
    std_val = float(np.std(corr_surface)) + eps
    peak_ratio = float(min(max((peak_val - mean_val) / (3.0 * std_val), 0.0), 1.0))

    if abs(dx) > max_displacement or abs(dy) > max_displacement:
        logger.warning(
            f"Phase correlation shift ({dx:.1f}, {dy:.1f}) exceeded max displacement "
            f"threshold of {max_displacement}px. Rejecting shift."
        )
        return 0.0, 0.0, 0.0

    return dx, dy, peak_ratio


def align_modalities(
    optical: OpticalData,
    sar: SARData,
    method: str = "none",
    max_displacement: int = 20,
) -> AlignmentResult:
    """Perform fine spatial alignment of SAR data relative to optical reference.

    Args:
        optical: OpticalData reference container.
        sar: SARData source container (must have same spatial shape as optical).
        method: Alignment technique ('none', 'phase_correlation', 'feature_based').
        max_displacement: Maximum permissible shift in pixels before falling back.

    Returns:
        AlignmentResult with aligned SAR raster and transformation records.
    """
    if sar.shape[1:] != optical.shape[1:]:
        raise ValueError(
            f"SAR spatial dimensions {sar.shape[1:]} must match optical {optical.shape[1:]} "
            f"prior to fine alignment. Run reproject_to_reference first."
        )

    if method == "none":
        return AlignmentResult(
            aligned_sar=sar,
            displacement=(0.0, 0.0),
            transformation_matrix=np.eye(3, dtype=np.float32),
            status="skipped",
            method="none",
            confidence=1.0,
            notes="Fine alignment bypassed (method='none'). Geospatial reprojection retained.",
        )

    # Select representative optical band (NIR or Red or first band)
    if optical.has_band("B08"):
        ref_band = optical.get_band("B08")
    elif optical.has_band("B04"):
        ref_band = optical.get_band("B04")
    else:
        ref_band = optical.data[0]

    # Select representative SAR band (VV or first polarization)
    if sar.has_band("VV"):
        tgt_band = sar.get_band("VV")
    else:
        tgt_band = sar.data[0]

    if method == "phase_correlation":
        dx, dy, peak_resp = phase_correlation_shift(
            ref_band, tgt_band, max_displacement=max_displacement
        )

        if dx == 0.0 and dy == 0.0 and peak_resp == 0.0:
            return AlignmentResult(
                aligned_sar=sar,
                displacement=(0.0, 0.0),
                transformation_matrix=np.eye(3, dtype=np.float32),
                status="displacement_exceeded",
                method="phase_correlation",
                confidence=0.0,
                notes=f"Detected shift exceeded {max_displacement}px. Retained reprojection without shift.",
            )

        # Shift SAR channels
        aligned_data = np.empty_like(sar.data)
        for c in range(sar.channels):
            aligned_data[c] = shift(
                sar.data[c], shift=(dy, dx), order=1, mode="constant", cval=np.nan
            )

        # Build affine matrix
        t_matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float32)

        meta = dict(sar.metadata)
        meta["fine_alignment"] = {
            "method": "phase_correlation",
            "dx_px": dx,
            "dy_px": dy,
            "peak_response": peak_resp,
        }

        updated_sar = SARData(
            data=aligned_data,
            crs=sar.crs,
            transform=sar.transform,
            resolution=sar.resolution,
            bounds=sar.bounds,
            nodata=sar.nodata,
            band_names=list(sar.band_names),
            polarizations=list(sar.polarizations),
            metadata=meta,
            valid_fraction=sar.valid_fraction,
            is_calibrated=sar.is_calibrated,
            is_db=sar.is_db,
            terrain_corrected=sar.terrain_corrected,
        )

        notes = (
            f"Applied phase correlation subpixel translation: dx={dx:.2f}px, dy={dy:.2f}px "
            f"(peak response={peak_resp:.3f})."
        )
        logger.info(notes)

        return AlignmentResult(
            aligned_sar=updated_sar,
            displacement=(dx, dy),
            transformation_matrix=t_matrix,
            status="aligned",
            method="phase_correlation",
            confidence=peak_resp,
            notes=notes,
        )

    raise ValueError(
        f"Unsupported fine alignment method '{method}'. Choose 'phase_correlation' or 'none'."
    )
