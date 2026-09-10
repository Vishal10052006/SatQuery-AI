"""Early pixel-level concatenation fusion of registered Optical and SAR channels."""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from modules.optical_sar.config import OpticalData, SARData

logger = logging.getLogger(__name__)


@dataclass
class EarlyFusionResult:
    """Structured container for early (pixel-level) concatenated multimodal raster data.

    Attributes:
        fused_data: Concatenated multi-band array of shape (total_channels, height, width).
        channel_names: Ordered list of channel names (e.g. ['B02', 'B03', 'B04', 'B08', 'VV', 'VH']).
        crs: Coordinate Reference System.
        transform: Affine geotransform matrix.
        resolution: Spatial resolution tuple (x_res, y_res).
        bounds: Bounding box tuple (left, bottom, right, top).
        optical_channels: Names of optical channels included.
        sar_channels: Names of SAR channels included.
        metadata: Geospatial and processing metadata.
    """
    fused_data: np.ndarray
    channel_names: List[str]
    crs: Any
    transform: Any
    resolution: Tuple[float, float]
    bounds: Tuple[float, float, float, float]
    optical_channels: List[str]
    sar_channels: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return self.fused_data.shape[0]

    @property
    def height(self) -> int:
        return self.fused_data.shape[1]

    @property
    def width(self) -> int:
        return self.fused_data.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.fused_data.shape


def fuse_early(
    optical_data: OpticalData,
    sar_data: SARData,
    optical_bands: Optional[List[str]] = None,
    sar_polarizations: Optional[List[str]] = None,
) -> EarlyFusionResult:
    """Perform early channel-level stacking of registered optical and SAR rasters.

    Dynamic Channel Composition:
        Does not hardcode a fixed channel count. For example:
        - Sentinel-2 (B02, B03, B04, B08) + Sentinel-1 (VV, VH) -> 6-channel composite.
        - Sentinel-2 (B02, B03, B04, B08) + Sentinel-1 (VV)     -> 5-channel composite.
        - Custom band subsets are supported dynamically.

    Args:
        optical_data: Registered and normalized OpticalData container.
        sar_data: Reprojected, registered, and normalized SARData container.
        optical_bands: Subset of optical bands to include. If None, includes all bands.
        sar_polarizations: Subset of SAR polarizations to include. If None, includes all.

    Returns:
        EarlyFusionResult with concatenated channels and complete metadata.

    Raises:
        ValueError: If spatial dimensions (H, W) do not match or requested bands are missing.
    """
    if optical_data.shape[1:] != sar_data.shape[1:]:
        raise ValueError(
            f"Spatial dimension mismatch: Optical is {optical_data.shape[1:]}, "
            f"SAR is {sar_data.shape[1:]}. Ensure SAR is reprojected to optical grid first."
        )

    # Resolve optical channels
    if optical_bands is not None:
        for b in optical_bands:
            if not optical_data.has_band(b):
                raise ValueError(
                    f"Requested optical band '{b}' not found in optical bands {optical_data.band_names}."
                )
        opt_indices = [optical_data.band_names.index(b) for b in optical_bands]
        opt_arr = optical_data.data[opt_indices]
        opt_names = list(optical_bands)
    else:
        opt_arr = optical_data.data
        opt_names = list(optical_data.band_names)

    # Resolve SAR channels
    if sar_polarizations is not None:
        for p in sar_polarizations:
            if not sar_data.has_band(p):
                raise ValueError(
                    f"Requested SAR polarization '{p}' not found in SAR bands {sar_data.band_names}."
                )
        sar_indices = [sar_data.band_names.index(p) for p in sar_polarizations]
        sar_arr = sar_data.data[sar_indices]
        sar_names = list(sar_polarizations)
    else:
        sar_arr = sar_data.data
        sar_names = list(sar_data.band_names)

    # Concatenate along channel axis (axis 0)
    fused_arr = np.concatenate([opt_arr, sar_arr], axis=0).astype(np.float32)
    all_channel_names = opt_names + sar_names

    metadata: Dict[str, Any] = {
        "fusion_type": "early_channel_stacking",
        "num_optical_channels": len(opt_names),
        "num_sar_channels": len(sar_names),
        "channel_ordering": all_channel_names,
        "optical_crs": str(optical_data.crs),
        "sar_crs": str(sar_data.crs),
    }

    logger.info(
        f"Fused Optical ({len(opt_names)} ch) and SAR ({len(sar_names)} ch) into "
        f"{fused_arr.shape[0]}-channel tensor: {all_channel_names}"
    )

    return EarlyFusionResult(
        fused_data=fused_arr,
        channel_names=all_channel_names,
        crs=optical_data.crs,
        transform=optical_data.transform,
        resolution=optical_data.resolution,
        bounds=optical_data.bounds,
        optical_channels=opt_names,
        sar_channels=sar_names,
        metadata=metadata,
    )
