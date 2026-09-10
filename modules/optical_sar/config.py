"""Configuration and shared data structures for the M3 Optical + SAR module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class RasterData:
    """Base container for geospatial raster datasets.

    Attributes:
        data: Array of raster data with shape (channels, height, width).
        crs: Coordinate Reference System (pyproj.CRS, rasterio.crs.CRS, or WKT/PROJ string).
        transform: Affine transformation mapping pixel coordinates to CRS coordinates.
        resolution: Tuple of (x_resolution, y_resolution).
        bounds: Tuple of (left, bottom, right, top) in CRS coordinates.
        nodata: Value representing nodata or missing pixels, if specified.
        band_names: List of names corresponding to each channel index.
        metadata: Additional arbitrary metadata dictionary from driver/headers.
    """
    data: np.ndarray
    crs: Any
    transform: Any
    resolution: Tuple[float, float]
    bounds: Tuple[float, float, float, float]
    nodata: Optional[float] = None
    band_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def height(self) -> int:
        return self.data.shape[1]

    @property
    def width(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def get_band(self, name: str) -> np.ndarray:
        """Retrieve a specific band by its name."""
        if name not in self.band_names:
            raise ValueError(
                f"Band '{name}' not found. Available bands: {self.band_names}"
            )
        idx = self.band_names.index(name)
        return self.data[idx]

    def has_band(self, name: str) -> bool:
        """Check whether a band with the given name is present."""
        return name in self.band_names


@dataclass
class OpticalData(RasterData):
    """Container for optical multi-band satellite raster data (e.g., Sentinel-2)."""
    cloud_fraction: Optional[float] = None
    valid_fraction: Optional[float] = None


@dataclass
class SARData(RasterData):
    """Container for Synthetic Aperture Radar data (e.g., Sentinel-1)."""
    polarizations: List[str] = field(default_factory=list)
    is_db: bool = False
    is_calibrated: bool = False
    terrain_corrected: bool = False
    valid_fraction: Optional[float] = None

    def __post_init__(self):
        if not self.band_names and self.polarizations:
            self.band_names = list(self.polarizations)
        elif not self.polarizations and self.band_names:
            self.polarizations = list(self.band_names)


@dataclass
class OpticalConfig:
    """Configuration options for optical data processing."""
    band_mapping: Dict[str, int] = field(default_factory=lambda: {
        "B02": 1,  # Blue
        "B03": 2,  # Green
        "B04": 3,  # Red
        "B08": 4,  # NIR
    })
    normalization_method: str = "percentile"  # 'percentile' or 'minmax'
    percentile_bounds: Tuple[float, float] = (1.0, 99.0)
    clip_output: bool = True


@dataclass
class SARConfig:
    """Configuration options for SAR data processing."""
    polarizations: List[str] = field(default_factory=lambda: ["VV", "VH"])
    is_already_calibrated: bool = True
    calibration_type: str = "sigma0"
    to_db: bool = True
    min_db: float = -35.0
    max_db: float = 5.0
    speckle_filter_method: str = "lee"  # 'lee', 'median', or 'none'
    speckle_kernel_size: int = 5
    normalization_method: str = "percentile"  # 'percentile' or 'minmax'
    percentile_bounds: Tuple[float, float] = (1.0, 99.0)


@dataclass
class RegistrationConfig:
    """Configuration options for geospatial reprojection and alignment."""
    resampling_method: str = "bilinear"  # 'bilinear', 'nearest', 'cubic', 'lanczos'
    fine_alignment_method: str = "none"   # 'phase_correlation', 'feature_based', 'none'
    max_displacement: int = 20
    validation_threshold: float = 0.70
    min_overlap_ratio: float = 0.80


@dataclass
class FusionConfig:
    """Configuration options for multimodal fusion."""
    method: str = "feature"  # 'early' or 'feature'
    feature_dim: int = 128
    num_classes: int = 3


@dataclass
class ConfidenceConfig:
    """Configuration options for confidence estimation."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "optical_quality": 0.25,
        "sar_quality": 0.20,
        "registration_quality": 0.20,
        "model_confidence": 0.35,
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.80,
        "medium": 0.50,
    })


@dataclass
class OpticalSARConfig:
    """Master configuration container for M3 Optical + SAR module."""
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    sar: SARConfig = field(default_factory=SARConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    random_seed: int = 42
