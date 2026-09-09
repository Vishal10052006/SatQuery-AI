"""SAR processing package for M3 Optical + SAR module."""

from modules.optical_sar.sar.loader import load_sar
from modules.optical_sar.sar.calibration import (
    linear_to_db,
    db_to_linear,
    calibrate_sar,
)
from modules.optical_sar.sar.speckle import apply_speckle_filter, lee_filter
from modules.optical_sar.sar.terrain import (
    apply_terrain_correction,
    TerrainCorrectionResult,
)
from modules.optical_sar.sar.normalization import normalize_sar

__all__ = [
    "load_sar",
    "linear_to_db",
    "db_to_linear",
    "calibrate_sar",
    "apply_speckle_filter",
    "lee_filter",
    "apply_terrain_correction",
    "TerrainCorrectionResult",
    "normalize_sar",
]
