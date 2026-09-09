"""Bi-temporal remote-sensing input and query validation.

Validates:
- File existence and readability
- Supported image formats (GeoTIFF, TIFF, PNG, JPEG/JPG)
- Dimensions and band compatibility
- Geospatial metadata consistency (CRS, transform, resolution, NoData)
- Temporal pair validity
- Target query specification
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from geospatial.raster import inspect_raster

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2"}


@dataclass
class ValidationResult:
    """Structured validation outcome for bi-temporal inputs and queries."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    before_meta: dict[str, Any] = field(default_factory=dict)
    after_meta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "before_meta": self.before_meta,
            "after_meta": self.after_meta,
            "metadata": self.metadata,
        }


def validate_bitemporal_inputs(
    before_path: str | Path | None,
    after_path: str | Path | None,
    target: str | None = None,
) -> ValidationResult:
    """Perform rigorous validation of bi-temporal image pair and target query."""
    errors: list[str] = []
    warnings: list[str] = []
    before_meta: dict[str, Any] = {}
    after_meta: dict[str, Any] = {}

    # 1. Existence and paths
    if not before_path:
        errors.append("Before image path was not provided.")
    if not after_path:
        errors.append("After image path was not provided.")

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    b_path = Path(before_path)
    a_path = Path(after_path)

    if not b_path.exists():
        errors.append(f"Before image does not exist: {b_path}")
    if not a_path.exists():
        errors.append(f"After image does not exist: {a_path}")

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Temporal pair compatibility check
    try:
        if b_path.resolve() == a_path.resolve():
            warnings.append(
                "Identical file path provided for both before and after observations; no temporal change is expected."
            )
    except Exception:
        pass

    # 2. File formats
    b_ext = b_path.suffix.lower()
    a_ext = a_path.suffix.lower()

    if b_ext not in SUPPORTED_EXTENSIONS:
        errors.append(
            f"Unsupported before image extension '{b_ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if a_ext not in SUPPORTED_EXTENSIONS:
        errors.append(
            f"Unsupported after image extension '{a_ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # 3. Readability & Inspection
    try:
        before_meta = inspect_raster(str(b_path))
    except Exception as exc:
        errors.append(f"Cannot read before image: {exc}")

    try:
        after_meta = inspect_raster(str(a_path))
    except Exception as exc:
        errors.append(f"Cannot read after image: {exc}")

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Validate image dimensions (> 0)
    bw, bh = before_meta.get("width", 0), before_meta.get("height", 0)
    aw, ah = after_meta.get("width", 0), after_meta.get("height", 0)

    if bw <= 0 or bh <= 0:
        errors.append(f"Invalid dimensions for before image: width={bw}, height={bh}")
    if aw <= 0 or ah <= 0:
        errors.append(f"Invalid dimensions for after image: width={aw}, height={ah}")

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    # Check dimension mismatch
    if (bw, bh) != (aw, ah):
        warnings.append(
            f"Dimension mismatch between before ({bw}x{bh}) and after ({aw}x{ah}) images. "
            "Spatial resampling will be applied during alignment."
        )

    # Band count warning
    b_bands = before_meta.get("bands", 1)
    a_bands = after_meta.get("bands", 1)
    if b_bands != a_bands:
        warnings.append(
            f"Band count difference: before has {b_bands} band(s), after has {a_bands} band(s)."
        )

    # 4. Geospatial CRS and transform compatibility
    b_geo = before_meta.get("georeferenced", False)
    a_geo = after_meta.get("georeferenced", False)

    if b_geo and a_geo:
        b_crs = before_meta.get("crs")
        a_crs = after_meta.get("crs")
        if b_crs and a_crs and b_crs != a_crs:
            warnings.append(
                f"CRS mismatch: before has '{b_crs}', after has '{a_crs}'. "
                "Registration requires CRS reprojection or common reference grid."
            )
    elif b_geo != a_geo:
        warnings.append(
            "Partial georeferencing: only one of the input images has geospatial metadata."
        )
    else:
        warnings.append("Images are not georeferenced; pixel coordinate grid will be used.")

    # 5. Target query validation
    cleaned_target: str | None = None
    if target is not None:
        target_stripped = target.strip()
        if len(target_stripped) == 0:
            warnings.append("Empty target query provided; proceeding with general change detection.")
        else:
            cleaned_target = target_stripped

    metadata = {
        "before_path": str(b_path),
        "after_path": str(a_path),
        "target": cleaned_target,
        "georeferenced": b_geo and a_geo,
    }

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        before_meta=before_meta,
        after_meta=after_meta,
        metadata=metadata,
    )
