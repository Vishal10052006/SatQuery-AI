"""End-to-end M2 Change Analysis Subsystem pipeline.

Orchestrates:
Validate -> Preprocess -> Align -> Change Detect / RCD -> Geospatial Localization
-> Statistics -> Visualization -> Quality / Confidence -> M2Result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.contracts import SpecialistResult
from geospatial.geometry import calculate_ground_area
from geospatial.raster import bbox_pixel_to_geo, polygon_pixel_to_geo

from .align import align_images
from .mask_processing import ChangeRegion, extract_change_regions
from .preprocess import load_and_preprocess
from .rcd import RCDAdapter
from .validation import validate_bitemporal_inputs
from .visualize import generate_m2_visualizations


@dataclass
class M2Result:
    """Structured and auditable result of the M2 Change Analysis Subsystem."""

    task: str = "change_detection"
    status: str = "success"             # 'success' | 'fallback_baseline' | 'awaiting_model' | 'failed'
    target: str | None = None
    detector: str = "pixel-difference-baseline"
    detector_type: str = "baseline"     # 'baseline' | 'neural' | 'fallback_baseline'
    change_detected: bool = False
    confidence: float = 0.0
    changed_pixels: int = 0
    change_fraction: float = 0.0
    mean_difference: float = 0.0
    regions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    mask_path: str | None = None
    overlay_path: str | None = None
    composite_path: str | None = None
    geospatial_reference_available: bool = False
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    geographic_bbox: dict[str, tuple[float, float]] | None = None
    changed_area_sq_m: float | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "target": self.target,
            "detector": self.detector,
            "detector_type": self.detector_type,
            "change_detected": self.change_detected,
            "confidence": round(self.confidence, 4),
            "changed_pixels": self.changed_pixels,
            "change_fraction": round(self.change_fraction, 6),
            "mean_difference": round(self.mean_difference, 6),
            "regions": self.regions,
            "number_of_regions": len(self.regions),
            "region_sizes": [r.get("pixel_count", 0) for r in self.regions],
            "artifacts": self.artifacts,
            "mask_path": self.mask_path,
            "overlay_path": self.overlay_path,
            "composite_path": self.composite_path,
            "geospatial_reference_available": self.geospatial_reference_available,
            "crs": self.crs,
            "transform": self.transform,
            "geographic_bbox": self.geographic_bbox,
            "changed_area_sq_m": self.changed_area_sq_m,
            "quality": self.quality,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "error": self.error,
        }

    def to_specialist_result(self) -> SpecialistResult:
        """Convert M2Result to M4 SpecialistResult maintaining 100% backward compatibility."""
        num_regs = len(self.regions)
        target_info = f" for target '{self.target}'" if self.target else ""
        if self.status == "failed":
            claim = f"Change detection failed: {self.error or 'invalid inputs'}."
        elif self.change_detected:
            claim = (
                f"Detected {num_regs} changed region(s){target_info} "
                f"({self.changed_pixels} pixels, {self.change_fraction * 100:.2f}% of scene)."
            )
        else:
            claim = f"No significant change detected{target_info} between before and after observations."

        evidence: dict[str, Any] = {
            "model": self.detector,
            "status": self.status,
            "detector_type": self.detector_type,
            "threshold": self.metadata.get("threshold", 0.15),
            "image_size": self.metadata.get("image_size", {}),
            "changed_pixels": self.changed_pixels,
            "changed_fraction": round(self.change_fraction, 6),
            "mean_difference": round(self.mean_difference, 6),
            "regions": self.regions,
            "number_of_regions": len(self.regions),
            "region_sizes": [r.get("pixel_count", 0) for r in self.regions],
            "target": self.target,
            "before": self.metadata.get("before_path"),
            "after": self.metadata.get("after_path"),
            "before_raster": self.metadata.get("before_meta", {}),
            "after_raster": self.metadata.get("after_meta", {}),
            "geospatial_reference_available": self.geospatial_reference_available,
            "crs": self.crs,
            "changed_area_sq_m": self.changed_area_sq_m,
            "quality": self.quality,
        }

        return SpecialistResult(
            task="change_detection",
            model=self.detector,
            status=self.status if self.status in {"success", "awaiting_model", "failed"} else "success",
            confidence=self.confidence,
            claim=claim,
            evidence=evidence,
            artifacts=self.artifacts,
            error=self.error,
        )


def run_m2(
    before_path: str | Path | None,
    after_path: str | Path | None,
    target: str | None = None,
    output_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> M2Result:
    """Run the complete M2 remote-sensing change-analysis pipeline.

    Parameters
    ----------
    before_path : Path or str to the earlier remote-sensing observation.
    after_path : Path or str to the later remote-sensing observation.
    target : Optional natural-language target query (e.g. 'newly constructed buildings').
    output_dir : Directory to persist visualization artifacts.
    config : Optional dict with hyperparameters ('threshold', 'min_pixels', 'use_percentiles').
    """
    cfg = config or {}
    threshold = float(cfg.get("threshold", 0.15))
    min_pixels = int(cfg.get("min_pixels", 8))
    use_percentiles = bool(cfg.get("use_percentiles", False))
    allow_fallback = bool(cfg.get("allow_fallback", True))

    # Stage 1: Bi-temporal Input Validation
    validation = validate_bitemporal_inputs(before_path, after_path, target=target)
    if not validation.is_valid:
        return M2Result(
            status="failed",
            error="; ".join(validation.errors),
            warnings=validation.warnings,
            metadata=validation.metadata,
        )

    # Stage 2: Remote-Sensing Preprocessing
    try:
        before_prep = load_and_preprocess(before_path, use_percentiles=use_percentiles)
        after_prep = load_and_preprocess(after_path, use_percentiles=use_percentiles)
    except Exception as exc:
        return M2Result(
            status="failed",
            error=f"Preprocessing failed: {exc}",
            warnings=validation.warnings,
        )

    # Stage 3: Image Alignment / Registration
    alignment = align_images(before_prep, after_prep)
    if not alignment.success:
        return M2Result(
            status="failed",
            error=alignment.error or "Spatial alignment failed.",
            warnings=validation.warnings + alignment.warnings,
        )

    aligned_after = alignment.after
    all_warnings = list(validation.warnings) + list(alignment.warnings)

    # Stage 4: Change Detection / Target-guided RCD
    cleaned_target = validation.metadata.get("target")
    if cleaned_target:
        # Use Referring Change Detection adapter
        rcd_adapter = RCDAdapter()
        rcd_res = rcd_adapter.detect_target_change(
            before_prep,
            aligned_after,
            target=cleaned_target,
            threshold=threshold,
            min_pixels=min_pixels,
            allow_fallback=allow_fallback,
        )
        detector_name = rcd_res.model_name
        detector_type = rcd_res.detector_type
        status = rcd_res.status
        change_mask = rcd_res.change_mask
        diff_map = rcd_res.difference_map
        change_regions = rcd_res.regions
    else:
        # Standard bi-temporal difference baseline
        detector_name = "pixel-difference-baseline"
        detector_type = "baseline"
        status = "success"

        diff_map = np.abs(aligned_after.gray - before_prep.gray)
        valid_mask = before_prep.valid_mask & aligned_after.valid_mask
        diff_map[~valid_mask] = 0.0

        raw_mask = diff_map >= threshold
        change_mask, change_regions = extract_change_regions(
            raw_mask,
            diff_map,
            min_pixels=min_pixels,
        )

    if change_mask is None:
        change_mask = np.zeros(before_prep.original_shape, dtype=bool)
    if diff_map is None:
        diff_map = np.zeros(before_prep.original_shape, dtype=np.float32)

    changed_pixels = int(change_mask.sum())
    change_fraction = float(change_mask.mean())
    mean_diff = float(diff_map.mean())
    change_detected = (changed_pixels > 0)

    # Stage 5: Geospatial Localization & Area Calculation
    before_meta = before_prep.metadata
    transform = before_meta.get("transform")
    crs = before_meta.get("crs")
    georeferenced = bool(before_meta.get("georeferenced") and transform)

    overall_geo_bbox = None
    total_area_sq_m = None

    if georeferenced and transform:
        bw = before_meta.get("width", change_mask.shape[1])
        bh = before_meta.get("height", change_mask.shape[0])
        overall_geo_bbox = bbox_pixel_to_geo(
            {"xmin": 0, "ymin": 0, "xmax": bw - 1, "ymax": bh - 1},
            transform,
        )
        total_area_sq_m = calculate_ground_area(changed_pixels, transform, crs)

        for r in change_regions:
            r.bbox_geo = bbox_pixel_to_geo(r.bbox_pixel, transform)
            cx, cy = r.centroid_pixel["x"], r.centroid_pixel["y"]
            r.centroid_geo = {
                "x": round(transform[0] * cx + transform[1] * cy + transform[2], 6),
                "y": round(transform[3] * cx + transform[4] * cy + transform[5], 6),
            }
            r.polygon_geo = polygon_pixel_to_geo(r.polygon_pixel, transform)
            r.area_sq_m = calculate_ground_area(r.pixel_count, transform, crs)

    # Convert regions to dictionary representations
    regions_dicts = [r.to_dict() for r in change_regions]

    # Stage 6: Visualization Artifacts
    artifacts: list[str] = []
    mask_path: str | None = None
    overlay_path: str | None = None
    composite_path: str | None = None

    if output_dir:
        out_dir = Path(output_dir)
        try:
            b_pil = Image.open(before_path).convert("RGB")
            a_pil = Image.open(after_path).convert("RGB")
            viz_files = generate_m2_visualizations(
                before_img=b_pil,
                after_img=a_pil,
                mask=change_mask,
                regions=change_regions,
                output_dir=out_dir,
                target=cleaned_target,
            )
            # Primary visual artifact at index 0 for contract & test compatibility
            artifacts = [viz_files["composite"]]
            composite_path = str(out_dir / viz_files["composite"])
            overlay_path = str(out_dir / viz_files["overlay"])
            mask_path = str(out_dir / viz_files["mask"])
        except Exception as exc:
            all_warnings.append(f"Visualization generation encountered warning: {exc}")

    # Stage 7: Confidence and Quality Metrics (honest scoring)
    valid_fraction = float(before_prep.metadata.get("valid_pixel_fraction", 1.0))
    align_quality = alignment.quality_score

    if not change_detected:
        # High confidence that no significant change occurred
        confidence = round(0.88 * align_quality * valid_fraction, 2)
    else:
        # Confidence incorporates alignment quality, valid pixels, and region evidence
        base_conf = 0.75 if detector_type == "baseline" else 0.70
        confidence = round(base_conf * align_quality * valid_fraction, 2)

    quality = {
        "alignment_status": alignment.status,
        "alignment_quality": align_quality,
        "valid_pixel_fraction": valid_fraction,
        "georeferenced": georeferenced,
        "detector_type": detector_type,
        "model_status": status,
        "num_regions": len(change_regions),
    }

    meta = {
        "before_path": str(before_path),
        "after_path": str(after_path),
        "threshold": threshold,
        "min_pixels": min_pixels,
        "image_size": {
            "width": int(change_mask.shape[1]),
            "height": int(change_mask.shape[0]),
        },
        "before_meta": before_meta,
        "after_meta": aligned_after.metadata,
    }

    return M2Result(
        task="change_detection",
        status=status,
        target=cleaned_target,
        detector=detector_name,
        detector_type=detector_type,
        change_detected=change_detected,
        confidence=confidence,
        changed_pixels=changed_pixels,
        change_fraction=round(change_fraction, 6),
        mean_difference=round(mean_diff, 6),
        regions=regions_dicts,
        artifacts=artifacts,
        mask_path=mask_path,
        overlay_path=overlay_path,
        composite_path=composite_path,
        geospatial_reference_available=georeferenced,
        crs=crs,
        transform=transform,
        geographic_bbox=overall_geo_bbox,
        changed_area_sq_m=total_area_sq_m,
        quality=quality,
        warnings=all_warnings,
        metadata=meta,
    )
