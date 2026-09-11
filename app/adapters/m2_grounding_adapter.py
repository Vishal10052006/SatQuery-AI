"""
M4 adapter for the native M2 bi-temporal grounding subsystem.

M4 owns the stable ToolResult contract. M2 owns the native grounding implementation.
The adapter preserves upstream M2 provenance and generated visual artifacts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import convert_tool_output, invoke_specialist
from app.query.schemas import Evidence, ExecutionStatus, ToolName, ToolResult
from models.change.grounding import GroundingAdapter
from models.change.mask_processing import ChangeRegion


def _change_regions_from_result(previous_result: ToolResult | None) -> list[ChangeRegion]:
    """Reconstruct usable native M2 regions without making grounding fatal."""
    if previous_result is None:
        return []
    data = previous_result.data or {}
    raw_regions = data.get("regions", [])
    if not isinstance(raw_regions, list):
        evidence_data = data.get("evidence", {})
        raw_regions = evidence_data.get("regions", []) if isinstance(evidence_data, dict) else []
    if not isinstance(raw_regions, list):
        return []

    regions: list[ChangeRegion] = []
    for index, raw in enumerate(raw_regions, start=1):
        if not isinstance(raw, dict):
            continue
        try:
            bbox = raw.get("bbox_pixel") or raw.get("bbox")
            polygon = raw.get("polygon_pixel")
            if not isinstance(bbox, dict):
                continue
            if not isinstance(polygon, list) or len(polygon) < 4:
                polygon = [
                    [int(bbox["xmin"]), int(bbox["ymin"])],
                    [int(bbox["xmax"]), int(bbox["ymin"])],
                    [int(bbox["xmax"]), int(bbox["ymax"])],
                    [int(bbox["xmin"]), int(bbox["ymax"])],
                    [int(bbox["xmin"]), int(bbox["ymin"])],
                ]
            centroid = raw.get("centroid_pixel") or {
                "x": (float(bbox["xmin"]) + float(bbox["xmax"])) / 2,
                "y": (float(bbox["ymin"]) + float(bbox["ymax"])) / 2,
            }
            regions.append(ChangeRegion(
                region_id=int(raw.get("region_id", index)),
                pixel_count=int(raw.get("pixel_count", 0)),
                bbox_pixel={
                    "xmin": int(bbox["xmin"]), "ymin": int(bbox["ymin"]),
                    "xmax": int(bbox["xmax"]), "ymax": int(bbox["ymax"]),
                },
                centroid_pixel={"x": float(centroid["x"]), "y": float(centroid["y"])},
                polygon_pixel=[[int(point[0]), int(point[1])] for point in polygon],
                confidence=float(raw.get("confidence", 0.0) or 0.0),
                bbox_geo=raw.get("bbox_geo"),
                centroid_geo=raw.get("centroid_geo"),
                polygon_geo=raw.get("polygon_geo"),
                area_sq_m=raw.get("area_sq_m"),
                target=raw.get("target"),
            ))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return regions


def _previous_image_path(previous_result: ToolResult | None) -> str | None:
    """Extract the after-image path from a previous M2 result."""
    if previous_result is None:
        return None
    data = previous_result.data or {}
    after = data.get("after")
    if after:
        return str(after)
    evidence = data.get("evidence", {})
    if isinstance(evidence, dict) and evidence.get("after"):
        return str(evidence["after"])
    return None


def _convert_grounding_output(raw_output: Any, previous_result: ToolResult | None = None) -> ToolResult:
    """Convert native M2 GroundingResult into the M4 ToolResult contract."""
    native_status = getattr(raw_output, "status", "failed")
    confidence = float(getattr(raw_output, "confidence", 0.0) or 0.0)
    model_name = getattr(raw_output, "model_name", "M2-grounding")
    target = getattr(raw_output, "target", "")
    message = getattr(raw_output, "message", "")
    evidence_source = getattr(raw_output, "evidence_source", "none")
    boxes_pixel = getattr(raw_output, "boxes_pixel", []) or []
    boxes_geo = getattr(raw_output, "boxes_geo", []) or []

    if native_status == "success":
        status = ExecutionStatus.SUCCESS
    elif native_status == "awaiting_model":
        status = ExecutionStatus.PARTIAL
    else:
        status = ExecutionStatus.FAILED

    data: dict[str, Any] = {
        "answer": message,
        "task": "grounding",
        "model": model_name,
        "native_status": native_status,
        "target": target,
        "evidence_source": evidence_source,
        "boxes_pixel": boxes_pixel,
        "boxes_geo": boxes_geo,
        "evidence": raw_output.to_dict() if hasattr(raw_output, "to_dict") else {},
    }

    if previous_result is not None:
        upstream = previous_result.data or {}
        for key in (
            "reference_geotiff", "reference_image", "reference_geotiff_path", "reference_image_path",
            "before", "after", "after_image", "after_image_path", "change_mask", "change_mask_path",
            "overlay_path", "mask_path", "composite_path", "artifacts", "artifact_paths",
            "bounding_boxes", "bounding_boxes_pixel", "regions", "change_detected", "changed_pixels",
            "changed_fraction", "change_fraction", "number_of_regions", "detector", "detector_type",
            "geospatial_reference_available", "crs", "transform", "geographic_bbox",
            "changed_area_sq_m", "quality", "warnings", "target",
        ):
            if upstream.get(key) is not None:
                data[key] = upstream[key]
        data["upstream_tool"] = previous_result.tool.value
        data["upstream_status"] = previous_result.status.value
        data["upstream_data"] = dict(upstream)
        if boxes_pixel:
            data["bounding_boxes_pixel"] = boxes_pixel
        if boxes_geo:
            data["bounding_boxes"] = boxes_geo

    return ToolResult(
        tool=ToolName.M2_GROUNDING,
        status=status,
        confidence=confidence,
        data=data,
        evidence=[Evidence(
            type="grounding_evidence",
            reference="m2_grounding",
            description="Spatial grounding evidence returned by the native M2 grounding subsystem.",
            metadata={"evidence_source": evidence_source},
        )],
        error=None,
    )


def build_grounding_adapter(specialist: Any | None = None):
    """Build the M4 M2-grounding adapter."""
    native_grounding = GroundingAdapter()

    def execute(previous_result: ToolResult | None = None, images: Any = None, target: str | None = None, **kwargs: Any) -> ToolResult:
        upstream_target = previous_result.data.get("target") if previous_result is not None else None
        target_text = target or upstream_target or "detected change regions"
        change_regions = _change_regions_from_result(previous_result)

        if specialist is None:
            image_path = None
            if images:
                image_path = str(images[0]) if isinstance(images, (list, tuple)) else str(images)
            else:
                image_path = _previous_image_path(previous_result)

            try:
                raw_output = native_grounding.ground_target(
                    image_path=image_path,
                    target=target_text,
                    change_regions=change_regions or None,
                )
                return _convert_grounding_output(raw_output, previous_result=previous_result)
            except Exception as exc:
                # Grounding must not erase a valid M2 change result. Return a
                # partial result and preserve the upstream metrics for M6.
                data = dict(previous_result.data) if previous_result is not None else {}
                data.update({
                    "task": "grounding",
                    "model": "change-derived-grounding",
                    "native_status": "partial",
                    "target": target_text,
                    "evidence_source": "change_regions" if change_regions else "none",
                    "boxes_pixel": [r.bbox_pixel for r in change_regions],
                    "grounding_warning": str(exc),
                })
                return ToolResult(
                    tool=ToolName.M2_GROUNDING,
                    status=ExecutionStatus.PARTIAL,
                    confidence=0.0,
                    data=data,
                    evidence=[Evidence(
                        type="grounding_evidence",
                        reference="m2_grounding",
                        description="Grounding degraded; upstream M2 change evidence was preserved.",
                    )],
                    error=None,
                )

        image_paths = [Path(image) for image in images] if images else None
        if image_paths is None and previous_result is not None:
            image_path = _previous_image_path(previous_result)
            if image_path:
                image_paths = [Path(image_path)]
        if not image_paths:
            raise ValueError("Grounding requires image input or a previous result with a reusable image.")

        raw_output = invoke_specialist(
            specialist,
            image_paths=image_paths,
            params={"query": target_text, "capability": "grounding", **kwargs},
            fallback_kwargs={"images": images, "previous_result": previous_result, "target": target, **kwargs},
        )
        return convert_tool_output(raw_output, ToolName.M2_GROUNDING)

    return execute
