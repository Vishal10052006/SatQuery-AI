"""
M4 adapter for the native M2 text-guided grounding subsystem.

M4 owns the stable ToolResult contract.
M2 owns the native GroundingAdapter implementation.

An explicitly supplied external specialist is still supported for
backward compatibility with existing tests/integrations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import (
    convert_tool_output,
    invoke_specialist,
)
from app.query.schemas import (
    Evidence,
    ExecutionStatus,
    ToolName,
    ToolResult,
)

from models.change.grounding import GroundingAdapter
from models.change.mask_processing import ChangeRegion


def _change_regions_from_result(
    previous_result: ToolResult | None,
) -> list[ChangeRegion]:
    """
    Reconstruct native M2 ChangeRegion objects from a previous
    M2 change-detection ToolResult.

    Native M2 data can expose regions either under the promoted
    ``data["regions"]`` field or inside ``data["evidence"]["regions"]``.
    Accept both forms so grounding consumes the actual M2 output.
    """

    if previous_result is None:
        return []

    data = previous_result.data or {}
    raw_regions = data.get("regions", [])

    # Older/alternate adapters may nest regions inside evidence.
    if not isinstance(raw_regions, list):
        evidence_data = data.get("evidence", {})
        if isinstance(evidence_data, dict):
            raw_regions = evidence_data.get("regions", [])

    if not isinstance(raw_regions, list):
        return []

    regions: list[ChangeRegion] = []

    for raw in raw_regions:
        if not isinstance(raw, dict):
            continue

        try:
            regions.append(
                ChangeRegion(
                    region_id=int(raw["region_id"]),
                    pixel_count=int(raw["pixel_count"]),
                    bbox_pixel={
                        "xmin": int(raw["bbox_pixel"]["xmin"]),
                        "ymin": int(raw["bbox_pixel"]["ymin"]),
                        "xmax": int(raw["bbox_pixel"]["xmax"]),
                        "ymax": int(raw["bbox_pixel"]["ymax"]),
                    },
                    centroid_pixel={
                        "x": float(raw["centroid_pixel"]["x"]),
                        "y": float(raw["centroid_pixel"]["y"]),
                    },
                    polygon_pixel=[
                        [int(point[0]), int(point[1])]
                        for point in raw["polygon_pixel"]
                    ],
                    confidence=float(raw.get("confidence", 0.0)),
                    bbox_geo=raw.get("bbox_geo"),
                    centroid_geo=raw.get("centroid_geo"),
                    polygon_geo=raw.get("polygon_geo"),
                    area_sq_m=raw.get("area_sq_m"),
                    target=raw.get("target"),
                )
            )
        except (KeyError, TypeError, ValueError):
            # Ignore malformed individual regions rather than
            # fabricating spatial evidence.
            continue

    return regions


def _previous_image_path(
    previous_result: ToolResult | None,
) -> str | None:
    """Extract the after-image path from a previous M2 result."""

    if previous_result is None:
        return None

    data = previous_result.data

    after = data.get("after")
    if after:
        return str(after)

    evidence = data.get("evidence", {})
    if isinstance(evidence, dict):
        after = evidence.get("after")
        if after:
            return str(after)

    return None


def _convert_grounding_output(
    raw_output: Any,
    previous_result: ToolResult | None = None,
) -> ToolResult:
    """Convert native M2 GroundingResult into M4 ToolResult.

    Preserve upstream M2 provenance so downstream M5 GIS can continue
    the execution chain without fabricating or re-discovering inputs.
    """

    native_status = getattr(
        raw_output,
        "status",
        "failed",
    )

    confidence = float(
        getattr(
            raw_output,
            "confidence",
            0.0,
        )
        or 0.0
    )

    model_name = getattr(
        raw_output,
        "model_name",
        "M2-grounding",
    )

    target = getattr(
        raw_output,
        "target",
        "",
    )

    message = getattr(
        raw_output,
        "message",
        "",
    )

    evidence_source = getattr(
        raw_output,
        "evidence_source",
        "none",
    )

    boxes_pixel = getattr(
        raw_output,
        "boxes_pixel",
        [],
    ) or []

    boxes_geo = getattr(
        raw_output,
        "boxes_geo",
        [],
    ) or []

    if native_status == "success":
        status = ExecutionStatus.SUCCESS
    elif native_status == "awaiting_model":
        status = ExecutionStatus.PARTIAL
    else:
        status = ExecutionStatus.FAILED

    data = {
        "answer": message,
        "task": "grounding",
        "model": model_name,
        "native_status": native_status,
        "target": target,
        "evidence_source": evidence_source,
        "boxes_pixel": boxes_pixel,
        "boxes_geo": boxes_geo,
        "evidence": (
            raw_output.to_dict()
            if hasattr(raw_output, "to_dict")
            else {}
        ),
    }

    # -------------------------------------------------------------
    # Preserve upstream M2/M3 provenance for downstream GIS.
    # -------------------------------------------------------------
    if previous_result is not None:
        upstream = previous_result.data

        provenance_keys = (
            "reference_geotiff",
            "reference_image",
            "reference_geotiff_path",
            "reference_image_path",
            "after",
            "after_image",
            "after_image_path",
            "change_mask",
            "change_mask_path",
            "bounding_boxes",
            "bounding_boxes_pixel",
            "regions",
        )

        for key in provenance_keys:
            value = upstream.get(key)
            if value is not None:
                data[key] = value

        data["upstream_tool"] = previous_result.tool.value
        data["upstream_status"] = previous_result.status.value
        data["upstream_data"] = dict(upstream)

        if boxes_pixel:
            data["bounding_boxes_pixel"] = boxes_pixel

        if boxes_geo:
            data["bounding_boxes"] = boxes_geo

    evidence = [
        Evidence(
            type="grounding_evidence",
            reference="m2_grounding",
            description=(
                "Spatial grounding evidence returned by "
                "the native M2 grounding subsystem."
            ),
            metadata={
                "evidence_source": evidence_source,
            },
        )
    ]

    return ToolResult(
        tool=ToolName.M2_GROUNDING,
        status=status,
        confidence=confidence,
        data=data,
        evidence=evidence,
        error=None,
    )


def build_grounding_adapter(
    specialist: Any | None = None,
):
    """
    Build the M4 M2-grounding adapter.

    With no specialist supplied, use the native M2 GroundingAdapter.

    With an explicit specialist supplied, preserve the existing
    external-specialist compatibility interface.
    """

    native_grounding = GroundingAdapter()

    def execute(
        previous_result: ToolResult | None = None,
        images: Any = None,
        target: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:

        target_text = (
            target
            or kwargs.get("query")
            or "Locate the requested objects."
        )

        if specialist is None:
            image_path = None

            if images:
                if isinstance(images, (list, tuple)):
                    image_path = str(images[0])
                else:
                    image_path = str(images)
            else:
                image_path = _previous_image_path(previous_result)

            change_regions = _change_regions_from_result(previous_result)

            raw_output = native_grounding.ground_target(
                image_path=image_path,
                target=target_text,
                change_regions=change_regions or None,
            )

            return _convert_grounding_output(
                raw_output,
                previous_result=previous_result,
            )

        if images:
            image_paths = [Path(image) for image in images]
        elif previous_result is not None:
            image_path = _previous_image_path(previous_result)

            if image_path:
                image_paths = [Path(image_path)]
            else:
                data = previous_result.data

                if data.get("bboxes"):
                    return previous_result.model_copy(
                        update={
                            "tool": ToolName.M2_GROUNDING,
                        }
                    )

                raise ValueError(
                    "Grounding received a previous result but "
                    "no reusable image/bounding-box input exists."
                )
        else:
            raise ValueError(
                "Grounding requires image input or a previous result."
            )

        params = {
            "query": target_text,
            "capability": "grounding",
            **kwargs,
        }

        raw_output = invoke_specialist(
            specialist,
            image_paths=image_paths,
            params=params,
            fallback_kwargs={
                "images": images,
                "previous_result": previous_result,
                "target": target,
                **kwargs,
            },
        )

        return convert_tool_output(
            raw_output,
            ToolName.M2_GROUNDING,
        )

    return execute
