"""
Specialist registration for the M4 ToolRegistry.

This module is the integration boundary between M4 and the
actual specialist implementations supplied by other modules.

M4 does not depend on specialist internals. Each specialist
is wrapped behind the stable M4 ToolResult contract.
"""

from __future__ import annotations

from typing import Any

from app.adapters.m1_vqa_adapter import build_vqa_adapter
from app.adapters.m2_change_adapter import build_change_detection_adapter
from app.adapters.m2_grounding_adapter import build_grounding_adapter
from app.adapters.m3_optical_sar_adapter import build_optical_sar_adapter
from app.adapters.m5_gis_adapter import make_m5_gis_adapter
from app.query.registry import ToolRegistry
from app.query.schemas import ToolName


def register_specialists(
    registry: ToolRegistry,
    *,
    vqa: Any | None = None,
    change_detection: Any | None = None,
    grounding: Any | None = None,
    optical_sar: Any | None = None,
    gis: Any | None = None,
) -> ToolRegistry:
    """
    Register whichever specialist implementations are available.

    Each external specialist is converted into the stable M4
    ToolRegistry interface through an adapter.
    """

    if vqa is not None:
        registry.register(
            ToolName.M1_VQA,
            build_vqa_adapter(vqa),
        )

    if change_detection is not None:
        registry.register(
            ToolName.M2_CHANGE_DETECTION,
            build_change_detection_adapter(change_detection),
        )

    # M2 grounding is natively integrated into this repository.
    #
    # When no external specialist is supplied, the adapter uses
    # models.change.grounding.GroundingAdapter.
    #
    # An explicitly supplied `grounding` specialist remains supported
    # for external integrations and compatibility tests.
    registry.register(
        ToolName.M2_GROUNDING,
        build_grounding_adapter(grounding),
    )

    if optical_sar is not None:
        registry.register(
            ToolName.M3_OPTICAL_SAR,
            build_optical_sar_adapter(optical_sar),
        )

    # M5 GIS is natively integrated into this repository.
    #
    # When no external specialist is supplied, make_m5_gis_adapter()
    # uses the native geospatial.integration.process_m2_m3_result()
    # implementation.
    #
    # An explicitly supplied `gis` callable is still supported for
    # compatibility with external specialists and existing tests.
    registry.register(
        ToolName.M5_GIS,
        make_m5_gis_adapter(gis),
    )

    return registry
