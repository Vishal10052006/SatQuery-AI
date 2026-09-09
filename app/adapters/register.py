"""
Specialist registration for the M4 ToolRegistry.

This module is the integration boundary between M4 and the
actual specialist implementations supplied by other modules.

M4 does not import or depend on the internal implementation
of M1/M2/M3. It receives specialist objects and wraps them
with the appropriate adapter.
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
    Register whichever real specialist implementations are available.

    Parameters
    ----------
    registry:
        M4 ToolRegistry instance.

    vqa:
        Real M1/VQA specialist instance.

    change_detection:
        Real M2 change-detection specialist instance.

    grounding:
        Real M2 grounding specialist instance.

    optical_sar:
        Real M3 optical/SAR specialist instance.

    Returns
    -------
    ToolRegistry
        The same registry, populated with available specialists.

    Notes
    -----
    gis:
        Real M5 GIS / geospatial specialist instance.

    M5 is registered when a GIS specialist is supplied.
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

    if grounding is not None:
        registry.register(
            ToolName.M2_GROUNDING,
            build_grounding_adapter(grounding),
        )

    if optical_sar is not None:
        registry.register(
            ToolName.M3_OPTICAL_SAR,
            build_optical_sar_adapter(optical_sar),
        )

    if gis is not None:
        registry.register(
            ToolName.M5_GIS,
            make_m5_gis_adapter(gis),
        )

    return registry
