"""
Adapter for the M5 GIS / geospatial specialist.

The adapter converts an external M5 output into the stable
M4 ToolResult contract.

M4 does not depend on the internal implementation of M5.
"""


from __future__ import annotations

from typing import Any

from app.adapters.common import convert_tool_output
from app.query.schemas import ToolName, ToolResult


def adapt_m5_gis_output(raw_output: Any) -> ToolResult:
    """
    Convert an external M5 GIS result into an M4 ToolResult.

    The external object may expose fields such as:

        answer
        confidence
        metrics
        warnings
        mask_geojson
        bboxes

    The common adapter performs the normalization.
    """

    return convert_tool_output(
        raw_output,
        ToolName.M5_GIS,
    )


def make_m5_gis_adapter(
    specialist: Any,
):
    """
    Wrap an M5 specialist callable with the M4 contract.

    Parameters
    ----------
    specialist:
        External M5 GIS function/callable.

    Returns
    -------
    callable
        Function returning ToolResult.
    """

    def adapter(**kwargs: Any) -> ToolResult:
        """Execute M5 and normalize its output."""

        raw_output = specialist(**kwargs)

        return adapt_m5_gis_output(raw_output)

    return adapter
