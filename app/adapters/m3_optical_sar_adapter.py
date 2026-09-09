"""
M4 adapter for the external Optical-SAR fusion specialist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import convert_tool_output
from app.query.schemas import ToolName, ToolResult


def build_optical_sar_adapter(
    specialist: Any,
):
    """
    Wrap an external OpticalSARFusionTool instance.

    Expected external interface:

        specialist.execute(
            image_paths=[optical, sar],
            params={...},
        )
    """

    def execute(
        optical: Any,
        sar: Any,
        target: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:

        params = {
            "query": target or "",
            **kwargs,
        }

        raw_output = specialist.execute(
            image_paths=[
                Path(optical),
                Path(sar),
            ],
            params=params,
        )

        return convert_tool_output(
            raw_output,
            ToolName.M3_OPTICAL_SAR,
        )

    return execute
