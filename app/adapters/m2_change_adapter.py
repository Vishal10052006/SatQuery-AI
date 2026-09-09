"""
M4 adapter for the external M2 bi-temporal change detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import convert_tool_output
from app.query.schemas import ToolName, ToolResult


def build_change_detection_adapter(
    specialist: Any,
):
    """
    Wrap an external ChangeDetectionTool instance.

    Expected external interface:

        specialist.execute(
            image_paths=[Path(T1), Path(T2)],
            params={...},
        )
    """

    def execute(
        before: Any,
        after: Any,
        target: str | None = None,
        temporal: bool = True,
        **kwargs: Any,
    ) -> ToolResult:

        params = {
            "query_hint": target or "",
            "temporal": temporal,
            **kwargs,
        }

        raw_output = specialist.execute(
            image_paths=[
                Path(before),
                Path(after),
            ],
            params=params,
        )

        return convert_tool_output(
            raw_output,
            ToolName.M2_CHANGE_DETECTION,
        )

    return execute
