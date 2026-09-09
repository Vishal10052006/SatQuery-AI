"""
M4 adapter for the external VLM/VQA specialist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import convert_tool_output
from app.query.schemas import ToolName, ToolResult


def build_vqa_adapter(
    specialist: Any,
):
    """Wrap the external VLMGroundingTool for M1 VQA."""

    def execute(
        images: Any,
        target: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:

        if not images:
            raise ValueError(
                "VQA requires at least one image."
            )

        image_paths = [
            Path(image)
            for image in images
        ]

        params = {
            "query": target or "Describe this satellite image.",
            "capability": "vqa",
            **kwargs,
        }

        raw_output = specialist.execute(
            image_paths=image_paths,
            params=params,
        )

        return convert_tool_output(
            raw_output,
            ToolName.M1_VQA,
        )

    return execute
