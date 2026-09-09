"""
M4 adapter for the external VLM/VQA specialist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import (
    convert_tool_output,
    invoke_specialist,
)
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

        raw_output = invoke_specialist(
            specialist,
            image_paths=image_paths,
            params=params,
            fallback_kwargs={
                "images": images,
                "target": target,
                **kwargs,
            },
        )

        return convert_tool_output(
            raw_output,
            ToolName.M1_VQA,
        )

    return execute
