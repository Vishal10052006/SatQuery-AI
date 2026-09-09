"""
M4 adapter for the grounding capability of the external VLM specialist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.common import (
    convert_tool_output,
    invoke_specialist,
)
from app.query.schemas import ToolName, ToolResult


def build_grounding_adapter(
    specialist: Any,
):
    """Wrap the external VLMGroundingTool for M2 grounding."""

    def execute(
        previous_result: ToolResult | None = None,
        images: Any = None,
        target: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:

        if images:
            image_paths = [
                Path(image)
                for image in images
            ]
        elif previous_result is not None:
            # A previous change result may already contain
            # geospatial/bounding-box information.
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
            "query": target or "Locate the requested objects.",
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
