"""Registry of specialist capabilities available to the M4 agent.

At this stage the registry contains capability metadata only. Actual model
implementations will be supplied by M1/M2/M3 and connected through adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .schemas import ToolName


@dataclass(frozen=True)
class ToolSpec:
    """Describes a callable specialist capability."""

    name: ToolName
    description: str
    handler: Optional[Callable] = None


TOOLS: Dict[ToolName, ToolSpec] = {
    ToolName.VQA: ToolSpec(
        name=ToolName.VQA,
        description="Answer questions about the contents of a satellite image.",
    ),
    ToolName.GROUNDING: ToolSpec(
        name=ToolName.GROUNDING,
        description="Locate or ground a text-described object/region in an image.",
    ),
    ToolName.CHANGE_DETECTION: ToolSpec(
        name=ToolName.CHANGE_DETECTION,
        description="Detect spatial changes between two temporally separated images.",
    ),
    ToolName.OPTICAL_SAR: ToolSpec(
        name=ToolName.OPTICAL_SAR,
        description="Compare or fuse optical and SAR observations for corroboration.",
    ),
}


def register_tool(tool: ToolSpec) -> None:
    """Register or replace a specialist tool implementation."""

    TOOLS[tool.name] = tool


def get_tool(name: ToolName) -> ToolSpec:
    """Return a registered tool specification or raise a clear error."""

    try:
        return TOOLS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown agent tool: {name}") from exc
