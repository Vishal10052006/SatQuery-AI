"""Optional adapters for connecting M1/M2/M3 implementations to M4.

Adapters normalize specialist callables to one simple interface without
forcing the specialist teams to depend on the orchestration internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .schemas import ToolName
from .tool_registry import ToolSpec, register_tool


Handler = Callable[..., Any]


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for one specialist handler."""

    tool: ToolName
    handler: Handler
    description: str


def _invoke(handler: Handler, context: Mapping[str, Any]) -> Any:
    """Invoke a specialist while supporting context-aware and no-arg handlers."""

    try:
        return handler(context)
    except TypeError:
        return handler()


def register_handler(config: AdapterConfig) -> None:
    """Register a concrete specialist handler with the M4 tool registry."""

    register_tool(
        ToolSpec(
            name=config.tool,
            description=config.description,
            handler=lambda context=None, _handler=config.handler: _invoke(
                _handler, dict(context or {})
            ),
        )
    )


def register_specialists(
    *,
    vqa: Handler | None = None,
    grounding: Handler | None = None,
    change_detection: Handler | None = None,
    optical_sar: Handler | None = None,
) -> None:
    """Convenience registration for M1/M2/M3 specialist implementations."""

    handlers = (
        (ToolName.VQA, vqa, "M1 visual question answering specialist."),
        (ToolName.GROUNDING, grounding, "M2 text-to-region grounding specialist."),
        (
            ToolName.CHANGE_DETECTION,
            change_detection,
            "M2 bi-temporal change detection specialist.",
        ),
        (
            ToolName.OPTICAL_SAR,
            optical_sar,
            "M3 optical + SAR corroboration specialist.",
        ),
    )

    for tool, handler, description in handlers:
        if handler is not None:
            register_handler(AdapterConfig(tool, handler, description))
