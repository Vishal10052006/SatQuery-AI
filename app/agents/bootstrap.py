"""
M4 runtime bootstrap.

This module creates the AgentController and injects whichever
specialist implementations are available at runtime.

The specialists remain external to M4. M4 only knows their
adapter/ToolResult contracts.
"""

from __future__ import annotations

from typing import Any

from app.adapters.register import register_specialists
from app.agents.controller import AgentController
from app.query.registry import ToolRegistry


def build_m4_controller(
    *,
    vqa: Any | None = None,
    change_detection: Any | None = None,
    grounding: Any | None = None,
    optical_sar: Any | None = None,
) -> AgentController:
    """
    Build a fully configured M4 AgentController.

    Any specialist that is not available remains unregistered.
    This allows M4 to start incrementally while teammates'
    implementations are integrated.

    Parameters
    ----------
    vqa:
        Real M1/VQA specialist.

    change_detection:
        Real M2 change-detection specialist.

    grounding:
        Real M2 grounding specialist.

    optical_sar:
        Real M3 optical/SAR specialist.

    Returns
    -------
    AgentController
        Configured M4 controller.
    """

    registry = ToolRegistry()

    register_specialists(
        registry,
        vqa=vqa,
        change_detection=change_detection,
        grounding=grounding,
        optical_sar=optical_sar,
    )

    return AgentController(
        registry=registry,
    )
