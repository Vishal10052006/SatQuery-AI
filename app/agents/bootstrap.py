"""
M4 runtime bootstrap.

This module provides two runtime construction paths:

1. build_m4_controller()
   Dependency-injection builder used by tests and custom deployments.

2. build_production_controller()
   Production builder that wires the native SatQuery specialist
   implementations into M4.
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
    gis: Any | None = None,
) -> AgentController:
    """
    Build an M4 controller using explicitly supplied specialists.

    This function intentionally does not discover or import native
    specialists automatically. That preserves dependency injection,
    partial integration, and deterministic testing.
    """

    registry = ToolRegistry()

    register_specialists(
        registry,
        vqa=vqa,
        change_detection=change_detection,
        grounding=grounding,
        optical_sar=optical_sar,
        gis=gis,
    )

    return AgentController(
        registry=registry,
    )


def build_production_controller() -> AgentController:
    """
    Build the production SatQuery controller.

    Native specialist implementations are explicitly wired here.
    The individual specialist adapters remain responsible for
    converting native outputs into M4 ToolResult objects.

    M2 grounding is intentionally not wired here yet because its
    native interface still requires a dedicated M4 bridge.
    """

    from m1_earthdial.earthdial_adapter import analyze_image
    from models.change.adapter import run_change_detection
    from modules.optical_sar.pipeline import run_optical_sar_pipeline

    return build_m4_controller(
        vqa=analyze_image,
        change_detection=run_change_detection,
        optical_sar=run_optical_sar_pipeline,
        # M5 is natively integrated by register_specialists().
        gis=None,
    )