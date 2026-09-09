"""Tests for the M4 runtime bootstrap."""

from types import SimpleNamespace

from app.agents.bootstrap import build_m4_controller
from app.query.schemas import ToolName


class FakeSpecialist:
    """Minimal specialist used to verify dependency injection."""

    def execute(self, image_paths, params, **kwargs):
        return SimpleNamespace(
            answer="test",
            confidence=0.90,
            metrics={},
            warnings=[],
            mask_geojson=None,
            bboxes=None,
        )


def test_bootstrap_registers_supplied_specialists() -> None:
    """Supplied specialists should be available through M4."""

    controller = build_m4_controller(
        vqa=FakeSpecialist(),
        change_detection=FakeSpecialist(),
        grounding=FakeSpecialist(),
        optical_sar=FakeSpecialist(),
    )

    assert controller.registry.is_registered(
        ToolName.M1_VQA
    )

    assert controller.registry.is_registered(
        ToolName.M2_CHANGE_DETECTION
    )

    assert controller.registry.is_registered(
        ToolName.M2_GROUNDING
    )

    assert controller.registry.is_registered(
        ToolName.M3_OPTICAL_SAR
    )


def test_bootstrap_allows_partial_integration() -> None:
    """M4 should support running with only available specialists."""

    controller = build_m4_controller(
        change_detection=FakeSpecialist(),
    )

    assert controller.registry.is_registered(
        ToolName.M2_CHANGE_DETECTION
    )

    assert not controller.registry.is_registered(
        ToolName.M1_VQA
    )

    assert not controller.registry.is_registered(
        ToolName.M3_OPTICAL_SAR
    )


def test_bootstrap_creates_independent_registries() -> None:
    """Each controller should receive its own registry."""

    controller_a = build_m4_controller(
        vqa=FakeSpecialist(),
    )

    controller_b = build_m4_controller(
        change_detection=FakeSpecialist(),
    )

    assert controller_a.registry is not controller_b.registry

    assert controller_a.registry.is_registered(
        ToolName.M1_VQA
    )

    assert not controller_a.registry.is_registered(
        ToolName.M2_CHANGE_DETECTION
    )

    assert controller_b.registry.is_registered(
        ToolName.M2_CHANGE_DETECTION
    )


def test_bootstrap_registers_m5_gis() -> None:
    """Runtime bootstrap should register an injected M5 specialist."""

    from types import SimpleNamespace

    from app.agents.bootstrap import build_m4_controller
    from app.query.schemas import ToolName

    def gis(**kwargs):
        return SimpleNamespace(
            answer="GIS complete.",
            confidence=0.90,
            metrics={},
            warnings=[],
            mask_geojson=None,
            bboxes=None,
        )

    controller = build_m4_controller(gis=gis)

    assert controller.registry.is_registered(
        ToolName.M5_GIS
    )
