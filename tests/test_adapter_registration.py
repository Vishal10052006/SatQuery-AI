"""Tests for specialist registration."""

from types import SimpleNamespace

from app.adapters.register import register_specialists
from app.query.registry import ToolRegistry
from app.query.schemas import ToolName


class FakeSpecialist:
    """Minimal specialist object for registration testing."""

    def execute(self, image_paths, params, **kwargs):
        return SimpleNamespace(
            answer="test",
            confidence=0.9,
            metrics={},
            warnings=[],
            mask_geojson=None,
            bboxes=None,
        )


def test_register_available_specialists() -> None:
    """Available specialists should be registered."""

    registry = ToolRegistry()

    register_specialists(
        registry,
        vqa=FakeSpecialist(),
        change_detection=FakeSpecialist(),
        grounding=FakeSpecialist(),
        optical_sar=FakeSpecialist(),
    )

    assert registry.is_registered(ToolName.M1_VQA)
    assert registry.is_registered(
        ToolName.M2_CHANGE_DETECTION
    )
    assert registry.is_registered(
        ToolName.M2_GROUNDING
    )
    assert registry.is_registered(
        ToolName.M3_OPTICAL_SAR
    )


def test_missing_specialists_are_not_registered() -> None:
    """Unavailable specialists should remain unregistered."""

    registry = ToolRegistry()

    register_specialists(
        registry,
        change_detection=FakeSpecialist(),
    )

    assert registry.is_registered(
        ToolName.M2_CHANGE_DETECTION
    )

    assert not registry.is_registered(
        ToolName.M1_VQA
    )

    assert not registry.is_registered(
        ToolName.M3_OPTICAL_SAR
    )


def test_registration_returns_same_registry() -> None:
    """Registration should mutate and return the supplied registry."""

    registry = ToolRegistry()

    result = register_specialists(
        registry,
        vqa=FakeSpecialist(),
    )

    assert result is registry
