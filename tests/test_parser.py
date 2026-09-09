"""
Tests for the SatQuery M4 query parser.

The tests cover the main SIH query categories:
    - VQA
    - Change Detection
    - New Construction
    - Grounding
    - Optical + SAR
"""

from app.query.parser import QueryParser
from app.query.schemas import Intent, Operation


def test_describe_image():
    """A scene-description query should route to M1/VQA."""

    parser = QueryParser()

    result = parser.parse(
        "Describe this satellite image."
    )

    assert result.intent == Intent.VQA
    assert result.operation == Operation.DESCRIBE


def test_change_detection():
    """A comparison query should route to M2."""

    parser = QueryParser()

    result = parser.parse(
        "What changed between these two images?",
        images=["before.tif", "after.tif"],
    )

    assert result.intent == Intent.CHANGE_DETECTION
    assert result.operation == Operation.DETECT_CHANGE
    assert result.temporal is True


def test_new_buildings():
    """A new-construction query should detect the building target."""

    parser = QueryParser()

    result = parser.parse(
        "Find newly constructed buildings between these two images.",
        images=["before.tif", "after.tif"],
    )

    assert result.intent == Intent.CHANGE_DETECTION
    assert result.operation == Operation.DETECT_NEW
    assert result.target == "buildings"
    assert result.temporal is True
    assert result.requires_grounding is True


def test_grounding():
    """A location query should be classified as grounding."""

    parser = QueryParser()

    result = parser.parse(
        "Where did the change occur?"
    )

    assert result.intent == Intent.CHANGE_DETECTION
    assert result.requires_grounding is True


def test_optical_sar():
    """An optical + SAR query should route to M3."""

    parser = QueryParser()

    result = parser.parse(
        "Use optical and SAR images to identify water regions."
    )

    assert result.intent == Intent.OPTICAL_SAR
    assert result.operation == Operation.ANALYZE
    assert "optical" in result.modalities
    assert "sar" in result.modalities
    assert result.target == "water"


def test_empty_query():
    """An empty query must be rejected."""

    parser = QueryParser()

    try:
        parser.parse("")
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError(
            "Empty query should raise ValueError."
        )
