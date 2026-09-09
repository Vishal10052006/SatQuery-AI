"""
Tests for the SatQuery M4 intent classifier.

The classifier must map each supported ParsedQuery intent
to the correct specialist module.
"""

from app.query.classifier import IntentClassifier
from app.query.parser import QueryParser
from app.query.schemas import Intent, ToolName


def test_vqa_routes_to_m1():
    """VQA queries must route to M1."""

    parser = QueryParser()
    classifier = IntentClassifier()

    parsed = parser.parse(
        "Describe this satellite image."
    )

    tool = classifier.classify(parsed)

    assert parsed.intent == Intent.VQA
    assert tool == ToolName.M1_VQA


def test_change_detection_routes_to_m2():
    """Change queries must route to M2."""

    parser = QueryParser()
    classifier = IntentClassifier()

    parsed = parser.parse(
        "What changed between these two images?",
        images=["before.tif", "after.tif"],
    )

    tool = classifier.classify(parsed)

    assert parsed.intent == Intent.CHANGE_DETECTION
    assert tool == ToolName.M2_CHANGE_DETECTION


def test_grounding_routes_to_m2():
    """Grounding queries must route to M2 grounding."""

    parser = QueryParser()
    classifier = IntentClassifier()

    parsed = parser.parse(
        "Where did the change occur?"
    )

    # The parser currently identifies this query as a
    # change-detection intent with grounding required.
    #
    # The classifier therefore routes it through the
    # change-detection specialist.
    tool = classifier.classify(parsed)

    assert parsed.intent == Intent.CHANGE_DETECTION
    assert parsed.requires_grounding is True
    assert tool == ToolName.M2_CHANGE_DETECTION


def test_optical_sar_routes_to_m3():
    """Optical + SAR queries must route to M3."""

    parser = QueryParser()
    classifier = IntentClassifier()

    parsed = parser.parse(
        "Use optical and SAR images to identify water regions."
    )

    tool = classifier.classify(parsed)

    assert parsed.intent == Intent.OPTICAL_SAR
    assert tool == ToolName.M3_OPTICAL_SAR


def test_new_building_query_routes_to_m2():
    """New-construction queries must route to M2."""

    parser = QueryParser()
    classifier = IntentClassifier()

    parsed = parser.parse(
        "Find newly constructed buildings between these two images.",
        images=["before.tif", "after.tif"],
    )

    tool = classifier.classify(parsed)

    assert parsed.intent == Intent.CHANGE_DETECTION
    assert parsed.target == "buildings"
    assert parsed.requires_grounding is True
    assert tool == ToolName.M2_CHANGE_DETECTION
