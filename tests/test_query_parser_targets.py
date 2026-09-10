from app.query.parser import QueryParser


def test_deforestation_is_not_parsed_as_forest():
    parser = QueryParser()

    parsed = parser.parse(
        "Where did the deforestation happen?"
    )

    assert parsed.target == "deforestation"


def test_forest_is_still_detected():
    parser = QueryParser()

    parsed = parser.parse(
        "Where is the forest?"
    )

    assert parsed.target == "forest"


def test_target_matching_uses_word_boundaries():
    parser = QueryParser()

    parsed = parser.parse(
        "Analyze vegetation changes."
    )

    assert parsed.target == "vegetation"


def test_duplicate_targets_do_not_change_result():
    parser = QueryParser()

    parsed = parser.parse(
        "Find the houses."
    )

    assert parsed.target == "houses"
