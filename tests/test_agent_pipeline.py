from agent.pipeline import run_agent


def test_unknown_query_does_not_execute_tools():
    result = run_agent("Tell me something random")

    assert result["route"]["intent"] == "unknown"
    assert result["plan"] == []
    assert result["evidence"]["total"] == 0


def test_change_query_returns_auditable_pipeline():
    result = run_agent("What changed between these two satellite images?")

    assert result["route"]["intent"] == "change_analysis"
    assert result["plan"] == ["change_detection"]
    assert result["evidence"]["total"] == 1
    assert result["confidence"]["evidence_count"] == 1
    assert "trace" in result
