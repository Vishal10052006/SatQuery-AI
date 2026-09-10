from agent.executor import execute_plan
from agent.schemas import ToolName
from agent.trace import AgentTrace


def test_executor_reports_unavailable_tools_without_guessing() -> None:
    result = execute_plan([ToolName.VQA], context={})

    assert result.success is False
    assert result.steps[0].tool == ToolName.VQA
    assert result.steps[0].status == "unavailable"


def test_executor_calls_registered_handler() -> None:
    from agent.tool_registry import ToolSpec, register_tool

    register_tool(
        ToolSpec(
            name=ToolName.VQA,
            description="test VQA handler",
            handler=lambda context: {"answer": "test", "input": context["query"]},
        )
    )

    result = execute_plan([ToolName.VQA], context={"query": "What is here?"})

    assert result.success is True
    assert result.steps[0].result["answer"] == "test"


def test_trace_serialises_plan_and_execution() -> None:
    from agent.schemas import Intent, RouteDecision

    decision = RouteDecision(intent=Intent.SCENE_UNDERSTANDING, tools=[ToolName.VQA])
    execution = execute_plan([ToolName.VQA], context={})
    trace = AgentTrace.from_run(
        query="What is in this image?",
        decision=decision,
        plan=[ToolName.VQA],
        execution=execution,
    )

    payload = trace.to_dict()
    assert payload["intent"] == "scene_understanding"
    assert payload["plan"] == ["vqa"]
    assert payload["execution"]["steps"][0]["status"] in {"success", "unavailable", "error"}
