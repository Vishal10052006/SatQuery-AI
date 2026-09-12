"""
SatQuery AI - M4 Agent Controller

Public orchestration entry point for the M4 agent.

Pipeline:

    QueryRequest
         ↓
      Parser
         ↓
     Classifier
         ↓
      Planner
         ↓
      Executor
         ↓
    Synthesizer
         ↓
    AgentResponse
"""

from __future__ import annotations

from typing import Any

from app.agents.executor import AgentExecutor
from app.agents.planner import QueryPlanner
from app.agents.synthesizer import ResponseSynthesizer
from app.query.classifier import IntentClassifier
from app.query.parser import QueryParser
from app.query.registry import ToolRegistry
from app.query.schemas import (
    AgentResponse,
    ExecutionStatus,
    QueryRequest,
)


class AgentController:
    """
    Main M4 orchestration controller.

    The controller owns the workflow but does not implement
    specialist satellite-analysis logic.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        parser: QueryParser | None = None,
        classifier: IntentClassifier | None = None,
        planner: QueryPlanner | None = None,
        executor: AgentExecutor | None = None,
        synthesizer: ResponseSynthesizer | None = None,
    ) -> None:
        self.registry = registry
        self.parser = parser or QueryParser()
        self.classifier = classifier or IntentClassifier()
        self.planner = planner or QueryPlanner()
        self.executor = executor or AgentExecutor(registry)
        self.synthesizer = synthesizer or ResponseSynthesizer()

    def run(
        self,
        request: QueryRequest,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Execute the complete M4 agent pipeline."""
        try:
            parsed_query = self.parser.parse(
                request.query,
                images=request.images,
            )

            self.classifier.classify(parsed_query)
            plan = self.planner.create_plan(parsed_query)

            runtime_context = dict(context or {})
            if request.images:
                runtime_context.setdefault("images", request.images)

            report = self.executor.execute(
                plan,
                context=runtime_context,
            )

            return self.synthesizer.synthesize(
                intent=parsed_query.intent,
                report=report,
                parsed_query=parsed_query,
            )

        except Exception as exc:
            return AgentResponse(
                status=ExecutionStatus.FAILED,
                answer="Unable to process the query.",
                confidence=0.0,
                error=str(exc),
            )
