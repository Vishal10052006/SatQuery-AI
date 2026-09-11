"""
SatQuery AI - M4 Execution Planner

Converts a ParsedQuery into an ordered ExecutionPlan.

The planner is responsible for deciding:

    1. Which specialist tool must run.
    2. Whether additional tools are required.
    3. The order in which those tools must execute.

Architecture:

    QueryParser
         ↓
    ParsedQuery
         ↓
    IntentClassifier
         ↓
    Planner
         ↓
    ExecutionPlan
         ↓
    Executor
"""

from app.query.schemas import (
    ExecutionPlan,
    Intent,
    ParsedQuery,
    PlanStep,
    ToolName,
)


class QueryPlanner:
    """
    Deterministic execution planner for the SatQuery MVP.

    The planner deliberately operates on ParsedQuery rather than
    raw natural language. Query interpretation therefore remains
    separated from execution planning.
    """

    def create_plan(
        self,
        parsed_query: ParsedQuery,
    ) -> ExecutionPlan:
        """
        Create an ordered execution plan.

        Args:
            parsed_query:
                Structured query produced by the query parser.

        Returns:
            ExecutionPlan containing one or more executable steps.
        """

        steps: list[PlanStep] = []

        # Generic change queries need a semantic scope even when the
        # parser did not find an explicit object such as buildings or roads.
        # This prevents M2 from falling back to an unscoped raw pixel diff
        # and gives the land-change/water-protection logic a clear target.
        change_target = parsed_query.target or "land-surface changes"

        # ----------------------------------------------------
        # VQA
        # ----------------------------------------------------
        if parsed_query.intent == Intent.VQA:
            steps.append(
                PlanStep(
                    step_id=1,
                    tool=ToolName.M1_VQA,
                    operation=parsed_query.operation.value,
                    inputs=["images"],
                    parameters={"target": parsed_query.target},
                )
            )

        # ----------------------------------------------------
        # Optical + SAR
        # ----------------------------------------------------
        elif parsed_query.intent == Intent.OPTICAL_SAR:
            steps.append(
                PlanStep(
                    step_id=1,
                    tool=ToolName.M3_OPTICAL_SAR,
                    operation=parsed_query.operation.value,
                    inputs=["optical", "sar"],
                    parameters={"target": parsed_query.target},
                )
            )

            if parsed_query.requires_geospatial:
                steps.append(
                    PlanStep(
                        step_id=2,
                        tool=ToolName.M5_GIS,
                        operation="localize",
                        inputs=["previous_result", "output_dir"],
                        parameters={"target": parsed_query.target},
                    )
                )

        # ----------------------------------------------------
        # Change Detection
        # ----------------------------------------------------
        elif parsed_query.intent == Intent.CHANGE_DETECTION:
            steps.append(
                PlanStep(
                    step_id=1,
                    tool=ToolName.M2_CHANGE_DETECTION,
                    operation=parsed_query.operation.value,
                    inputs=["before", "after"],
                    parameters={
                        "target": change_target,
                        "temporal": parsed_query.temporal,
                    },
                )
            )

            if parsed_query.requires_grounding:
                steps.append(
                    PlanStep(
                        step_id=2,
                        tool=ToolName.M2_GROUNDING,
                        operation="localize",
                        inputs=["previous_result"],
                        parameters={"target": change_target},
                    )
                )

            if parsed_query.requires_geospatial:
                steps.append(
                    PlanStep(
                        step_id=len(steps) + 1,
                        tool=ToolName.M5_GIS,
                        operation="geospatial_analysis",
                        inputs=["previous_result", "output_dir"],
                        parameters={"target": change_target},
                    )
                )

        # ----------------------------------------------------
        # Explicit Grounding
        # ----------------------------------------------------
        elif parsed_query.intent == Intent.GROUNDING:
            steps.append(
                PlanStep(
                    step_id=1,
                    tool=ToolName.M2_GROUNDING,
                    operation="localize",
                    inputs=["images"],
                    parameters={"target": parsed_query.target},
                )
            )

            if parsed_query.requires_geospatial:
                steps.append(
                    PlanStep(
                        step_id=2,
                        tool=ToolName.M5_GIS,
                        operation="geospatial_analysis",
                        inputs=["previous_result", "output_dir"],
                        parameters={"target": parsed_query.target},
                    )
                )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------
        if not steps:
            raise ValueError(
                f"Unable to create execution plan for intent: "
                f"{parsed_query.intent}"
            )

        return ExecutionPlan(
            intent=parsed_query.intent,
            steps=steps,
        )


planner = QueryPlanner()
