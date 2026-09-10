"""
Public M4 API.

This module exposes the stable interface that frontend/backend
components can use without depending on M4 internals.
"""

from __future__ import annotations

from typing import Any

from app.agents.bootstrap import build_m4_controller, build_production_controller
from app.agents.controller import AgentController
from app.query.schemas import AgentResponse, QueryRequest


class SatQueryAPI:
    """
    Public interface to the SatQuery AI M4 controller.

    The API owns the controller lifecycle and exposes a single
    query method to callers.
    """

    def __init__(
        self,
        controller: AgentController | None = None,
        **specialists: Any,
    ) -> None:
        """
        Initialize the public M4 API.

        An existing controller may be supplied for dependency
        injection and testing. Otherwise a controller is created
        from the supplied specialist implementations.
        """

        if controller is not None:
            self.controller = controller
        elif specialists:
            self.controller = build_m4_controller(
                **specialists
            )
        else:
            self.controller = build_production_controller()

    def ask(
        self,
        query: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """
        Execute a natural-language satellite query.

        Parameters
        ----------
        query:
            User's natural-language question.

        context:
            Runtime inputs such as image paths, temporal pairs,
            optical/SAR inputs, or other application data.

        Returns
        -------
        AgentResponse
            Structured M4 response.
        """

        request = QueryRequest(
            query=query,
        )

        return self.controller.run(
            request=request,
            context=context,
        )


# Convenient default API instance for simple integrations.
satquery = SatQueryAPI()


def ask(
    query: str,
    *,
    context: dict[str, Any] | None = None,
) -> AgentResponse:
    """
    Convenience function for callers that do not need to manage
    an API object explicitly.
    """

    return satquery.ask(
        query,
        context=context,
    )
