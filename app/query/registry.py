"""
SatQuery AI - M4 Tool Registry

The registry provides a stable interface between the M4 agent
and specialist modules developed by M1, M2, M3, and M5.

Architecture:

    M4 Classifier
          ↓
       ToolName
          ↓
     ToolRegistry
          ↓
    Specialist Adapter
          ↓
       ToolResult

The registry deliberately does not know how a specialist model
works internally. Each specialist only needs to expose a callable
that accepts the agreed input and returns a ToolResult.
"""

from collections.abc import Callable
from typing import Any

from app.query.schemas import (
    ExecutionStatus,
    ToolName,
    ToolResult,
)


# A tool implementation is simply a callable that receives
# keyword arguments and returns a validated ToolResult.
ToolFunction = Callable[..., ToolResult]


class ToolRegistry:
    """
    Central registry for all SatQuery specialist tools.

    Responsibilities:
        - Register tools.
        - Retrieve tools.
        - Check tool availability.
        - Execute registered tools.
        - Keep M4 independent from specialist internals.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._tools: dict[ToolName, ToolFunction] = {}

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    def register(
        self,
        name: ToolName,
        function: ToolFunction,
    ) -> None:
        """
        Register a specialist implementation.

        Args:
            name:
                Stable ToolName used by M4.

            function:
                Callable implementing that tool.

        Raises:
            TypeError:
                If the supplied function is not callable.
        """

        if not callable(function):
            raise TypeError(
                f"Tool '{name.value}' must be callable."
            )

        self._tools[name] = function

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    def get(self, name: ToolName) -> ToolFunction:
        """
        Retrieve a registered tool.

        Raises:
            KeyError:
                If the requested tool has not been registered.
        """

        if name not in self._tools:
            raise KeyError(
                f"Tool '{name.value}' is not registered."
            )

        return self._tools[name]

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    def is_registered(self, name: ToolName) -> bool:
        """Return whether a tool is currently registered."""

        return name in self._tools

    # --------------------------------------------------------
    # Listing
    # --------------------------------------------------------

    def list_tools(self) -> list[ToolName]:
        """Return all currently registered tools."""

        return list(self._tools.keys())

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def execute(
        self,
        name: ToolName,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute a registered specialist tool.

        Every specialist must return a ToolResult. This gives M4
        a consistent interface regardless of the underlying model.

        Exceptions are converted into a failed ToolResult so that
        the agent can handle tool failures without crashing the
        entire pipeline.
        """

        try:
            function = self.get(name)

            result = function(**kwargs)

            # Protect the M4 contract: every specialist must return
            # a ToolResult rather than an arbitrary Python object.
            if not isinstance(result, ToolResult):
                raise TypeError(
                    f"Tool '{name.value}' returned "
                    f"{type(result).__name__}; expected ToolResult."
                )

            return result

        except Exception as exc:
            return ToolResult(
                tool=name,
                status=ExecutionStatus.FAILED,
                confidence=0.0,
                data={},
                evidence=[],
                error=str(exc),
            )


# ------------------------------------------------------------
# Default registry
# ------------------------------------------------------------

registry = ToolRegistry()
