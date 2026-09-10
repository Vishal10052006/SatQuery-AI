"""Data contracts used by the SatQuery agent router.

The schemas intentionally stay framework-agnostic so the agent can later be
called from FastAPI, a CLI, or another orchestration layer without changing
its core routing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Intent(str, Enum):
    """Top-level analysis intents supported by the MVP router."""

    SCENE_UNDERSTANDING = "scene_understanding"
    GROUNDING = "grounding"
    CHANGE_ANALYSIS = "change_analysis"
    MULTIMODAL_ANALYSIS = "multimodal_analysis"
    UNKNOWN = "unknown"


class ToolName(str, Enum):
    """Specialist capabilities the agent may invoke."""

    VQA = "vqa"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR = "optical_sar"


@dataclass(frozen=True)
class RouteDecision:
    """Deterministic routing decision produced from a natural-language query."""

    intent: Intent
    tools: List[ToolName] = field(default_factory=list)
    requires_temporal: bool = False
    requires_optical: bool = False
    requires_sar: bool = False
    requires_grounding: bool = False
    confidence: float = 0.0
    matched_signals: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation for APIs and traces."""

        return {
            "intent": self.intent.value,
            "tools": [tool.value for tool in self.tools],
            "requires_temporal": self.requires_temporal,
            "requires_optical": self.requires_optical,
            "requires_sar": self.requires_sar,
            "requires_grounding": self.requires_grounding,
            "confidence": self.confidence,
            "matched_signals": list(self.matched_signals),
        }
