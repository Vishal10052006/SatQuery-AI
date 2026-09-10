"""Deterministic first-pass query router for SatQuery AI.

M4 owns the orchestration layer. This router deliberately uses transparent
keyword/signal matching for the MVP so routing remains debuggable and has a
safe fallback before an optional LLM planner is introduced.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .schemas import Intent, RouteDecision, ToolName


# Signals are intentionally broad enough for natural user wording while
# remaining easy to inspect during the SIH demo.
_CHANGE_SIGNALS = (
    "change",
    "changed",
    "difference",
    "compare",
    "comparison",
    "before and after",
    "temporal",
    "over time",
    "increase",
    "decrease",
    "growth",
    "newly built",
)
_GROUNDING_SIGNALS = (
    "where is",
    "where are",
    "locate",
    "location of",
    "find the",
    "highlight",
    "mark the",
    "ground",
    "bounding box",
    "region of",
)
_SAR_SIGNALS = (
    "sar",
    "synthetic aperture radar",
    "radar",
)
_OPTICAL_SIGNALS = (
    "optical",
    "multispectral",
    "sentinel-2",
    "landsat",
    "rgb",
)
_SCENE_SIGNALS = (
    "what is in",
    "what are in",
    "what do you see",
    "describe",
    "identify",
    "which objects",
    "scene",
    "satellite image",
)


def _normalise(query: str) -> str:
    """Lowercase and collapse whitespace for stable signal matching."""

    return re.sub(r"\s+", " ", query.strip().lower())


def _match_signals(text: str, signals: Iterable[str]) -> List[str]:
    """Return the signals present in the normalised query."""

    return [signal for signal in signals if signal in text]


def _score_signal_groups(text: str) -> List[Tuple[Intent, List[str]]]:
    """Compute transparent intent evidence from ordered signal groups."""

    return [
        (Intent.CHANGE_ANALYSIS, _match_signals(text, _CHANGE_SIGNALS)),
        (Intent.GROUNDING, _match_signals(text, _GROUNDING_SIGNALS)),
        (Intent.MULTIMODAL_ANALYSIS, _match_signals(text, _SAR_SIGNALS + _OPTICAL_SIGNALS)),
        (Intent.SCENE_UNDERSTANDING, _match_signals(text, _SCENE_SIGNALS)),
    ]


def route_query(query: str) -> RouteDecision:
    """Route a natural-language query to the smallest useful tool set.

    The router is intentionally deterministic for the MVP. When no reliable
    intent signals are found, it returns ``UNKNOWN`` rather than guessing.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    text = _normalise(query)
    scored = _score_signal_groups(text)
    scored.sort(key=lambda item: len(item[1]), reverse=True)

    intent, matched = scored[0]

    # Multimodal routing gets priority when both sensor families are explicit.
    has_sar = bool(_match_signals(text, _SAR_SIGNALS))
    has_optical = bool(_match_signals(text, _OPTICAL_SIGNALS))
    if has_sar and has_optical:
        intent = Intent.MULTIMODAL_ANALYSIS
        matched = _match_signals(text, _SAR_SIGNALS + _OPTICAL_SIGNALS)

    if not matched:
        return RouteDecision(intent=Intent.UNKNOWN)

    requires_temporal = intent == Intent.CHANGE_ANALYSIS
    requires_grounding = intent == Intent.GROUNDING

    tools: List[ToolName]
    if intent == Intent.CHANGE_ANALYSIS:
        tools = [ToolName.CHANGE_DETECTION]
    elif intent == Intent.GROUNDING:
        tools = [ToolName.GROUNDING]
    elif intent == Intent.MULTIMODAL_ANALYSIS:
        tools = [ToolName.OPTICAL_SAR]
    else:
        tools = [ToolName.VQA]

    # A grounded change request needs both analysis stages.
    if intent == Intent.CHANGE_ANALYSIS and requires_grounding:
        tools = [ToolName.CHANGE_DETECTION, ToolName.GROUNDING]

    confidence = min(1.0, 0.55 + 0.10 * len(matched))

    return RouteDecision(
        intent=intent,
        tools=tools,
        requires_temporal=requires_temporal,
        requires_optical=has_optical or intent == Intent.MULTIMODAL_ANALYSIS,
        requires_sar=has_sar or intent == Intent.MULTIMODAL_ANALYSIS,
        requires_grounding=requires_grounding,
        confidence=round(confidence, 2),
        matched_signals=matched,
    )
