"""Structured query understanding for satellite-analysis requests."""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class QuerySpec:
    raw_query: str
    intent: str
    target: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    optical: bool = False
    sar: bool = False
    needs_localization: bool = False
    needs_explanation: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


_TARGETS = {
    "building": "buildings", "buildings": "buildings", "construction": "new construction",
    "road": "roads", "roads": "roads", "water": "water bodies", "river": "water bodies",
    "forest": "forest", "vegetation": "vegetation", "crop": "cropland",
}


def _extract_location(text: str) -> Optional[str]:
    """Extract a simple location phrase after common location prepositions."""
    match = re.search(r"\b(?:in|near|around|at)\s+([A-Za-z][A-Za-z .'-]{1,50}?)(?=\s+(?:between|from|during|in|on|using|with|and)\b|[,.!?]|$)", text, re.I)
    return match.group(1).strip() if match else None


def parse_query(query: str) -> QuerySpec:
    """Extract intent, target, location, sensor and coarse temporal entities."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    text = query.strip().lower()
    change = any(x in text for x in ("change", "changed", "difference", "compare", "between", "before", "after", "increase", "decrease", "new construction", "newly built"))
    grounding = any(x in text for x in ("where", "locate", "highlight", "where did", "where are", "show me where"))
    sar = "sar" in text or "radar" in text
    optical = any(x in text for x in ("optical", "sentinel-2", "landsat", "rgb"))
    target = next((v for k, v in _TARGETS.items() if k in text), None)
    if change:
        intent = "change_analysis"
    elif sar and optical:
        intent = "multimodal_analysis"
    elif grounding:
        intent = "grounding"
    else:
        intent = "scene_understanding"
    years = re.findall(r"\b(20\d{2})\b", text)
    start_date = f"{years[0]}-01-01" if years else None
    end_date = f"{years[1]}-01-01" if len(years) > 1 else None
    return QuerySpec(
        query, intent, target, _extract_location(query), start_date, end_date,
        optical, sar, change or grounding, True,
    )
