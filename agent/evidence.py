"""Evidence collection and cross-checking for SatQuery AI.

M4 records the outputs produced by specialist tools and derives transparent
support/disagreement signals. This module does not invent scientific results.
It only evaluates the evidence supplied by downstream tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class EvidenceItem:
    """One evidence contribution from a specialist analysis step."""

    source: str
    status: str
    result: Any = None
    confidence: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "status": self.status,
            "result": self.result,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


def collect_evidence(execution_results: Iterable[dict]) -> List[EvidenceItem]:
    """Convert executor step results into normalized evidence records."""

    evidence: List[EvidenceItem] = []
    for item in execution_results:
        evidence.append(
            EvidenceItem(
                source=str(item.get("tool", "unknown")),
                status=str(item.get("status", "unknown")),
                result=item.get("result"),
                confidence=item.get("confidence"),
                metadata={
                    "error": item.get("error"),
                    "message": item.get("message"),
                },
            )
        )
    return evidence


def summarize_evidence(evidence: Iterable[EvidenceItem]) -> dict:
    """Produce an audit-friendly summary without fabricating missing evidence."""

    items = list(evidence)
    successful = [item for item in items if item.status == "success"]
    unavailable = [item for item in items if item.status == "unavailable"]
    failed = [item for item in items if item.status == "failed"]

    return {
        "total": len(items),
        "successful": len(successful),
        "unavailable": len(unavailable),
        "failed": len(failed),
        "sources": [item.source for item in items],
        "items": [item.to_dict() for item in items],
    }
