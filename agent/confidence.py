"""Evidence-derived confidence scoring for SatQuery AI.

This score is intentionally separate from router confidence. It estimates how
well the supplied downstream evidence supports a result and reports missing,
failed, or disagreeing evidence instead of silently treating it as support.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return _clamp(float(value))
    except (TypeError, ValueError):
        return None


def calculate_confidence(
    evidence: Iterable[Mapping[str, Any]],
    *,
    disagreement: float = 0.0,
) -> dict:
    """Calculate transparent confidence from evidence quality.

    Weighting:
    - specialist/model confidence: 50%
    - successful evidence coverage: 30%
    - sensor/cross-source agreement: 20%

    ``disagreement`` is a penalty in [0, 1]. It can be supplied by a later
    cross-modal/temporal validator when sources materially conflict.
    """

    items = list(evidence)
    successful = [item for item in items if item.get("status") == "success"]
    known_conf = [_as_float(item.get("confidence")) for item in successful]
    known_conf = [value for value in known_conf if value is not None]

    model_score = sum(known_conf) / len(known_conf) if known_conf else 0.0
    coverage = len(successful) / len(items) if items else 0.0
    agreement = _clamp(1.0 - disagreement)

    score = _clamp(
        0.50 * model_score
        + 0.30 * coverage
        + 0.20 * agreement
    )

    return {
        "score": round(score, 3),
        "percent": round(score * 100, 1),
        "components": {
            "model_confidence": round(model_score, 3),
            "evidence_coverage": round(coverage, 3),
            "cross_source_agreement": round(agreement, 3),
        },
        "disagreement_penalty": round(_clamp(disagreement), 3),
        "evidence_count": len(items),
        "successful_evidence_count": len(successful),
    }
