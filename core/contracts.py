"""Common specialist contracts for M1-M3 model implementations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpecialistResult:
    """Standard output consumed by M4 regardless of model implementation."""
    task: str
    model: str
    status: str
    confidence: float
    claim: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()
