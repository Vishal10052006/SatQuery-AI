"""M2: Remote-Sensing Change Analysis Subsystem.

Provides:
- run_m2: Unified end-to-end change analysis pipeline.
- M2Result: Structured result schema.
- detect_changes: Deterministic bi-temporal change baseline.
- run_change_detection: Specialist adapter for M4 integration.
- run_grounding: Text-to-region grounding adapter.
- RCDAdapter: Referring Change Detection adapter.
- GroundingAdapter: Text-guided grounding adapter.
- validate_bitemporal_inputs: Bi-temporal validation.
"""
from __future__ import annotations

from .adapter import run_change_detection, run_grounding
from .baseline import detect_changes
from .grounding import GroundingAdapter
from .pipeline import M2Result, run_m2
from .rcd import RCDAdapter
from .validation import validate_bitemporal_inputs

__all__ = [
    "run_m2",
    "M2Result",
    "detect_changes",
    "run_change_detection",
    "run_grounding",
    "RCDAdapter",
    "GroundingAdapter",
    "validate_bitemporal_inputs",
]
