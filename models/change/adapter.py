"""M2: change-detection and grounding adapters for M4 integration."""
from __future__ import annotations

from pathlib import Path

from core.contracts import SpecialistResult

from .grounding import GroundingAdapter
from .pipeline import run_m2


def run_change_detection(
    before: str | None,
    after: str | None,
    target: str | None = None,
    output_dir: str | Path | None = None,
) -> SpecialistResult:
    """Run the M2 change detection pipeline and adapt its result to the M4 contract.

    ``fallback_baseline`` is a native M2 detector status: it means a deterministic
    temporal-difference baseline was used because target-guided RCD inference was
    unavailable.  M4 historically treats a returned SpecialistResult as an
    operationally successful tool invocation, so the adapter exposes ``success``
    at the contract boundary while preserving the native detector status in
    evidence.  This keeps orchestration compatibility without hiding the
    scientific limitation from downstream consumers.
    """
    if before and after and Path(before).exists() and Path(after).exists():
        m2_result = run_m2(
            before_path=before,
            after_path=after,
            target=target,
            output_dir=output_dir,
        )
        specialist_result = m2_result.to_specialist_result()

        if m2_result.status == "fallback_baseline":
            specialist_result.evidence["native_status"] = m2_result.status
            specialist_result.evidence["operational_status"] = "success"
            specialist_result.status = "success"

        return specialist_result

    return SpecialistResult(
        task="change_detection",
        model="pixel-difference-baseline",
        status="awaiting_input",
        confidence=0.0,
        claim="Provide before and after image paths to run bi-temporal change detection.",
        evidence={
            "before_exists": bool(before and Path(before).exists()),
            "after_exists": bool(after and Path(after).exists()),
            "target": target,
        },
    )


def run_grounding(
    image_path: str | None,
    target: str,
) -> SpecialistResult:
    """Expose text-to-region grounding without inventing fake coordinates."""
    adapter = GroundingAdapter()
    return adapter.ground_target(image_path, target).to_specialist_result()
