"""M3: optical + SAR corroboration contract."""
from __future__ import annotations
from pathlib import Path
from core.contracts import SpecialistResult


def run_optical_sar(optical: str | None, sar: str | None) -> SpecialistResult:
    """Validate sensor availability and expose a fusion-ready result."""
    optical_ok = bool(optical and Path(optical).exists())
    sar_ok = bool(sar and Path(sar).exists())
    return SpecialistResult(
        task="optical_sar", model="optical-sar-adapter-baseline",
        status="ready" if optical_ok and sar_ok else "fallback", confidence=0.20,
        claim="Optical + SAR fusion interface is ready; real cross-modal weights are not activated.",
        evidence={"optical_exists": optical_ok, "sar_exists": sar_ok, "modalities": ["optical", "sar"]},
    )
