"""Integration tests for the native M3 -> M4 adapter boundary."""

from pathlib import Path

import affine
import numpy as np
import rasterio

from app.adapters.m3_optical_sar_adapter import (
    build_optical_sar_adapter,
)
from app.query.schemas import ExecutionStatus, ToolName


def _create_raster(
    path: Path,
    bands: int,
    height: int = 64,
    width: int = 64,
) -> Path:
    """Create a georeferenced synthetic raster for integration testing."""
    transform = affine.Affine(
        10.0,
        0.0,
        500000.0,
        0.0,
        -10.0,
        5200000.0,
    )

    rng = np.random.default_rng(42 + bands)

    data = rng.random(
        (bands, height, width),
        dtype=np.float32,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


def test_native_m3_executes_through_m4_adapter(tmp_path: Path) -> None:
    """Verify M4 -> native M3 -> trained checkpoint -> ToolResult."""

    optical = _create_raster(
        tmp_path / "optical.tif",
        bands=4,
    )

    sar = _create_raster(
        tmp_path / "sar.tif",
        bands=2,
    )

    adapter = build_optical_sar_adapter()

    result = adapter(
        optical=str(optical),
        sar=str(sar),
        run_inference=True,
    )

    assert result.tool == ToolName.M3_OPTICAL_SAR

    assert result.status in {
        ExecutionStatus.SUCCESS,
        ExecutionStatus.PARTIAL,
    }

    assert 0.0 <= result.confidence <= 1.0

    # Stable M4 payload.
    assert "answer" in result.data
    assert result.data["model"] == "M3-optical-sar"

    # Native M3 result must remain available.
    assert "m3" in result.data

    prediction = result.data["prediction"]

    assert prediction["status"] == "success"
    assert prediction["fusion_type"] == "feature"
    assert prediction["predicted_class"] in range(3)
    assert len(prediction["probabilities"]) == 3

    # Most important assertion:
    # the adapter must reach the real trained checkpoint.
    assert (
        "trained model weights"
        in prediction["notes"].lower()
    )

    # Native registration evidence must survive the adapter.
    assert "registration" in result.data

    # Adapter should expose evidence for both analysis and prediction.
    evidence_types = {
        evidence.type
        for evidence in result.evidence
    }

    assert "optical_sar_analysis" in evidence_types
    assert "prediction" in evidence_types
