"""
Integration test for the native M2 -> M4 execution path.

The test exercises:

    M4 Controller
        -> M4 Planner
        -> M4 Executor
        -> M2 adapter
        -> native M2 change detector
        -> M4 ToolResult
        -> M4 AgentResponse
"""

from pathlib import Path

from PIL import Image

from app.agents.bootstrap import build_m4_controller
from app.query.schemas import (
    ExecutionStatus,
    QueryRequest,
    ToolName,
)
from models.change.adapter import run_change_detection


def _create_test_image(
    path: Path,
    *,
    changed: bool,
) -> Path:
    """Create a deterministic 64x64 test image."""

    image = Image.new("L", (64, 64), 50)

    if changed:
        # Exactly 100 changed pixels.
        for x in range(20, 30):
            for y in range(20, 30):
                image.putpixel((x, y), 230)

    image.save(path)

    return path


def test_m2_executes_through_real_m4_controller(
    tmp_path: Path,
) -> None:
    """Verify the complete M4 -> M2 runtime path."""

    before = _create_test_image(
        tmp_path / "before.png",
        changed=False,
    )

    after = _create_test_image(
        tmp_path / "after.png",
        changed=True,
    )

    # Inject the REAL M2 implementation.
    controller = build_m4_controller(
        change_detection=run_change_detection,
    )

    response = controller.run(
        QueryRequest(
            query="What changed between these two images?"
        ),
        context={
            "before": str(before),
            "after": str(after),
        },
    )

    # ---------------------------------------------------------
    # Verify M4 completed successfully.
    # ---------------------------------------------------------

    assert response.status == ExecutionStatus.SUCCESS
    assert response.answer

    # ---------------------------------------------------------
    # Verify M4 actually executed M2.
    # ---------------------------------------------------------

    assert len(response.results) == 1

    result = response.results[0]

    assert result.tool == ToolName.M2_CHANGE_DETECTION
    assert result.status == ExecutionStatus.SUCCESS

    # ---------------------------------------------------------
    # Verify native M2 evidence survived the M4 boundary.
    # ---------------------------------------------------------

    evidence = result.data["evidence"]

    assert evidence["changed_pixels"] == 100
    assert evidence["number_of_regions"] == 1
    assert evidence["changed_fraction"] > 0.0
    assert evidence["region_sizes"] == [100]
    assert evidence["detector_type"] == "baseline"

    # M2's native claim must reach M4.
    assert result.data["answer"]
    assert result.data["answer"] in response.answer
