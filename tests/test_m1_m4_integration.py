"""Integration test for the real M4 -> M1 EarthDial path."""

from pathlib import Path

from app.agents.bootstrap import build_m4_controller
from app.query.schemas import (
    ExecutionStatus,
    QueryRequest,
    ToolName,
)
from m1_earthdial.config import EarthDialConfig
from m1_earthdial.earthdial_adapter import EarthDialAdapter


def test_m4_calls_real_m1_earthdial_adapter() -> None:
    """
    Verify the complete M4 -> M1 integration path.

    This test uses M1's real EarthDialAdapter with its offline
    mock backend. No M4 response is hard-coded.
    """

    image_path = Path(
        "m1_earthdial/examples/sample_satellite.jpg"
    )

    assert image_path.exists()

    # Use the actual M1 adapter with its supported offline backend.
    m1 = EarthDialAdapter(
        config=EarthDialConfig(
            backend="mock",
        )
    )

    # Inject the real M1 implementation into M4.
    controller = build_m4_controller(
        vqa=m1,
    )

    response = controller.run(
        QueryRequest(
            query="Describe this satellite image.",
            images=[str(image_path)],
        )
    )

    # ---------------------------------------------------------
    # Verify M4 completed successfully.
    # ---------------------------------------------------------

    assert response.status == ExecutionStatus.SUCCESS
    assert response.error is None

    # ---------------------------------------------------------
    # Verify M4 actually routed to M1 VQA.
    # ---------------------------------------------------------

    assert len(response.results) == 1

    result = response.results[0]

    assert result.tool == ToolName.M1_VQA
    assert result.status == ExecutionStatus.SUCCESS

    # ---------------------------------------------------------
    # Verify the answer came from M1.
    # ---------------------------------------------------------

    assert result.data["answer"]
    assert result.data["model"] == "EarthDial"

    # M1 intentionally does not fabricate confidence.
    assert result.confidence == 0.0

    # M1 should provide specialist evidence.
    assert result.data["evidence"] is not None

    # The synthesized M4 answer should contain the M1 answer.
    assert response.answer
