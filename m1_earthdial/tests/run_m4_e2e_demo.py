"""
Demonstration of M4 Agent Orchestrator calling M1 EarthDial VLM.
Runs locally without requiring Google Colab or external GPU servers.
"""

import sys
from pathlib import Path

# Add project root to python path
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from app.agents.controller import AgentController
from app.query.registry import ToolRegistry
from app.adapters.register import register_specialists
from m1_earthdial.earthdial_adapter import EarthDialAdapter
from app.query.schemas import QueryRequest


def run_m4_m1_demo():
    print("=" * 70)
    print("🤖 SatQuery AI: Full M4 Agent -> M1 EarthDial VLM Pipeline Demo")
    print("=" * 70)

    # 1. Initialize M1 adapter (uses local mock simulation on laptop)
    m1_adapter = EarthDialAdapter()

    # 2. Register M1 with M4 specialist registry
    registry = ToolRegistry()
    register_specialists(registry, vqa=m1_adapter)

    # 3. Build M4 Agent controller
    controller = AgentController(registry=registry)

    sample_image = "m1_earthdial/examples/sample_satellite.jpg"
    context = {"images": [sample_image]}

    demo_queries = [
        "What can you see in this satellite image?",
        "What type of land cover is visible in this satellite scene?",
        "Is there any water body or river in this satellite image?",
    ]

    for idx, query in enumerate(demo_queries, start=1):
        print(f"\n[{idx}/3] USER QUERY: \"{query}\"")
        response = controller.run(
            request=QueryRequest(query=query),
            context=context
        )

        print(f"  -> M4 Status:     {response.status.value}")
        print(f"  -> M4 Intent:     {response.intent.value if response.intent else 'None'}")
        print(f"  -> M4 Confidence: {response.confidence}")
        print(f"  -> Final Answer:  {response.answer}")
        print(f"  -> Trace:")
        for step in response.trace:
            print(f"       * {step}")
        print("-" * 70)

    print("\n[+] M4 -> M1 pipeline successfully verified locally!")


if __name__ == "__main__":
    run_m4_m1_demo()
