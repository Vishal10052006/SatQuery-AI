"""
Interactive End-to-End demonstration of SatQuery AI M1 EarthDial.
Executes 3 domain-specific Earth Observation questions against a sample satellite image.
"""

import os
import sys
from pathlib import Path

# Ensure root workspace directory is in python search path
workspace_root = Path(__file__).resolve().parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import json
from m1_earthdial import analyze_image


def run_e2e_demo():
    sample_img = "m1_earthdial/examples/sample_satellite.jpg"
    if not Path(sample_img).exists():
        from m1_earthdial.examples.generate_sample_image import create_sample_satellite_image
        create_sample_satellite_image(sample_img)

    test_questions = [
        "What can you see in this satellite image?",
        "What are the major features and land cover types visible in this scene?",
        "Are there any water bodies, roads, or buildings in this image?",
    ]

    print("=" * 70)
    print("🛰️  SatQuery AI - M1 EarthDial End-to-End VQA Demonstration")
    print("=" * 70)
    print(f"Target Image: {sample_img}\n")

    for i, question in enumerate(test_questions, start=1):
        print(f"\n--- [Query {i}/3] ---")
        print(f"Question: \"{question}\"")
        result = analyze_image(sample_img, question)
        print(f"Answer:   {result['answer']}")
        print(f"Model:    {result['model']}")
        print(f"Backend:  {result['evidence']['backend']}")
        print(f"Success:  {result['success']}")
        print("Structured JSON Result:")
        print(json.dumps(result, indent=2))
        print("-" * 70)


if __name__ == "__main__":
    run_e2e_demo()

