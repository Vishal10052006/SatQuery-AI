"""
Command-line interface and standalone runner for SatQuery AI M1 EarthDial.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .earthdial_adapter import EarthDialAdapter, analyze_image
from .config import EarthDialConfig


def run_cli():
    parser = argparse.ArgumentParser(
        description="SatQuery AI - Module M1: EarthDial Visual Question Answering CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the satellite image file"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="Natural language question about the satellite image"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default=os.getenv("EARTHDIAL_BACKEND", "auto"),
        choices=["auto", "remote", "direct", "mock"],
        help="Inference backend to use"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.getenv("EARTHDIAL_API_URL", "http://localhost:8000"),
        help="Remote Colab GPU server URL (used when backend is 'remote')"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam search size"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the JSON output file (defaults to m1_earthdial/outputs/result_<timestamp>.json)"
    )

    args = parser.parse_args()

    config = EarthDialConfig(
        backend=args.backend,
        api_url=args.api_url,
    )
    adapter = EarthDialAdapter(config=config)

    print(f"[*] Processing image: {args.image}")
    print(f"[*] Question: {args.question}")
    print(f"[*] Backend: {adapter._resolved_backend}")

    response = adapter.analyze(
        image_path=args.image,
        question=args.question,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )
    result_dict = response.to_dict()

    print("\n=== Structured M1 EarthDial Response ===")
    print(json.dumps(result_dict, indent=2))

    # Save output
    output_path = args.output
    if output_path is None:
        out_dir = Path("./m1_earthdial/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_stem = Path(args.image).stem
        output_path = out_dir / f"result_{img_stem}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n[+] Saved response to: {output_path}")

    if not response.success:
        sys.exit(1)


if __name__ == "__main__":
    run_cli()

