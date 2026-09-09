"""
CLI and execution script for SatQuery-AI Module 5 (M5):
GIS, Geospatial Processing and Evidence Generation.
"""

import argparse
import json
from pathlib import Path
from geospatial.pipeline import run_geospatial_pipeline
from data.mock.generate_mock import create_sample_mock_data


def main():
    parser = argparse.ArgumentParser(
        description="SatQuery-AI - Module 5 (M5): Geospatial Processing & Evidence Generation"
    )
    parser.add_argument(
        "--geotiff",
        type=str,
        default=None,
        help="Path to reference GeoTIFF file (default: data/mock/sample.tif)",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to change mask (.npy or raster, default: data/mock/change_mask.npy)",
    )
    parser.add_argument(
        "--m2-result",
        type=str,
        default=None,
        help="Path to upstream M2 detection JSON (overrides other inputs if provided)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="deforestation",
        help="Target classification name (default: deforestation)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.94,
        help="Detection confidence score between 0.0 and 1.0 (default: 0.94)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for evidence artifacts (default: output)",
    )

    args = parser.parse_args()

    # Ensure mock data exists if using defaults
    mock_dir = Path("data/mock")
    sample_tif = mock_dir / "sample.tif"
    if not sample_tif.exists():
        print("[*] Generating mock GeoTIFF and change mask...")
        create_sample_mock_data(mock_dir)

    if args.m2_result:
        m2_path = Path(args.m2_result)
        if not m2_path.exists():
            raise FileNotFoundError(f"M2 result file not found: {m2_path}")
        with open(m2_path, "r", encoding="utf-8") as f:
            m2_data = json.load(f)

        geotiff = m2_data.get("reference_geotiff")
        mask = m2_data.get("change_mask_path")
        bboxes = m2_data.get("bounding_boxes_pixel", [])
        target = m2_data.get("target", args.target)
        confidence = m2_data.get("confidence", args.confidence)
    else:
        geotiff = args.geotiff or str(mock_dir / "sample.tif")
        mask = args.mask or str(mock_dir / "change_mask.npy")
        bboxes = [[40, 50, 90, 100], [140, 160, 200, 185]]
        target = args.target
        confidence = args.confidence

    print(f"\n========================================================")
    print(f"   SatQuery-AI - Module 5: Geospatial Processing Engine   ")
    print(f"========================================================")
    print(f"Reference GeoTIFF : {geotiff}")
    print(f"Change Mask       : {mask}")
    print(f"Target Category   : {target}")
    print(f"Confidence        : {confidence:.2f}")
    print(f"Output Directory  : {args.output}")
    print("--------------------------------------------------------")

    evidence = run_geospatial_pipeline(
        reference_geotiff=geotiff,
        change_mask=mask,
        bounding_boxes=bboxes,
        target=target,
        confidence=confidence,
        output_dir=args.output,
    )

    print("\n[SUCCESS] M5 Pipeline execution finished successfully!")
    print(f"  * Change Confirmed : {evidence['change_detected']}")
    print(f"  * Polygons Detected: {len(evidence['polygons'])}")
    print(f"  * Total Area (m²)  : {evidence['area']['total_sq_meters']:,.1f} m²")
    print(f"  * Total Area (ha)  : {evidence['area']['total_hectares']:.4f} ha")
    print(f"  * Evidence JSON    : {evidence['geojson_path']}")
    print(f"  * GeoJSON Path     : {args.output}/evidence.geojson")
    print(f"  * Interactive Map  : {evidence['map_path']}\n")

    return evidence


if __name__ == "__main__":
    main()
