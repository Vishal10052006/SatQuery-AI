"""
CLI and execution script for SatQuery-AI Module 5 (M5):
GIS, Geospatial Processing and Evidence Generation.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from data.mock.generate_mock import create_sample_mock_data
from geospatial.integration import (
    process_m2_result,
    process_m3_result,
    process_m4_result,
)
from geospatial.pipeline import run_geospatial_pipeline


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
        "--m3-result",
        type=str,
        default=None,
        help="Path to upstream M3 multimodal (Optical + SAR) analysis JSON",
    )
    parser.add_argument(
        "--m4-result",
        type=str,
        default=None,
        help="Path to upstream M4 SpecialistResult JSON wrapper",
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

    if args.m4_result:
        m4_path = Path(args.m4_result)
        if not m4_path.exists():
            raise FileNotFoundError(f"M4 result file not found: {m4_path}")
        with open(m4_path, "r", encoding="utf-8") as f:
            m4_data = json.load(f)

        print("\n========================================================")
        print("   SatQuery-AI - Module 5: Geospatial Processing Engine   ")
        print("   (Processing Upstream M4 SpecialistResult Payload)      ")
        print("========================================================")
        print(f"Input Payload     : {args.m4_result}")
        print(f"Task              : {m4_data.get('task')}")
        print(f"Model             : {m4_data.get('model')}")
        print(f"Confidence        : {float(m4_data.get('confidence', 0.75)):.2f}")
        print(f"Claim             : {m4_data.get('claim', '')}")
        print(f"Output Directory  : {args.output}")
        print("--------------------------------------------------------")

        evidence = process_m4_result(m4_path, output_dir=args.output)
    elif args.m2_result:
        m2_path = Path(args.m2_result)
        if not m2_path.exists():
            raise FileNotFoundError(f"M2 result file not found: {m2_path}")
        with open(m2_path, "r", encoding="utf-8") as f:
            m2_data = json.load(f)

        geotiff = m2_data.get("reference_image") or m2_data.get("reference_geotiff")
        mask = m2_data.get("change_mask") or m2_data.get("change_mask_path")
        pred_obj = m2_data.get("prediction", {})
        target = (
            m2_data.get("target")
            or pred_obj.get("predicted_class_name")
            or pred_obj.get("predicted_class")
            or args.target
        )
        conf_val = m2_data.get("confidence", args.confidence)
        if isinstance(conf_val, dict):
            confidence = float(conf_val.get("score", pred_obj.get("model_confidence", 0.85)))
        else:
            confidence = float(conf_val)

        print("\n========================================================")
        print("   SatQuery-AI - Module 5: Geospatial Processing Engine   ")
        print("   (Processing Upstream M2 Detection Payload)             ")
        print("========================================================")
        print(f"Input Payload     : {args.m2_result}")
        if geotiff:
            print(f"Reference Image   : {geotiff}")
        if mask:
            print(f"Change Mask       : {mask}")
        print(f"Target Category   : {target}")
        print(f"Confidence        : {confidence:.2f}")
        print(f"Output Directory  : {args.output}")
        print("--------------------------------------------------------")

        evidence = process_m2_result(m2_path, output_dir=args.output)
    elif args.m3_result:
        m3_path = Path(args.m3_result)
        if not m3_path.exists():
            raise FileNotFoundError(f"M3 result file not found: {m3_path}")
        with open(m3_path, "r", encoding="utf-8") as f:
            m3_data = json.load(f)

        pred_obj = m3_data.get("prediction", {})
        target = pred_obj.get("predicted_class_name") or pred_obj.get("predicted_class") or "Vegetation"
        conf_obj = m3_data.get("confidence", {})
        confidence = float(conf_obj.get("score", pred_obj.get("model_confidence", 0.85))) if isinstance(conf_obj, dict) else float(conf_obj)
        reg_obj = m3_data.get("registration", {})

        print("\n========================================================")
        print("   SatQuery-AI - Module 5: Geospatial Processing Engine   ")
        print("   (Processing Upstream M3 Optical+SAR Pipeline Payload)  ")
        print("========================================================")
        print(f"Input Payload     : {args.m3_result}")
        print(f"Target Category   : {target}")
        print(f"Confidence        : {confidence:.2f}")
        print(f"Registration Pass : {reg_obj.get('passed', False)}")
        print(f"Registration Score: {reg_obj.get('registration_score', 0.0)}")
        print(f"Output Directory  : {args.output}")
        print("--------------------------------------------------------")

        evidence = process_m3_result(m3_path, output_dir=args.output)
    else:
        geotiff = args.geotiff or str(mock_dir / "sample.tif")
        mask = args.mask or str(mock_dir / "change_mask.npy")
        bboxes = [[40, 50, 90, 100], [140, 160, 200, 185]]
        target = args.target
        confidence = args.confidence

        print("\n========================================================")
        print("   SatQuery-AI - Module 5: Geospatial Processing Engine   ")
        print("========================================================")
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
    poly_cnt = len(evidence["polygons"]) if evidence.get("polygons") is not None else 0
    print(f"  * Polygons Detected: {poly_cnt}")
    if evidence.get("area"):
        print(f"  * Total Area (m²)  : {evidence['area']['total_sq_meters']:,.1f} m²")
        print(f"  * Total Area (ha)  : {evidence['area']['total_hectares']:.4f} ha")
    else:
        print("  * Total Area       : null (non-georeferenced)")
    print(f"  * Evidence JSON    : {evidence.get('evidence_path', str(Path(args.output, 'evidence.json').as_posix()))}")
    print(f"  * GeoJSON Path     : {evidence['geojson_path']}")
    print(f"  * Interactive Map  : {evidence['map_path']}\n")

    return evidence


if __name__ == "__main__":
    main()
