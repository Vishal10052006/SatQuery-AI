"""Execute complete M3 Optical + SAR Pipeline on real satellite input data.

Loads real Sentinel-2 and Sentinel-1 rasters, applies preprocessing, reprojection,
registration, 6-channel early fusion, and deep learning inference using the trained
weights checkpoint. Prints the comprehensive end-to-end output.
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.optical_sar.config import OpticalSARConfig
from modules.optical_sar.fusion.dataset import CLASS_NAMES
from modules.optical_sar.fusion.model import OpticalSARModel
from modules.optical_sar.pipeline import run_optical_sar_pipeline


def main():
    optical_path = REPO_ROOT / "data" / "optical" / "sentinel2_real.tif"
    sar_path = REPO_ROOT / "data" / "sar" / "sentinel1_real.tif"
    scl_path = REPO_ROOT / "data" / "optical" / "sentinel2_scl.tif"
    weights_path = REPO_ROOT / "modules" / "optical_sar" / "weights" / "m3_optical_sar_model.pth"

    print("\n" + "=" * 70)
    print("      SAT QUERY — M3 REAL SATELLITE PIPELINE EXECUTION OUTPUT")
    print("=" * 70)

    # 1. Load Trained Deep Learning Model
    print("\n[1/5] Loading Trained M3 Multimodal Model...")
    model = OpticalSARModel(
        fusion_type="feature",
        optical_channels=4,
        sar_channels=2,
        feature_dim=128,
        num_classes=len(CLASS_NAMES),
        weights_path=str(weights_path),
    )
    print(f"      Architecture: Feature Fusion Two-Stream CNN")
    print(f"      Checkpoint:   {weights_path}")
    print(f"      Status:       {'Trained weights loaded successfully' if model.has_trained_weights else 'Untrained'}")

    # 2. Configure Pipeline
    cfg = OpticalSARConfig()
    cfg.optical.band_mapping = {"B02": 1, "B03": 2, "B04": 3, "B08": 4}
    cfg.sar.is_already_calibrated = True
    cfg.sar.polarizations = ["VV", "VH"]
    cfg.fusion.num_classes = len(CLASS_NAMES)

    # 3. Execute Master Pipeline
    print("\n[2/5] Executing End-to-End Pipeline on Real Satellite Inputs...")
    print(f"      Optical Input: {optical_path}")
    print(f"      SAR Input:     {sar_path}")
    print(f"      Cloud / SCL:   {scl_path}")

    result = run_optical_sar_pipeline(
        optical_path=optical_path,
        sar_path=sar_path,
        cloud_mask_path=scl_path,
        config=cfg,
        model=model,
        run_inference=True,
    )

    out_dict = result.to_dict()
    gis_dict = result.to_gis_evidence()

    # 4. Detailed Stage-by-Stage Output Report
    print("\n" + "=" * 70)
    print("                     PIPELINE OUTPUT SUMMARY")
    print("=" * 70)

    print("\n--- A. OPTICAL PREPROCESSING OUTPUT ---")
    opt = out_dict["optical"]
    print(f"  Raster Dimensions:       {opt['shape']} (Bands, Height, Width)")
    print(f"  CRS:                     {opt['crs']}")
    print(f"  Spatial Resolution:      {opt['resolution']} meters/pixel")
    print(f"  Bands Loaded:            {opt['bands']}")
    print(f"  Valid Data Coverage:     {opt['valid_fraction'] * 100:.1f}%")
    print(f"  Cloud Cover Fraction:    {opt['cloud_fraction'] * 100:.1f}%")
    print(f"  Normalized Value Range:  [{opt['stats']['B04']['min']:.4f}, {opt['stats']['B04']['max']:.4f}] (Red band)")
    print(f"  Extracted Mean NDVI:     {opt['stats']['B08']['mean'] - opt['stats']['B04']['mean']:.4f} (NIR - Red contrast)")

    print("\n--- B. SAR PREPROCESSING OUTPUT ---")
    sar = out_dict["sar"]
    print(f"  Raster Dimensions:       {sar['shape']} (Polarizations, Height, Width)")
    print(f"  CRS:                     {sar['crs']}")
    print(f"  Spatial Resolution:      {sar['resolution']} meters/pixel")
    print(f"  Polarizations Loaded:    {sar['polarizations']}")
    print(f"  Calibration Type:        {sar['calibration_type']}")
    print(f"  Speckle Filter Applied:  {sar['speckle_filter']} (kernel=5x5)")
    print(f"  Terrain Correction:      {sar['terrain_corrected']} (DEM omitted; status: {sar['terrain_status']})")
    print(f"  VV Backscatter (dB):     Mean = {sar['stats']['VV']['mean']:.2f} dB, Std = {sar['stats']['VV']['std']:.2f} dB")
    print(f"  VH Backscatter (dB):     Mean = {sar['stats']['VH']['mean']:.2f} dB, Std = {sar['stats']['VH']['std']:.2f} dB")

    print("\n--- C. GEOSPATIAL REGISTRATION & ALIGNMENT OUTPUT ---")
    reg = out_dict["registration"]
    print(f"  Reference Target Grid:   Optical (EPSG:32632)")
    print(f"  Reprojection Method:     {reg['reprojection_method']}")
    print(f"  Fine Alignment Method:   {reg['fine_alignment_method']}")
    print(f"  Spatial Overlap Ratio:   {reg['overlap_ratio'] * 100:.1f}%")
    print(f"  Registration Score:      {reg['registration_score']:.4f}")
    print(f"  Validation Status:       {reg['validation_status']}")

    print("\n--- D. MULTIMODAL FUSION OUTPUT ---")
    fus = out_dict["fusion"]
    print(f"  Early Fusion Tensor:     {fus['shape']} (Channels, Height, Width)")
    print(f"  Channels Composition:    {fus['channels']}")
    print(f"  Data Type:               float32")
    print(f"  Memory Footprint:        {fus['shape'][0] * fus['shape'][1] * fus['shape'][2] * 4 / (1024 * 1024):.2f} MB")

    print("\n--- E. MULTIMODAL DEEP LEARNING MODEL PREDICTION ---")
    pred = out_dict["prediction"]
    pred_idx = pred["predicted_class"]
    pred_class_name = CLASS_NAMES[pred_idx] if pred_idx is not None else "N/A"
    print(f"  Model Architecture:      Two-Stream CNN (Feature Fusion, 128-d)")
    print(f"  Model Weight Status:     {pred['status']}")
    print(f"  Predicted Category:      >>> {pred_class_name} <<< (Class ID: {pred_idx})")
    print(f"  Model Confidence:        {pred['model_confidence'] * 100:.2f}%")
    print("  Class Probability Distribution:")
    for name, p in zip(CLASS_NAMES, pred["probabilities"]):
        bar = "█" * int(p * 30)
        print(f"    - {name:<12}: {p * 100:5.2f}% | {bar}")
    print(f"  Scientific Notes:        {pred['notes']}")

    print("\n--- F. COMPOSITE QUALITY & CONFIDENCE ASSESSMENT ---")
    conf = out_dict["confidence"]
    print(f"  Overall Pipeline Status: {result.status.upper()}")
    print(f"  Composite Confidence:   {conf['score'] * 100:.2f}%")
    print(f"  Quality Rating Level:    {conf['level'].upper()}")
    print(f"  Confidence Component Breakdown:")
    for k, v in conf["components"].items():
        if v is not None:
            print(f"    - {k:<20}: {v * 100:5.2f}% (weight: {conf['weights_used'].get(k.replace('_quality', '_data').replace('model_confidence', 'model_prediction'), 'N/A')})")
    print(f"  Assessment Notes:        {conf['notes']}")

    print("\n--- G. MODULE HANDOFF PAYLOADS (M4 AGENT & M5 GIS) ---")
    print(f"  GIS Bounding Box (UTM):  {gis_dict['bounds']}")
    print(f"  GIS Target CRS:          {gis_dict['crs']}")
    print(f"  JSON Payload Keys:       {list(out_dict.keys())}")
    print("\n" + "=" * 70)
    print("               PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
