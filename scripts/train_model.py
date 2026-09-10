"""Train the M3 Optical + SAR Multimodal Model on Real Paired Satellite Rasters.

This script executes the complete deep learning training pipeline:
1. Loads real Sentinel-2 (B02, B03, B04, B08) and Sentinel-1 (VV, VH) rasters.
2. Extracts paired spatial patches with ESA SCL ground-truth land cover labels.
3. Splits into train, validation, and hold-out test sets.
4. Trains the M3 OpticalSARModel (Feature Fusion CNN) with cross-entropy loss.
5. Evaluates the trained model on test data, reporting Accuracy, Precision, Recall, F1, and Confusion Matrix.
6. Saves the best model checkpoint to .pth.
7. Demonstrates real inference on test samples and through the master pipeline.
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.optical_sar.config import OpticalSARConfig
from modules.optical_sar.fusion.dataset import CLASS_NAMES, create_real_dataloaders
from modules.optical_sar.fusion.model import OpticalSARModel
from modules.optical_sar.fusion.train import evaluate_multimodal_model, train_multimodal_model
from modules.optical_sar.pipeline import run_optical_sar_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m3_training")


def main():
    parser = argparse.ArgumentParser(description="Train M3 Optical + SAR multimodal model on real satellite data.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size in pixels (default: 32)")
    parser.add_argument("--stride", type=int, default=16, help="Patch sampling stride (default: 16)")
    parser.add_argument(
        "--optical-path",
        type=str,
        default=str(REPO_ROOT / "data" / "optical" / "sentinel2_real.tif"),
        help="Path to real Sentinel-2 GeoTIFF",
    )
    parser.add_argument(
        "--sar-path",
        type=str,
        default=str(REPO_ROOT / "data" / "sar" / "sentinel1_real.tif"),
        help="Path to real Sentinel-1 GeoTIFF",
    )
    parser.add_argument(
        "--scl-path",
        type=str,
        default=str(REPO_ROOT / "data" / "optical" / "sentinel2_scl.tif"),
        help="Path to real Sentinel-2 SCL GeoTIFF",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=str(REPO_ROOT / "modules" / "optical_sar" / "weights" / "m3_optical_sar_model.pth"),
        help="Path to save best model checkpoint",
    )
    args = parser.parse_args()

    optical_path = Path(args.optical_path)
    sar_path = Path(args.sar_path)
    scl_path = Path(args.scl_path)
    weights_path = Path(args.weights_path)

    for p, name in [(optical_path, "Optical"), (sar_path, "SAR"), (scl_path, "SCL")]:
        if not p.is_file():
            logger.error(f"Required {name} file not found: {p}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("   SAT QUERY — M3 MULTIMODAL MODEL TRAINING ON REAL DATA")
    print("=" * 60)
    print(f"Optical input:    {optical_path}")
    print(f"SAR input:        {sar_path}")
    print(f"SCL ground truth: {scl_path}")
    print(f"Target classes:   {CLASS_NAMES}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Learning rate:    {args.lr}")
    print("=" * 60 + "\n")

    # Step 1: Create Dataloaders from Real Satellite Rasters
    print(">>> Step 1/6: Extracting Paired Patches & Preparing DataLoaders...")
    train_loader, val_loader, test_loader, class_counts = create_real_dataloaders(
        optical_path=optical_path,
        sar_path=sar_path,
        scl_path=scl_path,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        random_seed=42,
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    n_total = n_train + n_val + n_test
    print(f"    Total patches extracted: {n_total}")
    print(f"    Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"    Class distribution: {class_counts}")

    # Step 2: Initialize M3 OpticalSARModel
    print("\n>>> Step 2/6: Initializing Multimodal OpticalSARModel...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Compute device: {device}")
    model = OpticalSARModel(
        fusion_type="feature",
        optical_channels=4,
        sar_channels=2,
        feature_dim=128,
        num_classes=len(CLASS_NAMES),
        device=str(device),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Architecture: Two-Stream CNN Feature Fusion")
    print(f"    Trainable parameters: {total_params:,}")

    # Step 3: Train Model
    print(f"\n>>> Step 3/6: Training for {args.epochs} Epochs...")
    history = train_multimodal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_path=weights_path,
        device=device,
    )

    final_train_loss = history["train_loss"][-1]
    final_val_loss = history["val_loss"][-1]
    best_val_loss = min(history["val_loss"])
    best_val_acc = max(history["val_acc"])
    print(f"\n    Training complete!")
    print(f"    Best Validation Loss:     {best_val_loss:.4f}")
    print(f"    Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"    Saved Checkpoint:         {weights_path} ({weights_path.stat().st_size / 1024:.1f} KB)")

    # Step 4: Evaluate on Hold-out Test Set
    print("\n>>> Step 4/6: Evaluating on Hold-out Test Set...")
    metrics = evaluate_multimodal_model(model=model, test_loader=test_loader, device=device)
    print(f"    Test Accuracy:            {metrics.accuracy * 100:.2f}%")
    print(f"    Precision (Macro):        {metrics.precision_macro:.4f}")
    print(f"    Precision (Weighted):     {metrics.precision_weighted:.4f}")
    print(f"    Recall (Macro):           {metrics.recall_macro:.4f}")
    print(f"    Recall (Weighted):        {metrics.recall_weighted:.4f}")
    print(f"    F1-Score (Macro):         {metrics.f1_macro:.4f}")
    print(f"    F1-Score (Weighted):      {metrics.f1_weighted:.4f}")
    print("\n    Confusion Matrix (rows: True, cols: Pred):")
    header = "             " + "  ".join(f"{name[:8]:>8}" for name in CLASS_NAMES)
    print(header)
    for i, row in enumerate(metrics.confusion_matrix):
        row_str = f"{CLASS_NAMES[i][:11]:<11}: " + "  ".join(f"{val:>8d}" for val in row)
        print(row_str)

    # Step 5: Demonstrate Sample Test Inferences
    print("\n>>> Step 5/6: Running Sample Test Inferences...")
    test_samples = []
    for opt_x, sar_x, y in test_loader:
        for i in range(min(4, opt_x.size(0))):
            test_samples.append((opt_x[i].numpy(), sar_x[i].numpy(), int(y[i].item())))
        if len(test_samples) >= 4:
            break

    for idx, (opt_sample, sar_sample, true_label) in enumerate(test_samples[:4]):
        pred_res = model.predict(opt_sample, sar_sample)
        pred_name = CLASS_NAMES[pred_res.predicted_class]
        true_name = CLASS_NAMES[true_label]
        match = "MATCH" if pred_res.predicted_class == true_label else "MISMATCH"
        prob_str = ", ".join(f"{CLASS_NAMES[c]}: {p:.3f}" for c, p in enumerate(pred_res.probabilities))
        print(f"    Sample {idx + 1}: True={true_name:<11} | Pred={pred_name:<11} | Conf={pred_res.model_confidence * 100:.1f}% | [{match}]")
        print(f"             Distribution: [{prob_str}]")

    # Step 6: Verify Integration with Master Pipeline
    print("\n>>> Step 6/6: Verifying Full Master Pipeline Integration...")
    cfg = OpticalSARConfig()
    cfg.optical.band_mapping = {"B02": 1, "B03": 2, "B04": 3, "B08": 4}
    cfg.sar.is_already_calibrated = True
    cfg.sar.polarizations = ["VV", "VH"]
    cfg.fusion.num_classes = len(CLASS_NAMES)

    pipeline_result = run_optical_sar_pipeline(
        optical_path=optical_path,
        sar_path=sar_path,
        cloud_mask_path=scl_path,
        config=cfg,
        model=model,
        run_inference=True,
    )
    pipe_dict = pipeline_result.to_dict()
    pred_info = pipe_dict.get("prediction", {})
    print(f"    Pipeline Status:          {pipeline_result.status}")
    print(f"    Model Inference Status:   {pred_info.get('status')}")
    print(f"    Predicted Full-Tile Class:{CLASS_NAMES[pred_info.get('predicted_class')]} (idx: {pred_info.get('predicted_class')})")
    print(f"    Model Confidence:         {pred_info.get('model_confidence') * 100:.2f}%")
    print(f"    Composite Confidence:     {pipe_dict.get('confidence', {}).get('score') * 100:.2f}%")
    print(f"    Quality Assessment:       {pipe_dict.get('confidence', {}).get('level')}")

    # Final Output Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"- Dataset:               Real Copernicus Sentinel-2 + Sentinel-1 paired rasters with ESA SCL ground truth")
    print(f"- Total Patches:         {n_total} ({args.patch_size}x{args.patch_size}, stride {args.stride})")
    print(f"- Split Breakdown:       Train: {n_train} | Validation: {n_val} | Test: {n_test}")
    print(f"- Model Architecture:    OpticalSARModel (Feature Fusion Two-Stream CNN, 128-d latent)")
    print(f"- Target Classes:        {CLASS_NAMES}")
    print(f"- Epochs Trained:        {args.epochs}")
    print(f"- Final Train Loss:      {final_train_loss:.4f}")
    print(f"- Final Val Loss:        {final_val_loss:.4f}")
    print(f"- Test Accuracy:         {metrics.accuracy * 100:.2f}%")
    print(f"- Test Precision (Macro):{metrics.precision_macro:.4f}")
    print(f"- Test Recall (Macro):   {metrics.recall_macro:.4f}")
    print(f"- Test F1-Score (Macro): {metrics.f1_macro:.4f}")
    print(f"- Confusion Matrix:      {metrics.confusion_matrix}")
    print(f"- Saved Model:           {weights_path}")
    print(f"- Pipeline Status:       Verified ({pred_info.get('status')})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
