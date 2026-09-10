"""Export complete M3 multimodal analysis result as a structured JSON file for M5 GIS comparison."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.optical_sar.config import OpticalSARConfig
from modules.optical_sar.fusion.dataset import CLASS_NAMES
from modules.optical_sar.fusion.model import OpticalSARModel
from modules.optical_sar.pipeline import run_optical_sar_pipeline


def export_m5_json(output_json_path: str = "data/results/m3_output_for_m5.json") -> Path:
    optical_path = REPO_ROOT / "data" / "optical" / "sentinel2_real.tif"
    sar_path = REPO_ROOT / "data" / "sar" / "sentinel1_real.tif"
    scl_path = REPO_ROOT / "data" / "optical" / "sentinel2_scl.tif"
    weights_path = REPO_ROOT / "modules" / "optical_sar" / "weights" / "m3_optical_sar_model.pth"

    # 1. Load trained model
    model = OpticalSARModel(
        fusion_type="feature",
        optical_channels=4,
        sar_channels=2,
        feature_dim=128,
        num_classes=len(CLASS_NAMES),
        weights_path=str(weights_path) if weights_path.exists() else None,
    )

    # 2. Configure pipeline
    cfg = OpticalSARConfig()
    cfg.optical.band_mapping = {"B02": 1, "B03": 2, "B04": 3, "B08": 4}
    cfg.sar.is_already_calibrated = True
    cfg.sar.polarizations = ["VV", "VH"]
    cfg.fusion.num_classes = len(CLASS_NAMES)

    # 3. Run pipeline
    result = run_optical_sar_pipeline(
        optical_path=optical_path,
        sar_path=sar_path,
        cloud_mask_path=scl_path,
        config=cfg,
        model=model,
        run_inference=True,
    )

    # 4. Generate clean JSON dictionary
    m3_dict = result.to_dict()
    
    # Enhance with M5 GIS-specific comparison helpers
    min_x, min_y, max_x, max_y = m3_dict["metadata"]["bounds"]
    m3_dict["gis_evidence"] = {
        "crs": m3_dict["metadata"]["crs"],
        "bounds": {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        },
        "polygon_geojson": {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y],
            ]],
        },
        "resolution_meters": m3_dict["metadata"]["resolution"],
        "pixel_dimensions": m3_dict["metadata"]["spatial_shape"],
    }
    
    # Add human-readable class name to prediction
    pred_idx = m3_dict["prediction"].get("predicted_class")
    if pred_idx is not None and pred_idx < len(CLASS_NAMES):
        m3_dict["prediction"]["predicted_class_name"] = CLASS_NAMES[pred_idx]
        m3_dict["prediction"]["class_probabilities_breakdown"] = {
            name: round(prob, 4)
            for name, prob in zip(CLASS_NAMES, m3_dict["prediction"].get("probabilities", []))
        }

    # 5. Write to JSON file
    out_file = REPO_ROOT / output_json_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(m3_dict, f, indent=2)

    return out_file


if __name__ == "__main__":
    out = export_m5_json()
    print(f"Exported JSON for M5 to: {out}")
