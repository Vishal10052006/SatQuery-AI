# 🛰️ M2 Subsystem Documentation: Remote-Sensing Change Analysis

> **SatQuery AI — Smart India Hackathon 2026 Problem Statement 26167**
>
> *"An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries."*

---

## 1. M2 Role & Core Responsibilities

The **M2 subsystem** is the specialized Earth-observation computer-vision engine responsible for bi-temporal and text-guided change intelligence.

Key capabilities:
1. **Remote-Sensing Preprocessing**: Standardized normalization, multi-band handling, and NoData/NaN masking while strictly preserving raster metadata.
2. **Bi-temporal Input Validation**: Robust format, dimension, band count, and georeferencing checks.
3. **Spatial Alignment / Registration**: Grid matching, affine transform verification, and bilinear resampling of secondary imagery onto the reference grid.
4. **Bi-temporal Change Detection**: Deterministic absolute difference baseline with morphological cleanup and connected component extraction.
5. **Referring Change Detection (RCD)**: Semantic, query-driven change localization (e.g. *"newly constructed buildings"*, *"water body that has increased"*).
6. **Text-Guided Grounding**: Target-to-region spatial bounding and evidence attribution.
7. **Mask & Region Processing**: Extraction of connected change components, bounding boxes (`xmin, ymin, xmax, ymax`), centroids, and approximate boundary polygons.
8. **Geospatial Localization**: Transforming pixel geometry to geographic coordinates via GDAL affine transform.
9. **Area Calculation**: Calculating ground area in square meters for projected CRS (UTM) while safely returning `null` for non-georeferenced or geographic-degree coordinates.
10. **Rich Visualization**: Multi-panel composites (`BEFORE | AFTER | DETECTED CHANGE`), target-annotated overlays with bounding boxes, and binary mask rasters.
11. **Honest Confidence & Quality Metrics**: Auditable quality metrics without hallucinated probabilities.
12. **M4/M5/M6 Integration**: Stable, backward-compatible Python API and schema contracts.

---

## 2. End-to-End Processing Pipeline

```text
       Before Image (T1)       After Image (T2)       Target Query (Text)
              │                       │                       │
              └───────────────┬───────┘                       │
                              ▼                               │
                   Bi-temporal Validation                     │
                    (formats, CRS, NoData)                     │
                              ▼                               │
                   Remote-Sensing Preprocess                  │
                 (multi-band, radiometric scale)              │
                              ▼                               │
                   Spatial Grid Alignment                     │
                (identity or bilinear resample)               │
                              ▼                               │
              ┌───────────────┴────────────────┐              │
              │ Does query specify a target?   │              │
              └───────┬────────────────┬───────┘              │
                  No  │                │ Yes                  ▼
                      │                └─────────────► Referring Change Detection
                      ▼                                (RCDAdapter + fallback)
             Bi-temporal Baseline                              │
                      │                                        │
                      └────────────────┬───────────────────────┘
                                       ▼
                            Change Mask Processing
                         (clean binary mask + filters)
                                       ▼
                          Connected Regions Extraction
                         (IDs, pixel counts, bboxes)
                                       ▼
                            Geospatial Localization
                         (pixel-to-geo, UTM m² area)
                                       ▼
                             Quality & Confidence
                         (alignment & valid pixel score)
                                       ▼
                            Visualization Generation
                         (composite, overlay, mask PNG)
                                       ▼
                                   M2Result
                                       ▼
                         SpecialistResult (for M4)
```

---

## 3. Supported Input Formats

| Format | Extensions | Georeferenced? | Area Unit |
|---|---|---|---|
| **GeoTIFF** | `.tif`, `.tiff` | Yes (via Rasterio/affine) | Square meters (if projected CRS) |
| **JPEG 2000** | `.jp2` | Optional | Square meters (if projected CRS) |
| **PNG** | `.png` | No | Pixels (`area_sq_m = null`) |
| **JPEG / JPG**| `.jpg`, `.jpeg` | No | Pixels (`area_sq_m = null`) |

---

## 4. Module Architecture & Key Classes

### Directory Structure
```text
models/change/
├── __init__.py           # Public exports (run_m2, detect_changes, M2Result, etc.)
├── validation.py         # Bi-temporal input & query validation (validate_bitemporal_inputs)
├── preprocess.py         # Remote-sensing preprocessing (load_and_preprocess)
├── align.py              # Spatial alignment and grid resampling (align_images)
├── mask_processing.py    # Connected components and region extraction (extract_change_regions)
├── baseline.py           # Backward-compatible deterministic detector (detect_changes)
├── rcd.py                # Referring Change Detection adapter (RCDAdapter)
├── grounding.py          # Text-guided grounding adapter (GroundingAdapter)
├── visualize.py          # Visualization generation (generate_m2_visualizations)
├── adapter.py            # Stable M4 adapter (run_change_detection, run_grounding)
└── pipeline.py           # Unified entry point (run_m2) and M2Result schema
```

---

## 5. Python API Usage Examples

### Example 1: General Change Detection (No Target Query)

```python
from models.change import run_m2

result = run_m2(
    before_path="data/t1_optical.tif",
    after_path="data/t2_optical.tif",
    output_dir="outputs/demo_mission",
    config={"threshold": 0.15, "min_pixels": 10},
)

print(f"Status: {result.status}")
print(f"Change detected: {result.change_detected}")
print(f"Changed pixels: {result.changed_pixels} ({result.change_fraction * 100:.2f}%)")
print(f"Number of regions: {len(result.regions)}")
if result.geospatial_reference_available:
    print(f"Ground Area: {result.changed_area_sq_m} m² in CRS {result.crs}")
print(f"Artifacts generated: {result.artifacts}")
```

### Example 2: Referring Change Detection (RCD with Target Query)

```python
from models.change import run_m2

result = run_m2(
    before_path="data/t1_optical.tif",
    after_path="data/t2_optical.tif",
    target="newly constructed buildings",
    output_dir="outputs/rcd_demo",
)

print(f"Target: {result.target}")
print(f"Detector type: {result.detector_type}")  # 'fallback_baseline' if weights unmounted
print(f"Model status: {result.status}")
for region in result.regions:
    print(f"Region {region['region_id']}: BBox={region['bbox_pixel']}, Area={region['area_sq_m']} m²")
```

### Example 3: Consuming from M4 Agentic Orchestrator

```python
from mission.orchestrator import run_mission

mission_result = run_mission(
    "Find newly constructed buildings between before and after images",
    context={
        "before": "data/before.png",
        "after": "data/after.png",
        "output_dir": "outputs/artifacts",
    },
)

change_evidence = next(
    r for r in mission_result["results"] if r["task"] == "change_detection"
)
print(f"Claim: {change_evidence['claim']}")
print(f"Regions: {len(change_evidence['evidence']['regions'])}")
```

---

## 6. Structured Output Contract (`M2Result`)

```json
{
  "task": "change_detection",
  "status": "success",
  "target": "newly constructed buildings",
  "detector": "rcd-fallback-baseline",
  "detector_type": "fallback_baseline",
  "change_detected": true,
  "confidence": 0.75,
  "changed_pixels": 450,
  "change_fraction": 0.043945,
  "mean_difference": 0.038120,
  "regions": [
    {
      "region_id": 1,
      "pixel_count": 450,
      "bbox_pixel": { "xmin": 24, "ymin": 30, "xmax": 54, "ymax": 62 },
      "centroid_pixel": { "x": 39.2, "y": 46.1 },
      "polygon_pixel": [[24, 30], [54, 30], [54, 62], [24, 62], [24, 30]],
      "confidence": 0.88,
      "bbox_geo": {
        "top_left": [500240.0, 2999700.0],
        "top_right": [500550.0, 2999700.0],
        "bottom_right": [500550.0, 2999370.0],
        "bottom_left": [500240.0, 2999370.0]
      },
      "centroid_geo": { "x": 500392.0, "y": 2999539.0 },
      "polygon_geo": [[500240.0, 2999700.0], [500550.0, 2999700.0], [500550.0, 2999370.0], [500240.0, 2999370.0], [500240.0, 2999700.0]],
      "area_sq_m": 45000.0,
      "target": "newly constructed buildings"
    }
  ],
  "artifacts": ["change-4805ef6a941e.png"],
  "composite_path": "outputs/change-4805ef6a941e.png",
  "overlay_path": "outputs/overlay-96d780074c83.png",
  "mask_path": "outputs/mask-1b1c940bc7ec.png",
  "geospatial_reference_available": true,
  "crs": "EPSG:32643",
  "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0],
  "geographic_bbox": {
    "top_left": [500000.0, 3000000.0],
    "bottom_right": [501024.0, 2989760.0]
  },
  "changed_area_sq_m": 45000.0,
  "quality": {
    "alignment_status": "already_aligned",
    "alignment_quality": 1.0,
    "valid_pixel_fraction": 1.0,
    "georeferenced": true,
    "detector_type": "fallback_baseline",
    "model_status": "fallback_baseline",
    "num_regions": 1
  },
  "warnings": []
}
```

---

## 7. Model Integration Status & Checkpoints

In accordance with SatQuery AI's design principles:
- **No Hallucinated Inference**: The subsystem explicitly reports `status="awaiting_model"` or `status="fallback_baseline"` rather than generating fake predictions.
- **Pluggable Weights**: Placing real checkpoint files in `checkpoints/rcd_model.pth` or `checkpoints/grounding_model.pth` will seamlessly transition the adapter from fallback to neural inference.
- **CPU / Lightweight Environment Ready**: The entire pipeline operates deterministically without GPU or heavyweight models, allowing CI and development to proceed smoothly.

---

## 8. Integration with Other Team Modules

| Team | How they interact with M2 |
|---|---|
| **M4 (Agentic Controller)** | Calls `models.change.adapter.run_change_detection()` and `run_grounding()`. Consumes the returned `SpecialistResult` containing evidence, regions, and artifacts. |
| **M5 (Evidence & Geospatial)** | Consumes `regions`, `bbox_geo`, `centroid_geo`, `polygon_geo`, and `changed_area_sq_m` for GIS mapping and spatial cross-checking. |
| **M6 (UI & Mission Workspace)** | Displays the artifacts served via FastAPI (`/outputs/{artifact_name}`): 3-panel composite, bounding-box overlay, and binary change mask. |
