# Module 3 (M3): Optical + SAR Multimodal Analysis

Part of the **Sat Query** Geospatial AI System.

## Overview

The `optical_sar` module is responsible for multimodal Sentinel-1 (SAR) and Sentinel-2 (Optical) preprocessing, cross-modal geospatial reprojection, subpixel alignment, registration validation, channel- and feature-level fusion, baseline neural network inference, and multi-criteria confidence estimation.

```
Optical GeoTIFF (Sentinel-2)       SAR GeoTIFF (Sentinel-1)
       │                                     │
       ▼                                     ▼
[Cloud Mask / SCL]                  [Radiometric Calibration]
       │                                     │
[Percentile Normalization]          [Lee Speckle Filtering]
       │                                     │
       │                            [Terrain Correction Adapter]
       │                                     │
       │                            [Percentile Normalization]
       │                                     │
       └──────────────┬──────────────────────┘
                      │
                      ▼
        [Geospatial Reprojection] (SAR -> Optical Grid)
                      │
                      ▼
         [Cross-Modal Fine Alignment] (Phase Correlation)
                      │
                      ▼
        [Registration Quality Validation]
        (Overlap, Mutual Info, Structural Correlation)
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 [Early Fusion]             [Feature-Level Fusion]
 (6-Channel Composite)       (Two-Stream CNN Backbone)
        │                           │
        └─────────────┬─────────────┘
                      │
                      ▼
         [Multimodal Confidence Engine]
                      │
                      ▼
        [Structured Pipeline Result]
         (Handoff to M2, M4, M5, M6)
```

---

## Directory Layout

```
modules/optical_sar/
├── __init__.py           # Unified public API exports
├── config.py             # Dataclasses and configuration schemas
├── pipeline.py           # Master end-to-end pipeline orchestrator
├── README.md             # This documentation
├── optical/
│   ├── loader.py         # Multi-band GeoTIFF loader with CRS validation
│   ├── cloud_mask.py     # External SCL/cloud mask handler
│   ├── normalization.py  # Min-max and percentile normalization
│   └── features.py       # NDVI, NDWI, and RGB composition
├── sar/
│   ├── loader.py         # VV/VH polarimetric SAR loader
│   ├── calibration.py    # Linear to dB conversion and calibration checks
│   ├── speckle.py        # Vectorized Lee and median speckle filters
│   ├── terrain.py        # Range-Doppler terrain correction adapter
│   └── normalization.py  # SAR dB/linear percentile normalization
├── registration/
│   ├── reprojection.py   # Rasterio geospatial warping to reference grid
│   ├── alignment.py      # Gradient-based phase correlation fine alignment
│   └── validation.py     # Overlap, NMI, and edge structure validation
├── fusion/
│   ├── early_fusion.py   # Dynamic channel stacking (e.g. 6-channel raster)
│   ├── feature_fusion.py # PyTorch two-stream CNN architecture
│   └── model.py          # Unified OpticalSARModel inference wrapper
└── confidence/
    └── confidence.py     # Multi-criteria confidence scoring engine
```

---

## Quickstart

```python
from pathlib import Path
from modules.optical_sar import run_optical_sar_pipeline, OpticalSARConfig

# 1. Initialize configuration (defaults: percentile normalization, Lee filter, early fusion)
config = OpticalSARConfig()

# 2. Run multimodal analysis
result = run_optical_sar_pipeline(
    optical_path="data/optical/sentinel2_sample.tif",
    sar_path="data/sar/sentinel1_sample.tif",
    cloud_mask_path="data/optical/scl_sample.tif",  # Optional
    dem_path="data/dem/copernicus_30m.tif",          # Optional
    config=config,
    run_inference=True,
)

# 3. Inspect structured output
print("Status:", result.status)
print("Registration Score:", result.registration["registration_score"])
print("Registration Passed:", result.registration["passed"])
print("Confidence:", result.confidence["score"], f"({result.confidence['level']})")
print("Fused Channels:", result.fusion["channels"])

# 4. JSON-serializable dictionary for M4 Agent
m4_payload = result.to_dict()

# 5. Geospatial evidence dictionary for M5 GIS Module
m5_evidence = result.to_gis_evidence()
```

---

## Modality Assumptions & Scientific Rules

1. **Cloud Masking**:
   Optical cloud detection cannot be reliably performed on RGB alone. If no SCL or cloud mask is provided, `cloud_mask_status` is explicitly set to `"unavailable"`.
2. **SAR Calibration**:
   Converts linear backscatter ($\sigma^0$) to decibel (dB) via $10 \log_{10}(\max(\text{linear}, 10^{-6}))$. If input rasters are already calibrated, set `is_already_calibrated=True` (default).
3. **Terrain Correction**:
   Full SAR Range-Doppler Terrain Correction requires external orbit vectors and DEM processing (e.g. ESA SNAP GPT). If no DEM is provided, terrain correction explicitly reports `applied=False` with status `"dem_not_provided"`.
4. **Registration Validation**:
   Validation does not rely on naive pixel correlation. It evaluates valid overlap ratio (IoU), Normalized Mutual Information (NMI), and structural gradient consistency.
5. **Confidence Estimation**:
   Engineering baseline weighting data validity, alignment quality, and neural margins:
   $$\text{score} = 0.25 Q_{\text{opt}} + 0.20 Q_{\text{sar}} + 0.20 Q_{\text{reg}} + 0.35 Q_{\text{model}}$$
   Missing modalities dynamically trigger penalty factors and explicit notification.

---

## Inter-Module Interfaces

- **M2 (Change Detection & Grounding)**: Consumes `result.optical_data`, `result.registered_sar_data`, and `result.early_fusion_result.fused_data`.
- **M4 (Agent Orchestrator)**: Consumes `result.to_dict()`.
- **M5 (GIS & Evidence)**: Consumes `result.to_gis_evidence()` containing CRS, bounds, transform, and spatial resolution.
- **M6 (Frontend & API)**: Consumes serializable dictionary or export GeoTIFFs.
