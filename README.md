# SatQuery-AI — Geospatial AI Pipeline

**SatQuery-AI** is a modular satellite-imagery analysis pipeline that chains multimodal AI detection with GIS-compliant evidence generation.

| Module | Name | Role |
|--------|------|------|
| **M2** | Change Detection | Pixel-difference baseline detector; outputs regions, masks, and georeferenced bounding boxes |
| **M3** | Optical + SAR Multimodal Analysis (`modules/optical_sar`) | Sentinel-1/2 preprocessing, cross-modal registration, early/feature fusion, confidence scoring |
| **M4** | Agentic Specialist Wrapper | Orchestrates M2/M3 specialist results into a structured `SpecialistResult` envelope |
| **M5** | GIS & Evidence Generation (`geospatial/`) | Bridges AI outputs with real-world spatial references; produces GeoJSON, geodesic area measurements, interactive satellite maps, and evidence artifacts |

---

## Repository Layout

```text
SatQuery-AI/
├── geospatial/                     # M5 — GIS & Evidence Generation engine
│   ├── __init__.py                 # Package exports
│   ├── metadata.py                 # GeoTIFF metadata extractor (rasterio)
│   ├── coordinates.py              # Pixel <-> Geo coordinates & bbox transformations
│   ├── polygons.py                 # Change mask vectorization → EPSG:4326 polygons
│   ├── area.py                     # Geodesic area calculations (m², ha, km²)
│   ├── visualization.py            # Interactive Folium satellite map generator
│   ├── evidence.py                 # GeoJSON & evidence.json generator
│   ├── schema.py                   # Typed dataclasses (M2M3Payload, M2ChangeDetectionResult,
│   │                               #   M4SpecialistResult, EvidenceOutput, M2Region)
│   ├── integration.py              # M2 / M3 / M4 integration adapters
│   │                               #   (process_m2_result, process_m3_result, process_m4_result,
│   │                               #    process_m2_m3_result — legacy alias)
│   ├── m3_adapter.py               # M3 multimodal payload → GIS layer exporter
│   │                               #   (GeoTIFF export for Optical RGB, SAR VV/VH, NDVI, 6-ch stack)
│   └── pipeline.py                 # Core pipeline orchestrator (run_geospatial_pipeline)
├── modules/
│   └── optical_sar/                # M3 — Optical + SAR Multimodal Analysis
│       ├── __init__.py             # Unified public API
│       ├── config.py               # OpticalSARConfig dataclass
│       ├── pipeline.py             # End-to-end multimodal pipeline orchestrator
│       ├── README.md               # M3 module documentation
│       ├── optical/                # Sentinel-2 loader, cloud mask, normalisation, NDVI/NDWI
│       ├── sar/                    # Sentinel-1 loader, calibration, Lee speckle, terrain correction
│       ├── registration/           # Reprojection, phase-correlation alignment, NMI validation
│       ├── fusion/                 # Early fusion (6-ch stack), two-stream CNN feature fusion
│       ├── confidence/             # Multi-criteria confidence scoring engine
│       └── weights/                # Pretrained model weight files
├── tests/
│   ├── __init__.py
│   ├── test_m5.py                  # M5 geospatial pipeline unit & integration tests
│   ├── test_optical_sar.py         # M3 optical+SAR module tests
│   └── run_m3_integration.py       # M3 → M5 end-to-end integration smoke test
├── data/
│   └── mock/
│       ├── generate_mock.py        # Generator for synthetic GeoTIFF & change mask
│       ├── sample.tif              # Synthetic 256×256 GeoTIFF (UTM Zone 43N / EPSG:32643)
│       ├── change_mask.npy         # Synthetic 256×256 binary change mask
│       ├── m2_result.json          # Generic upstream M2 detection payload
│       ├── m2_georef_output.json   # Georeferenced M2 output with CRS & transform
│       ├── m2_non_georef_output.json # Non-georeferenced M2 output (pixel-only)
│       ├── m2_non_georeferenced.json # Non-georeferenced M2 alternative format
│       ├── m2_real_result.json     # Realistic M2 output with full region objects
│       ├── m3_pipeline_output.json # M3 Optical+SAR pipeline result payload
│       ├── m4_m2_result.json       # M4 SpecialistResult wrapping M2 evidence
│       └── m4_specialist_result.json # M4 SpecialistResult with mission_result wrapper
├── output/                         # Default artifact destination (git-ignored)
│   ├── evidence.json
│   ├── evidence.geojson
│   └── map.html
├── run_m5.py                       # CLI entry point (supports M2 / M3 / M4 payloads)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Run All Tests
```bash
# M5 geospatial tests
python -m pytest tests/test_m5.py -v

# M3 optical+SAR tests
python -m pytest tests/test_optical_sar.py -v

# All tests
python -m pytest tests/ -v
```

### 2. Run M5 Pipeline with Default Mock Data
```bash
python run_m5.py
```

### 3. Run with an Upstream M2 Payload
```bash
python run_m5.py --m2-result data/mock/m2_result.json
# Georeferenced variant (contains CRS + affine transform)
python run_m5.py --m2-result data/mock/m2_georef_output.json
```

### 4. Run with an Upstream M3 Multimodal Payload
```bash
python run_m5.py --m3-result data/mock/m3_pipeline_output.json
```

### 5. Run with an Upstream M4 SpecialistResult Payload
```bash
python run_m5.py --m4-result data/mock/m4_m2_result.json
python run_m5.py --m4-result data/mock/m4_specialist_result.json
```

### 6. Run with Custom GeoTIFF & Mask
```bash
python run_m5.py --geotiff path/to/image.tif --mask path/to/mask.npy \
                 --target "deforestation" --confidence 0.94 --output output/
```

### 7. View the Interactive Satellite Map
```bash
# Windows
start output/map.html

# macOS / Linux
open output/map.html          # macOS
xdg-open output/map.html      # Linux
```

---

## CLI Reference (`run_m5.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--geotiff` | `data/mock/sample.tif` | Path to reference GeoTIFF |
| `--mask` | `data/mock/change_mask.npy` | Path to binary change mask (`.npy` or raster) |
| `--m2-result` | — | Upstream M2 detection JSON (overrides `--geotiff` / `--mask`) |
| `--m3-result` | — | Upstream M3 multimodal (Optical + SAR) analysis JSON |
| `--m4-result` | — | Upstream M4 `SpecialistResult` JSON wrapper |
| `--target` | `deforestation` | Target classification name |
| `--confidence` | `0.94` | Detection confidence score (0.0 – 1.0) |
| `--output` | `output` | Output directory for evidence artifacts |

Priority when multiple flags are supplied: `--m4-result` > `--m2-result` > `--m3-result` > standalone mode.

---

## Module 3 (M3): Optical + SAR Multimodal Analysis

See [`modules/optical_sar/README.md`](modules/optical_sar/README.md) for the full M3 documentation.

### Quick Example
```python
from modules.optical_sar import run_optical_sar_pipeline, OpticalSARConfig

config = OpticalSARConfig()
result = run_optical_sar_pipeline(
    optical_path="data/optical/sentinel2_sample.tif",
    sar_path="data/sar/sentinel1_sample.tif",
    cloud_mask_path="data/optical/scl_sample.tif",   # Optional
    dem_path="data/dem/copernicus_30m.tif",           # Optional
    config=config,
    run_inference=True,
)

print("Status          :", result.status)
print("Reg. Score      :", result.registration["registration_score"])
print("Confidence      :", result.confidence["score"], f"({result.confidence['level']})")

# Hand off to M4 or M5
m4_payload   = result.to_dict()
m5_evidence  = result.to_gis_evidence()
```

### Confidence Formula
$$\text{score} = 0.25\,Q_\text{opt} + 0.20\,Q_\text{sar} + 0.20\,Q_\text{reg} + 0.35\,Q_\text{model}$$

---

## Module 5 (M5): GIS & Evidence Generation

### Key Capabilities
- **Dynamic GeoTIFF Metadata** — CRS, bounds, resolution, and affine transform via `rasterio`.
- **Pixel-to-Geographic Transformation** — Projects pixel coordinates to `EPSG:4326`.
- **Mask Vectorization** — Converts binary change masks into `shapely.Polygon` geometries.
- **Geodesic Surface Area** — Exact ellipsoidal areas ($m^2$, ha, $km^2$) via `pyproj.Geod(ellps="WGS84")`.
- **Interactive Satellite Maps** — Folium maps with Esri World Imagery + OSM tiles, layer toggles, and popups.
- **Standards-Compliant Evidence** — `evidence.json` + `evidence.geojson` (RFC 7946 / CRS84).
- **Multi-Module Integration** — Accepts M2, M3, and M4 payloads via typed adapters.

### Integration API

#### Option A — Standard dictionary
```python
from geospatial import process_m2_m3_result   # legacy alias → process_m2_result

payload = {
    "target": "new building",
    "confidence": 0.91,
    "change_detected": True,
    "reference_image": "path/to/reference.tif",
    "change_mask": "path/to/change_mask.npy",
    "bounding_boxes": [[50, 60, 100, 110]],
}
evidence = process_m2_m3_result(payload, output_dir="output")

print(f"Detected Area: {evidence['area']['total_hectares']:.2f} ha")
print(f"Evidence JSON: {evidence['evidence_path']}")
print(f"GeoJSON      : {evidence['geojson_path']}")
print(f"Map          : {evidence['map_path']}")
```

#### Option B — Typed dataclass (`M2M3Payload`)
```python
from geospatial import process_m2_m3_result, M2M3Payload

payload = M2M3Payload(
    target="unauthorized clearing",
    confidence=0.95,
    change_detected=True,
    reference_image="path/to/reference.tif",
    change_mask="path/to/change_mask.npy",
    bounding_boxes=[[40, 50, 90, 100]],
)
evidence = process_m2_m3_result(payload)
```

#### Option C — JSON file path
```python
from geospatial import process_m2_m3_result

evidence = process_m2_m3_result("data/mock/m2_result.json", output_dir="output")
```

#### Option D — M3 multimodal payload
```python
from geospatial.integration import process_m3_result

evidence = process_m3_result("data/mock/m3_pipeline_output.json", output_dir="output")
```

#### Option E — M4 SpecialistResult wrapper
```python
from geospatial.integration import process_m4_result

evidence = process_m4_result("data/mock/m4_m2_result.json", output_dir="output")
```

### Core API Reference

#### `run_geospatial_pipeline()`
```python
from geospatial.pipeline import run_geospatial_pipeline

evidence = run_geospatial_pipeline(
    reference_geotiff="data/mock/sample.tif",
    change_mask="data/mock/change_mask.npy",
    bounding_boxes=[[40, 50, 90, 100], [140, 160, 200, 185]],
    target="deforestation",
    confidence=0.94,
    output_dir="output",
)
```

#### Standalone Geospatial Utilities
```python
from geospatial import (
    read_geotiff_metadata,
    pixel_to_geo,
    pixel_bbox_to_geo_bbox,
    mask_to_polygons,
    calculate_polygon_area,
    generate_folium_map,
    generate_geojson,
    generate_evidence_json,
)

meta      = read_geotiff_metadata("sample.tif")
lon, lat  = pixel_to_geo(col=128, row=128, transform=meta["transform"], crs=meta["crs"])
bbox_geo  = pixel_bbox_to_geo_bbox([40, 50, 90, 100], meta["transform"], meta["crs"])
polygons  = mask_to_polygons("change_mask.npy", meta["transform"], meta["crs"])
area_info = calculate_polygon_area(polygons[0])
print(f"Area: {area_info['area_sq_meters']} m² | {area_info['area_hectares']} ha")
```

#### `m3_adapter` — GeoTIFF Layer Export
```python
from geospatial.m3_adapter import ingest_m3_output

layers = ingest_m3_output(
    m3_payload="data/mock/m3_pipeline_output.json",
    output_dir="output/layers",
)
# layers → dict of exported GeoTIFF paths:
#   optical_rgb.tif, sar_vvvh.tif, ndvi.tif, fused_6ch.tif, ...
```

---

## Data Schemas

| Dataclass | Module | Description |
|-----------|--------|-------------|
| `M2M3Payload` | `geospatial.schema` | Generic upstream payload (target, confidence, reference_image, change_mask, bboxes) |
| `M2ChangeDetectionResult` | `geospatial.schema` | Full M2 change detection output with region list |
| `M2Region` | `geospatial.schema` | Individual detected region (pixel + geo bounds, centroid, polygon) |
| `M4SpecialistResult` | `geospatial.schema` | M4 wrapper around M2 evidence |
| `EvidenceOutput` | `geospatial.schema` | M5 evidence contract (paths, area, bboxes, polygons) |

---

## Output Artifacts

### `output/evidence.json`
```json
{
  "target": "deforestation",
  "confidence": 0.94,
  "change_detected": true,
  "bounding_boxes": [
    {
      "pixel_bbox": [40.0, 50.0, 90.0, 100.0],
      "geo_bbox": [77.037969, 28.001043, 77.043136, 28.005629],
      "corners_geo": [[77.038054, 28.005629], [77.043136, 28.005554],
                      [77.043051, 28.001043], [77.037969, 28.001118]],
      "centroid": [77.040553, 28.003336]
    }
  ],
  "geographic_coordinates": {
    "center": [27.999419, 77.046049],
    "overall_bounds_4326": [77.037969, 27.993208, 77.054129, 28.005629],
    "polygon_centroids": [[28.003336, 77.040553], [27.994381, 77.051059]]
  },
  "area": {
    "total_sq_meters": 399920.9,
    "total_hectares": 39.9921,
    "total_sq_km": 0.399921,
    "polygon_count": 2
  },
  "evidence_path": "output/evidence.json",
  "geojson_path": "output/evidence.geojson",
  "map_path": "output/map.html",
  "raster_metadata": {
    "crs": "EPSG:32643",
    "resolution": [10.0, 10.0],
    "dimensions": {"width": 256, "height": 256}
  }
}
```

### `output/evidence.geojson`
RFC 7946 `FeatureCollection` with `change_polygon` and `bounding_box` features, each carrying area ($m^2$, ha, $km^2$) and confidence properties.

### `output/map.html`
Standalone interactive Folium map:
- **Esri World Imagery** satellite basemap + OpenStreetMap overlay
- Layer toggles for *Detected Regions (Polygons)* and *Bounding Boxes*
- Hover tooltips and popups with target, confidence, area, and centroid
- Auto-fitted bounds

---

## Inter-Module Data Flow

```text
Sentinel-1/2 GeoTIFFs
        │
        ▼
  [M3: optical_sar]  ─────────────────────────────────────────┐
  run_optical_sar_pipeline()                                   │
        │                                                      │
        │ result.to_dict()                                     │
        ▼                                                      │
  [M4: SpecialistResult]  ◄────────────────────────────────── ┘
  M4SpecialistResult envelope
        │
        │  process_m4_result()  ──── or ──── process_m3_result()
        │                                    process_m2_result()
        ▼
  [M5: geospatial pipeline]
  run_geospatial_pipeline()
        │
        ▼
┌───────────────────────────┐
│  output/evidence.json     │
│  output/evidence.geojson  │
│  output/map.html          │
└───────────────────────────┘
```

---

## License

Part of the SatQuery-AI Project.
