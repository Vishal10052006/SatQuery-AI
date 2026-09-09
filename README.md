# SatQuery-AI — Module 5 (M5): GIS, Geospatial Processing & Evidence Generation

**M5** is the geospatial engine of the **SatQuery-AI** pipeline. It bridges AI detection outputs (change masks, pixel bounding boxes) with real-world spatial references, producing GIS-compliant **GeoJSON**, ellipsoidal **geodesic area measurements**, interactive **Folium satellite maps**, and comprehensive **evidence artifacts**.

M5 is built to operate **independently** with full mock data support, while exposing a clean integration interface (`process_m2_m3_result`) for upstream AI modules (M2: Change Detection, M3: Target Classification).

---

## Key Features

- **Dynamic GeoTIFF Metadata**: Extracts CRS, native bounds, resolution, and affine transform dynamically via `rasterio` without hardcoding projections.
- **Pixel-to-Geographic Transformation**: Projects pixel coordinates and bounding boxes to standard `EPSG:4326` (WGS84 longitude/latitude).
- **Mask Vectorization**: Converts binary change masks into valid `shapely.geometry.Polygon` geometries reprojected to `EPSG:4326`.
- **Geodesic Surface Area Precision**: Calculates exact ellipsoidal surface areas ($m^2$, hectares, $km^2$) using `pyproj.Geod(ellps="WGS84")`, eliminating latitude-dependent distortion.
- **Interactive Satellite Mapping**: Generates interactive Folium maps combining high-resolution **Esri World Imagery** and OpenStreetMap tiles with hover highlights, popups, and layer toggles.
- **Standardized Evidence Generation**: Generates standards-compliant `evidence.json` and `evidence.geojson` (RFC 7946 / CRS84).
- **M2/M3 Team Integration Interface**: Clean adapter accepting typed dataclass payloads (`M2M3Payload`), Python dictionaries, or JSON files.

---

## Directory Structure

```text
SatQuery-AI/
├── geospatial/
│   ├── __init__.py          # Package exports
│   ├── metadata.py          # GeoTIFF metadata extractor (rasterio)
│   ├── coordinates.py       # Pixel <-> Geo coordinates & bbox transformations
│   ├── polygons.py          # Change mask vectorization to EPSG:4326 polygons
│   ├── area.py              # Geodesic area calculations (m², ha, km²)
│   ├── visualization.py     # Interactive Folium satellite map generator
│   ├── evidence.py          # GeoJSON & evidence.json generator
│   ├── schema.py            # Typed dataclasses (M2M3Payload, EvidenceOutput)
│   ├── integration.py       # Upstream AI integration adapter (process_m2_m3_result)
│   └── pipeline.py          # Core pipeline orchestrator (run_geospatial_pipeline)
├── tests/
│   ├── __init__.py
│   └── test_m5.py           # 13 automated unit and integration tests
├── data/
│   └── mock/
│       ├── generate_mock.py # Generator for sample GeoTIFF & change mask
│       ├── sample.tif       # Synthetic 256x256 GeoTIFF with UTM Zone 43N CRS
│       ├── change_mask.npy  # Synthetic 256x256 binary change mask
│       └── m2_result.json   # Sample upstream M2 AI detection payload
├── output/                  # Default artifact destination (ignored by git)
│   ├── evidence.json
│   ├── evidence.geojson
│   └── map.html
├── run_m5.py                # Standalone CLI execution script
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignores output/ and Python/test caches
└── README.md                # Project documentation
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

## Quick Start & Testing

### 1. Run All Automated Tests (13/13)
```bash
python -m pytest tests/test_m5.py -v
```

### 2. Run the Pipeline (Default Mock Data)
```bash
python run_m5.py
```

### 3. Run Pipeline with Upstream M2/M3 Detection JSON
```bash
python run_m5.py --m2-result data/mock/m2_result.json
```

### 4. View Interactive Satellite Map
```bash
# Windows
start output/map.html

# macOS
open output/map.html

# Linux
xdg-open output/map.html
```

---

## Integration with Upstream Modules (M2 / M3)

Upstream modules pass detection results directly into `process_m2_m3_result`:

```text
M2 / M3 Output Payload
         ↓
geospatial.process_m2_m3_result(payload, output_dir="output")
         ↓
geospatial.run_geospatial_pipeline()
         ↓
┌────────────────────────────────────────────────────────┐
│  • output/evidence.json                                │
│  • output/evidence.geojson                             │
│  • output/map.html                                     │
└────────────────────────────────────────────────────────┘
```

### Option A: Passing a Standard Dictionary
```python
from geospatial import process_m2_m3_result

payload = {
    "target": "new building",
    "confidence": 0.91,
    "change_detected": True,
    "reference_image": "path/to/reference.tif",
    "change_mask": "path/to/change_mask.npy",
    "bounding_boxes": [
        [50, 60, 100, 110]  # [xmin, ymin, xmax, ymax] in pixel space
    ]
}

evidence = process_m2_m3_result(payload, output_dir="output")

print(f"Detected Area: {evidence['area']['total_hectares']:.2f} ha")
print(f"Evidence JSON: {evidence['evidence_path']}")
print(f"GeoJSON      : {evidence['geojson_path']}")
print(f"Map          : {evidence['map_path']}")
```

### Option B: Using the Typed Dataclass (`M2M3Payload`)
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

### Option C: Passing a JSON File Path
```python
from geospatial import process_m2_m3_result

evidence = process_m2_m3_result("data/mock/m2_result.json", output_dir="output")
```

---

## Core API Reference

### `run_geospatial_pipeline()`
```python
from geospatial.pipeline import run_geospatial_pipeline

evidence = run_geospatial_pipeline(
    reference_geotiff="data/mock/sample.tif",
    change_mask="data/mock/change_mask.npy",
    bounding_boxes=[[40, 50, 90, 100], [140, 160, 200, 185]],
    target="deforestation",
    confidence=0.94,
    output_dir="output"
)
```

### Standalone Geospatial Utilities
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

# Read metadata
meta = read_geotiff_metadata("sample.tif")

# Convert single pixel coordinate (col, row) -> (longitude, latitude)
lon, lat = pixel_to_geo(col=128, row=128, transform=meta["transform"], crs=meta["crs"])

# Convert pixel bbox -> geographic bbox & polygon ring
bbox_geo = pixel_bbox_to_geo_bbox([40, 50, 90, 100], meta["transform"], meta["crs"])

# Vectorize change mask into EPSG:4326 polygons
polygons = mask_to_polygons("change_mask.npy", meta["transform"], meta["crs"])

# Calculate exact geodesic area
area_info = calculate_polygon_area(polygons[0])
print(f"Area: {area_info['area_sq_meters']} m² | {area_info['area_hectares']} ha")
```

---

## Output Artifacts Specification

### 1. `output/evidence.json`
Main evidence contract containing all detection metrics, geographic coordinates, and file paths:
```json
{
  "target": "deforestation",
  "confidence": 0.94,
  "change_detected": true,
  "bounding_boxes": [
    {
      "pixel_bbox": [40.0, 50.0, 90.0, 100.0],
      "geo_bbox": [77.037969, 28.001043, 77.043136, 28.005629],
      "corners_geo": [[77.038054, 28.005629], [77.043136, 28.005554], [77.043051, 28.001043], [77.037969, 28.001118]],
      "centroid": [77.040553, 28.003336]
    }
  ],
  "geographic_coordinates": {
    "center": [27.999419, 77.046049],
    "overall_bounds_4326": [77.037969, 27.993208, 77.054129, 28.005629],
    "polygon_centroids": [[28.003336, 77.040553], [27.994381, 77.051059]]
  },
  "polygons": [ ... ],
  "area": {
    "total_sq_meters": 399920.9,
    "total_hectares": 39.9921,
    "total_sq_km": 0.399921,
    "polygon_count": 2,
    "individual_polygons": [ ... ]
  },
  "evidence_path": "output/evidence.json",
  "geojson_path": "output/evidence.geojson",
  "map_path": "output/map.html",
  "raster_metadata": {
    "crs": "EPSG:32643",
    "crs_epsg": 32643,
    "resolution": [10.0, 10.0],
    "dimensions": { "width": 256, "height": 256 },
    "bounds": { "left": 700000.0, "bottom": 3097440.0, "right": 702560.0, "top": 3100000.0 }
  }
}
```

### 2. `output/evidence.geojson`
Standard `FeatureCollection` with:
- `change_polygon` features with area ($m^2$, $ha$, $km^2$) and confidence properties.
- `bounding_box` features with pixel and geographic bounds.

### 3. `output/map.html`
Standalone interactive map featuring:
- **Esri World Imagery** satellite basemap & OpenStreetMap.
- Layer toggles for `Detected Regions (Polygons)` and `Bounding Boxes`.
- Detailed popups and tooltips showing target name, confidence, area in $m^2$/$ha$, and centroid coordinates.
- Automatic bounds fitting.

---

## License

Part of the SatQuery-AI Project.
