# SatQuery AI 🛰️

> **An agentic satellite-query orchestration platform for converting natural-language geospatial questions into structured, executable analysis workflows.**

SatQuery AI is the orchestration layer of a larger satellite intelligence system. It accepts a user's natural-language query, determines the required analytical intent, builds an execution plan, invokes the appropriate specialist modules, and returns a structured response with evidence, confidence, results, and an execution trace.

The current repository implements the **M4 Agentic Orchestration Layer** and its integration contracts for specialist capabilities including VQA, change detection, grounding, Optical + SAR analysis, and GIS processing.

---

## 🎯 Problem

Satellite imagery analysis is typically fragmented across multiple specialist tools and processing pipelines. A user asking a seemingly simple question such as:

> *"What changed in this area between the two dates, and where exactly did the change occur?"*

may require several dependent operations:

1. Understand the user's intent.
2. Identify the relevant satellite-analysis capability.
3. Execute change detection.
4. Ground the detected region spatially.
5. Perform GIS/geospatial processing when required.
6. Combine the outputs into one understandable response.

SatQuery AI provides a **single query-driven orchestration interface** for this workflow.

---

## 🚀 Core Concept

```text
                    Natural-Language Query
                              │
                              ▼
                    ┌───────────────────┐
                    │    Query Parser   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Intent Classifier │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   M4 Planner      │
                    │ Execution Planning│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │     Executor      │
                    └─────────┬─────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
          M1 VQA        M2 Change/Ground   M3 Optical+SAR
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                         M5 GIS Layer
                              │
                              ▼
                    ┌───────────────────┐
                    │    Synthesizer    │
                    │ Answer + Evidence │
                    └─────────┬─────────┘
                              │
                              ▼
                    Structured AgentResponse
```

The architecture deliberately separates **query understanding, planning, execution, specialist integration, and response synthesis** so that individual satellite-analysis modules can evolve independently.

---

## 🧠 Supported Query Intents

The current M4 schema defines the following high-level intents:

| Intent | Purpose | Specialist |
|---|---|---|
| `VQA` | Ask questions about satellite imagery | M1 VQA |
| `CHANGE_DETECTION` | Detect changes between observations | M2 Change Detection |
| `GROUNDING` | Locate/ground objects or regions in imagery | M2 Grounding |
| `OPTICAL_SAR` | Combine or analyze optical and SAR information | M3 Optical + SAR |

The planner can also chain operations when a query requires multiple capabilities. For example, a change-detection query requiring geographic localization can execute **Change Detection → Grounding → GIS**.

---

## 🧩 System Modules

### M4 Agentic Controller

The M4 layer is responsible for coordinating the complete query lifecycle:

- Query parsing
- Intent classification
- Execution planning
- Specialist selection
- Ordered tool execution
- Result collection
- Evidence propagation
- Response synthesis
- Execution tracing
- Failure/partial-result handling

### Specialist Integration Adapters

The repository provides standardized adapter boundaries for:

- **M1 — VQA**
- **M2 — Change Detection**
- **M2 — Grounding**
- **M3 — Optical + SAR**
- **M5 — GIS**

These adapters isolate M4 from specialist implementation details and provide a consistent interface for future model/service integration.

### Query Contract Layer

Pydantic schemas define explicit contracts for:

- `QueryRequest`
- `ParsedQuery`
- `PlanStep`
- `ExecutionPlan`
- `Evidence`
- `ToolResult`
- `AgentResponse`

This creates a stable interface between the orchestration layer and the rest of the application.

---

## 📁 Repository Structure

```text
SatQuery-AI/
│
├── app/
│   ├── adapters/
│   │   ├── common.py
│   │   ├── m1_vqa_adapter.py
│   │   ├── m2_change_adapter.py
│   │   ├── m2_grounding_adapter.py
│   │   ├── m3_optical_sar_adapter.py
│   │   ├── m5_gis_adapter.py
│   │   └── register.py
│   │
│   ├── agents/
│   │   ├── bootstrap.py
│   │   ├── controller.py
│   │   ├── executor.py
│   │   ├── planner.py
│   │   └── synthesizer.py
│   │
│   ├── query/
│   │   ├── classifier.py
│   │   ├── parser.py
│   │   ├── registry.py
│   │   └── schemas.py
│   │
│   ├── api.py
│   ├── __init__.py
│   └── __main__.py
│
├── demo/
│   └── m4_demo.py
│
├── geospatial/                     # M5 — GIS & Evidence Generation engine
│   ├── __init__.py
│   ├── metadata.py
│   ├── coordinates.py
│   ├── polygons.py
│   ├── area.py
│   ├── visualization.py
│   ├── evidence.py
│   ├── schema.py
│   ├── integration.py
│   ├── m3_adapter.py
│   └── pipeline.py
│
├── modules/
│   └── optical_sar/                # M3 — Optical + SAR Multimodal Analysis
│       ├── __init__.py
│       ├── config.py
│       ├── pipeline.py
│       ├── README.md
│       ├── optical/
│       ├── sar/
│       ├── registration/
│       ├── fusion/
│       ├── confidence/
│       └── weights/
│
├── tests/
│   ├── test_adapter_registration.py
│   ├── test_adapters.py
│   ├── test_api.py
│   ├── test_bootstrap.py
│   ├── test_classifier.py
│   ├── test_controller.py
│   ├── test_executor.py
│   ├── test_m4_acceptance.py
│   ├── test_m5.py
│   ├── test_optical_sar.py
│   ├── test_parser.py
│   ├── test_planner.py
│   ├── test_query_matrix.py
│   ├── test_registry.py
│   ├── test_schemas.py
│   ├── test_specialist_contract.py
│   └── test_synthesizer.py
│
├── data/
│   └── mock/
│       ├── generate_mock.py
│       ├── sample.tif
│       ├── change_mask.npy
│       ├── m2_result.json
│       ├── m2_georef_output.json
│       ├── m2_non_georef_output.json
│       ├── m3_pipeline_output.json
│       ├── m4_m2_result.json
│       └── m4_specialist_result.json
│
├── output/
│   ├── evidence.json
│   ├── evidence.geojson
│   └── map.html
│
├── run_m5.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Execution Flow

A typical request follows this pipeline:

```text
User Query
   ↓
QueryRequest
   ↓
Query Parser
   ↓
Intent Classifier
   ↓
ParsedQuery
   ↓
Query Planner
   ↓
ExecutionPlan
   ↓
Executor
   ↓
Specialist Adapter(s)
   ↓
ToolResult(s)
   ↓
Synthesizer
   ↓
AgentResponse
```

For multi-step analysis, the executor passes the appropriate previous results into subsequent steps rather than treating every specialist as an isolated request.

---

## 💻 Public API

SatQuery exposes a small integration surface through `app/api.py`.

```python
from app.api import ask

response = ask(
    "Detect changes in the construction area between the two images."
)

print(response.answer)
print(response.confidence)
print(response.evidence)
```

The public API accepts an optional runtime context for application-specific inputs such as image identifiers, temporal image pairs, Optical/SAR data, and other processing metadata.

```python
from app.api import ask

response = ask(
    "Where did the land-use change occur?",
    context={
        "before": "before_image.tif",
        "after": "after_image.tif",
    },
)
```

The caller receives a structured `AgentResponse` rather than depending directly on specialist implementation details.

---

## 🧪 Testing

The project includes unit, integration, contract, query-matrix, and acceptance coverage across M4 and M5.

Run the complete test suite from the project root:

```bash
pytest -q
```

For focused test runs:

```bash
pytest tests/test_planner.py -q
pytest tests/test_controller.py -q
pytest tests/test_executor.py -q
pytest tests/test_m4_acceptance.py -q
pytest tests/test_m5.py -v
pytest tests/test_optical_sar.py -v
```

---

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vishal10052006/SatQuery-AI.git
cd SatQuery-AI
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the tests

```bash
pytest -q
```

### 5. Run the M4 demonstration

```bash
python demo/m4_demo.py
```

### 6. Run the M5 geospatial pipeline

```bash
# Default mock data
python run_m5.py

# With upstream M2 payload
python run_m5.py --m2-result data/mock/m2_result.json

# With upstream M3 multimodal payload
python run_m5.py --m3-result data/mock/m3_pipeline_output.json

# With upstream M4 SpecialistResult payload
python run_m5.py --m4-result data/mock/m4_m2_result.json
```

---

## 🔌 Specialist Contract Architecture

M4 does not need to know how an individual specialist model is implemented.

Instead, specialist capabilities are exposed through adapters and standardized result contracts:

```text
                    M4 Executor
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      M1 Adapter     M2 Adapter     M3 Adapter
          │              │              │
          ▼              ▼              ▼
      VQA Model      M2 Models      Optical/SAR
                         │
                         ▼
                    M5 GIS Adapter
```

This enables specialist implementations to be replaced or upgraded without rewriting the core orchestration logic.

---

## 📊 Structured Result Model

Every specialist execution is normalized into a `ToolResult` containing:

- **Tool identity**
- **Execution status**
- **Confidence score**
- **Structured data**
- **Evidence references**
- **Error information when applicable**

The final `AgentResponse` additionally exposes:

- Natural-language answer
- Overall confidence
- Detected intent
- Evidence
- Individual tool results
- Execution trace
- Error/partial-result state

This structure is designed to make the backend suitable for a downstream frontend, GIS visualization layer, or higher-level application module.

---

## 🗺️ Roadmap

### Current — M4 Agentic Orchestration & M5 GIS Engine

- [x] Query schemas and contracts
- [x] Query parsing
- [x] Intent classification
- [x] Deterministic execution planning
- [x] Specialist registry
- [x] Specialist adapter contracts
- [x] Multi-step execution
- [x] Response synthesis
- [x] Public query API
- [x] M4 CLI demonstration
- [x] Acceptance and regression tests
- [x] M5 GIS & Evidence Generation engine
- [x] M3 Optical + SAR multimodal pipeline
- [x] M5 integration adapters (M2 / M3 / M4 payloads)

### Next Integration Stage

- [ ] Connect production M1 VQA inference
- [ ] Connect production M2 change-detection inference
- [ ] Connect production M2 grounding/localization
- [ ] Connect production M3 Optical + SAR inference
- [ ] Connect production M5 GIS processing
- [ ] Integrate real satellite datasets
- [ ] Add production authentication and request management
- [ ] Expose the orchestration layer to the frontend
- [ ] Add map-based result visualization
- [ ] Add observability and production deployment

---

## 🏗️ Design Principles

### 1. Separation of Concerns

Query understanding, planning, execution, specialist inference, and response synthesis are separate components.

### 2. Contract-First Integration

Specialist modules communicate with M4 through explicit schemas instead of tightly coupled implementation details.

### 3. Deterministic Orchestration

The current planner produces explicit execution plans from structured query information, making execution behavior testable and reproducible.

### 4. Evidence-Aware Responses

Results are designed to carry evidence such as masks, overlays, bounding boxes, polygons, coordinates, and statistics instead of returning an opaque answer only.

### 5. Extensibility

New specialist capabilities can be introduced through the adapter/registry architecture without redesigning the entire controller.

---

## 🌍 Intended Applications

SatQuery AI is designed for satellite and geospatial intelligence workflows such as:

- Land-use and land-cover analysis
- Infrastructure monitoring
- Construction detection
- Environmental change analysis
- Disaster and damage assessment
- Agricultural monitoring
- Urban expansion analysis
- Optical/SAR comparative analysis
- Geographic localization and mapping

---

## 🔐 Project Status

**Status: Active Development / Working M4 Integration Layer + M5 GIS Engine**

The repository currently represents the orchestration and integration foundation of the SatQuery AI system. Specialist capabilities are exposed through defined contracts so that real inference models and geospatial processing services can be integrated progressively.

> **Important:** The M4 layer is an orchestration system. Its specialist adapters are integration boundaries; production satellite inference depends on the corresponding specialist implementations being connected.

---

## 🗺️ M5 GIS & Evidence Generation

M5 is the geospatial engine of SatQuery-AI. It bridges AI detection outputs (change masks, pixel bounding boxes) with real-world spatial references.

### Key Capabilities

- **Dynamic GeoTIFF Metadata** — CRS, bounds, resolution, and affine transform via `rasterio`.
- **Pixel-to-Geographic Transformation** — Projects pixel coordinates to `EPSG:4326`.
- **Mask Vectorization** — Converts binary change masks into `shapely.Polygon` geometries.
- **Geodesic Surface Area** — Exact ellipsoidal areas ($m^2$, ha, $km^2$) via `pyproj.Geod(ellps="WGS84")`.
- **Interactive Satellite Maps** — Folium maps with Esri World Imagery + OSM, layer toggles, and popups.
- **Standards-Compliant Evidence** — `evidence.json` + `evidence.geojson` (RFC 7946 / CRS84).
- **Multi-Module Integration** — Accepts M2, M3, and M4 payloads via typed adapters.

### M5 CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--geotiff` | `data/mock/sample.tif` | Path to reference GeoTIFF |
| `--mask` | `data/mock/change_mask.npy` | Path to binary change mask (`.npy` or raster) |
| `--m2-result` | — | Upstream M2 detection JSON |
| `--m3-result` | — | Upstream M3 multimodal (Optical + SAR) JSON |
| `--m4-result` | — | Upstream M4 `SpecialistResult` JSON wrapper |
| `--target` | `deforestation` | Target classification name |
| `--confidence` | `0.94` | Detection confidence score (0.0 – 1.0) |
| `--output` | `output` | Output directory for evidence artifacts |

### M5 Integration API

```python
from geospatial.integration import process_m2_result, process_m3_result, process_m4_result

# From an M2 change detection payload
evidence = process_m2_result("data/mock/m2_result.json", output_dir="output")

# From an M3 multimodal pipeline payload
evidence = process_m3_result("data/mock/m3_pipeline_output.json", output_dir="output")

# From an M4 SpecialistResult wrapper
evidence = process_m4_result("data/mock/m4_m2_result.json", output_dir="output")

print(f"Area     : {evidence['area']['total_hectares']:.2f} ha")
print(f"GeoJSON  : {evidence['geojson_path']}")
print(f"Map      : {evidence['map_path']}")
```

---

## 👥 Team

**SatQuery AI** — Smart India Hackathon Project

Built as a modular AI-driven satellite-query platform with a focus on natural-language interaction, multi-agent orchestration, remote-sensing analysis, and geospatial intelligence.

---

## 📜 License

License information will be added as the project moves toward public release.

---

**SatQuery AI 🛰️ — Ask questions about satellite data. Let the system plan the analysis.**

