# SatQuery AI 🛰️

> **An agentic satellite-query orchestration platform for converting natural-language geospatial questions into structured, executable analysis workflows.**

SatQuery AI converts natural-language satellite questions into explicit analytical workflows. The system separates query understanding, deterministic planning, specialist execution, geospatial evidence generation, and frontend presentation.

## 🏗️ Architecture

```text
M6 Frontend
    ↓ HTTP / multipart
M5 FastAPI / API Gateway
    ↓
M4 Agentic Controller
    ├── Parser
    ├── Classifier
    ├── Planner
    ├── Executor
    └── Synthesizer
    ↓
┌────────┬────────┬────────────┐
│ M1 VQA │ M2     │ M3         │
│EarthDial│Change │Optical+SAR │
│        │Ground │Fusion       │
└────────┴───┬────┴────────────┘
             ↓
        M5 GIS Core
             ↓
       Structured Response
             ↓
          M6 UI
```

The orchestration layer is contract-first: specialist implementations are isolated behind adapters and return structured `ToolResult` objects. M2/M3 native results are preserved for downstream evidence and GIS processing.

## 🎯 Supported Workflows

| Workflow | Route |
|---|---|
| Image understanding | M6 → M5 → M4 → M1 → M4 → M6 |
| Change + location | M6 → M5 → M4 → M2 Change → M2 Grounding → M5 GIS → M4 → M6 |
| Optical + SAR | M6 → M5 → M4 → M3 → optional M5 GIS → M4 → M6 |
| Direct GIS evidence | M4 → M5 GIS |

## 🧩 Modules

### M1 — EarthDial / VQA
- Earth-observation visual question answering boundary
- preprocessing and image handling
- direct/remote inference adapters
- explicit model-health and fallback semantics
- heavyweight production inference is environment-dependent

### M2 — Change Detection + Grounding
- raster validation and preprocessing
- true geospatial registration/reprojection
- deterministic absolute-difference baseline
- RCD/model adapter boundary
- region extraction and grounding
- geospatial coordinates and area calculations
- honest fallback semantics when trained weights are unavailable

### M3 — Optical + SAR Fusion
- optical raster loading, cloud masking and normalization
- SAR calibration, dB conversion, normalization and Lee speckle filtering
- DEM-aware terrain-correction adapter with explicit limitations
- CRS/grid reprojection and fine alignment
- optical feature extraction and SAR cross-ratio features
- early channel-level fusion
- feature-fusion model architecture
- trained-weight-only scientific inference
- multimodal confidence assessment

M3 deliberately does **not** treat an untrained neural network as a scientific predictor. Without trained weights, the pipeline returns a partial result and clearly reports that model inference is unavailable. fileciteturn219file0L1-L2

### M4 — Agentic Controller
- deterministic query parser/classifier
- execution planner
- registry and adapters
- dependency-aware executor
- response synthesis
- evidence/confidence propagation
- API and acceptance contracts

### M5 — GIS / Geospatial Evidence
- CRS-aware raster processing
- pixel-to-geographic localization
- polygons and bounding boxes
- area calculation
- GeoJSON/evidence artifacts
- interactive map generation
- M4 adapter integration

### M6 — Frontend / Judge Dashboard
- React/Vite dashboard
- image and temporal-pair upload
- Optical + SAR upload workflow
- VQA/change/GIS/evidence views
- history/report/team architecture views
- live FastAPI contract

## 🔌 M4 Specialist Contract

```text
M4 Executor
   │
   ├── M1 Adapter → EarthDial
   ├── M2 Adapter → Change Detection
   ├── M2 Adapter → Grounding
   ├── M3 Adapter → Optical + SAR
   └── M5 Adapter → GIS
```

Every specialist result is normalized into a stable contract containing status, confidence, structured data, evidence, and errors where applicable.

## 🌐 Live API

FastAPI exposes:

- `GET /health`
- `GET /api/health` — compatibility alias
- `POST /api/v1/mission` — JSON mission endpoint
- `POST /api/analyze` — M6 multipart analysis endpoint
- `POST /api/v1/change-detection` — temporal-pair upload endpoint
- `/outputs/*` — generated artifact serving

The live multipart API accepts:

- `image` for image understanding
- `before_image` + `after_image` for change detection
- `optical_image` + `sar_image` for Optical + SAR

CORS origins are configurable through `SATQUERY_CORS_ORIGINS`.

## 🧪 Testing

Run the complete Python suite:

```bash
pytest -q
```

Frontend verification:

```bash
cd M6/frontend
npm install
npm run lint
npm run build
```

The CI integration workflow verifies the Python integration suite and M6 frontend lint/build. At the time of this update, the newest workflow run is still executing, so its final result must be checked in GitHub Actions rather than assumed. fileciteturn243file0L1-L2

## ⚙️ Local Setup

```bash
git clone https://github.com/Vishal10052006/SatQuery-AI.git
cd SatQuery-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Run the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Open the judge workspace at `http://localhost:8000/` or `http://localhost:8000/dashboard`.

## 📊 Current Engineering Status

| Module | Status | Assessment |
|---|---|---|
| M1 | 🟢 Integrated | ~90% |
| M2 | 🟢 Integrated | ~95% |
| M3 | 🟢 Integrated | ~95% after hardening |
| M4 | 🟢 Core complete | ~95% |
| M5 | 🟢 Integrated | ~90% |
| M6 | 🟢 Demo-ready | ~85% |

### What is proven

Repository tests and integration contracts establish the software wiring and deterministic behavior. M6's latest workflow has already completed frontend lint successfully and was progressing through the frontend build while the Python job was installing its runtime dependencies. fileciteturn243file0L1-L2

### What is not yet proven

Passing tests do not establish scientific accuracy on final SIH satellite datasets, availability of heavyweight production checkpoints, GPU performance, or production-load behavior. Those require validation in the intended runtime environment.

## 👥 SIH Team Work Split

| Member | Primary responsibility |
|---|---|
| M1 | EarthDial / VQA specialist |
| M2 | Change detection + grounding |
| M3 | Optical + SAR fusion |
| M4 | Agentic controller / query intelligence |
| M5 | GIS + backend/API |
| M6 | Frontend + judge-facing integration |

The architecture/progress document contains the detailed module assessment and judge-facing workflows.

## 🗺️ Judge Demo Flow

### 1. VQA
Upload an Earth-observation image → ask a natural-language question → M4 routes to M1 → answer and evidence appear in M6.

### 2. Change detection
Upload BEFORE + AFTER → M4 routes to M2 → detected regions are grounded → M5 produces geospatial evidence → M6 renders the result.

### 3. Optical + SAR
Upload optical + SAR → M4 routes to M3 → data are independently preprocessed and registered → multimodal fusion/inference is attempted only when trained weights exist → confidence and evidence are returned to M6.

## 🔐 Honesty / Scientific Safety

SatQuery distinguishes **software capability**, **architecture baselines**, **fallback algorithms**, and **trained-model inference**. A missing checkpoint is never silently represented as a successful scientific prediction. Likewise, geospatial transformations are performed only when the required metadata are available; otherwise the system reports the limitation.

## 📜 License

License information will be added as the project moves toward public release.

**SatQuery AI 🛰️ — Ask questions about satellite data. Let the system plan the analysis.**
