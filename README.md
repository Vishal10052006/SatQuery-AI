# 🛰️ SatQuery AI

> **Query-driven, agentic Earth-observation intelligence for satellite imagery.**
>
> **Smart India Hackathon (SIH) project — working M1–M4 foundation + executable bi-temporal demo baseline**

SatQuery AI is an intelligent operating layer for **multi-modal Earth-observation analysis**. Instead of forcing a user to choose a model, sensor, or processing pipeline, the system accepts a natural-language mission request and uses an agentic controller to understand the request, select the required capabilities, execute the analysis, and return evidence-backed results.

The long-term system combines **Remote-Sensing VLM/VQA, natural-language change detection (RCD), bi-temporal analysis, text-to-region grounding, Optical + SAR corroboration, deterministic geospatial measurement, evidence, confidence, and GIS visualization**.

---

## 🎯 Problem We Are Solving

Satellite imagery contains enormous amounts of information, but extracting a useful answer normally requires a user to understand:

- which satellite/sensor to use,
- which dates to compare,
- which AI model performs the task,
- how to localize the result geographically,
- and how to validate that an AI-generated answer is trustworthy.

SatQuery AI changes that interaction from **model-driven** to **query-driven**.

### Example mission

> **“Find newly constructed buildings between January 2024 and January 2026 and show where the changes occurred.”**

The intended pipeline is:

```text
Natural-language mission
        ↓
Query Understanding
(target + location + time + modality + task)
        ↓
M4 Agentic Controller
        ↓
Mission Plan
   ┌────┼───────────┐
   ↓    ↓           ↓
  RCD  Grounding  Optical + SAR
   ↓    ↓           ↓
Bi-temporal change analysis
        ↓
Change localization
        ↓
Geospatial measurement
        ↓
Evidence + confidence
        ↓
Human-readable explanation
        ↓
GIS / Mission Workspace
```

---

## 💡 What Makes SatQuery AI Different?

### 1. Query-first interaction

The user describes **what they want to know**, not which model to run.

### 2. Agentic model/tool selection

M4 acts as the orchestration layer. It can determine whether the mission requires VQA, change detection, grounding, Optical + SAR analysis, or a combination of these capabilities.

### 3. Natural-language change detection

The target is expressed semantically — for example:

> “Show newly built structures.”

rather than requiring the user to manually inspect an entire image pair.

### 4. Bi-temporal reasoning

The system treats satellite observations as a **time-aware pair**, enabling before/after analysis rather than single-image interpretation only.

### 5. No fabricated geometry

AI perception and geospatial measurement are separated. A model may propose a changed region, but geographic coordinates and area calculations should be derived from deterministic image/GIS transformations rather than invented by an LLM.

### 6. Multi-modal corroboration

Optical imagery and SAR provide complementary evidence. The architecture allows both modalities to participate in the same mission.

### 7. Evidence-backed output

The final system is designed to expose not only an answer, but also **what evidence produced the answer, which tools ran, and how confident the system is**.

---

# 🧠 System Architecture

```text
                           ┌──────────────────────┐
                           │       USER QUERY      │
                           │ Natural-language task │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  QUERY UNDERSTANDING │
                           │ target/location/time │
                           │ intent + modalities  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │        M4 AGENTIC CONTROLLER      │
                    │ route → plan → execute → trace   │
                    └───────────────┬──────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │ M1 VLM/VQA │       │ M2 RCD /    │       │ M3 Optical  │
       │ scene       │       │ Grounding   │       │ + SAR       │
       │ understanding│      │ change      │       │ corroboration│
       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  EVIDENCE / CONF.    │
                         │ provenance + trace   │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ GEOSPATIAL / GIS     │
                         │ localization + area  │
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ FINAL MISSION RESULT │
                         └──────────────────────┘
```

---

# 🧩 M1–M4 Responsibilities

| Module | Responsibility | Current state |
|---|---|---|
| **M1** | Remote-sensing VLM/VQA and scene understanding | Adapter contract + explicit fallback; neural checkpoint integration is next |
| **M2** | RCD, bi-temporal change detection, grounding | **Working deterministic image-difference baseline** + model adapter contract |
| **M3** | Optical + SAR analysis | Input validation + fusion adapter contract; neural fusion is next |
| **M4** | Query understanding, routing, planning, execution, evidence, confidence, trace | **Integrated orchestration foundation** |

> **Important:** the repository does not falsely claim that large neural checkpoints are already running. The executable demo currently uses a deterministic change-detection baseline so the end-to-end workflow can be tested without downloading multi-GB model weights.

---

# ⚙️ Current Working Demo

The repository now includes a real, dependency-light bi-temporal baseline.

### Input

Two images:

```text
before.png
after.png
```

### Processing

```text
Before image ──┐
               ├─ grayscale normalization
After image ───┘
                      ↓
               absolute difference
                      ↓
                  threshold
                      ↓
              binary change mask
                      ↓
          connected-component analysis
                      ↓
             changed regions / boxes
```

### Output

The baseline returns:

- image dimensions,
- changed-pixel count,
- changed-pixel fraction,
- mean image difference,
- changed-region bounding boxes,
- pixel counts per region,
- processing status,
- detector identity.

This gives the project an **actual executable change-analysis path**, rather than a placeholder response.

Implementation: `models/change/baseline.py`.

---

# 🗂️ Repository Structure

```text
SIH-SatQuery/
│
├── agent/                         # M4 agentic controller
│   ├── adapters.py                # specialist adapter contracts
│   ├── confidence.py              # confidence scoring
│   ├── evidence.py                # evidence construction
│   ├── executor.py                # tool execution
│   ├── planner.py                 # mission planning
│   ├── pipeline.py                # orchestration pipeline
│   ├── router.py                  # deterministic intent router
│   ├── schemas.py                 # routing/tool schemas
│   ├── tool_registry.py           # registered capabilities
│   └── trace.py                   # execution trace
│
├── api/
│   └── main.py                    # FastAPI service
│
├── core/
│   └── contracts.py               # shared specialist result contract
│
├── geospatial/
│   └── geometry.py                # dependency-light geometry utilities
│
├── mission/
│   └── orchestrator.py             # end-to-end M1–M4 mission runner
│
├── models/
│   ├── vlm/
│   │   └── adapter.py             # VLM/VQA adapter contract
│   ├── change/
│   │   ├── adapter.py             # RCD/grounding adapter contract
│   │   └── baseline.py             # executable bi-temporal baseline
│   └── multimodal/
│       └── adapter.py             # Optical + SAR adapter contract
│
├── query/
│   └── parser.py                  # natural-language mission parser
│
├── tests/
│   └── test_full_mission.py       # mission smoke tests
│
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/Vishal10052006/SIH-SatQuery.git
cd SIH-SatQuery
git checkout m1-m4-complete
```

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the tests

```bash
pytest -q
```

## 5. Start the API

```bash
uvicorn api.main:app --reload
```

The service exposes:

```text
GET  /health
POST /api/v1/mission
POST /api/v1/change-detection
```

Open the FastAPI interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

# 🔬 Run the Change-Detection Baseline Directly

The baseline can be called from Python without any neural model weights:

```python
from models.change.baseline import detect_changes

result = detect_changes(
    "before.png",
    "after.png",
    threshold=0.15,
    min_pixels=8,
)

print(result)
```

For a controlled test, create two images where a rectangular region changes between the two dates. The detector should report that changed region as a connected component.

---

# 🤖 Run M4 Mission Orchestration

```python
from mission.orchestrator import run_mission

result = run_mission(
    "Find changes between these two satellite images"
)

print(result["query_spec"])
print(result["plan"])
print(result["results"])
print(result["confidence"])
```

The mission runner follows:

```text
query
  ↓
parse_query()
  ↓
route_query()
  ↓
build_plan()
  ↓
execute specialists
  ↓
evidence + confidence
  ↓
mission result
```

---

# 🌐 API Example

### Request

```http
POST /api/v1/mission
Content-Type: application/json
```

```json
{
  "query": "Find changes between optical and SAR images from 2024 to 2026",
  "context": {}
}
```

### Response shape

```json
{
  "query": "Find changes between optical and SAR images from 2024 to 2026",
  "query_spec": {
    "intent": "multimodal_analysis",
    "target": null,
    "location": null,
    "start_date": "2024-01-01",
    "end_date": "2026-01-01",
    "optical": true,
    "sar": true
  },
  "plan": ["optical_sar"],
  "results": [],
  "confidence": 0.0
}
```

> Exact response fields may evolve as the real specialist models and geospatial data pipeline are integrated.

---

# 🧭 Query Understanding

M4 currently extracts structured information from natural-language requests, including:

- **intent** — scene understanding, change analysis, grounding, multimodal analysis
- **target** — buildings, roads, water bodies, vegetation, cropland, construction
- **time range** — years detected in the query
- **optical requirement**
- **SAR requirement**
- **localization requirement**
- **explanation requirement**

Example:

```text
“Find newly constructed buildings in Lucknow between 2024 and 2026 and show where they occurred.”
```

Conceptually becomes:

```json
{
  "intent": "change_analysis",
  "target": "buildings",
  "location": "Lucknow",
  "start_date": "2024-01-01",
  "end_date": "2026-01-01",
  "needs_localization": true
}
```

The parser is intentionally lightweight at this stage. Production deployment will replace coarse entity extraction with stronger geospatial/entity resolution and satellite catalogue integration.

---

# 🗺️ Geospatial Strategy

SatQuery AI follows a strict separation between **AI perception** and **geospatial measurement**.

```text
AI model
  ↓
semantic detection / region proposal
  ↓
deterministic pixel geometry
  ↓
CRS / affine transform
  ↓
map coordinates
  ↓
polygon / area calculation
  ↓
GIS visualization
```

This prevents an LLM from inventing latitude/longitude values.

The current `geospatial/geometry.py` module contains dependency-light primitives for pixel-to-geographic transformation and polygon area calculation. Full GeoTIFF/CRS handling will be added with the geospatial data layer.

---

# 🛰️ Optical + SAR Strategy

Optical and SAR observations provide different physical signals:

| Modality | Useful evidence |
|---|---|
| Optical | visual structure, land cover, RGB/multispectral appearance |
| SAR | radar backscatter, structure, roughness, all-weather/night capability |

The architecture allows a mission to request both and use them as complementary evidence rather than treating them as interchangeable images.

Current implementation validates both inputs and exposes a fusion-ready contract. **Actual learned cross-modal fusion is a next-stage model integration.**

---

# 🧾 Evidence & Confidence

A central design goal is to make the system inspectable.

The M4 layer is structured to retain:

```text
Query
  ↓
Route decision
  ↓
Selected tools
  ↓
Tool results
  ↓
Evidence
  ↓
Confidence
  ↓
Final answer
```

The project should eventually expose evidence such as:

- source image/date,
- model used,
- detected region,
- pixel/ground geometry,
- modality agreement,
- confidence factors,
- execution trace.

Confidence should describe **system evidence quality**, not pretend to be a scientifically calibrated probability until calibration data is available.

---

# 🧪 Testing Philosophy

The project uses deterministic tests wherever possible so that core orchestration and geometry can be verified without requiring large model checkpoints.

Current test coverage includes:

- query routing,
- temporal parsing,
- mission planning,
- executable change detection,
- changed-region detection.

Future model tests will distinguish between:

1. **unit tests** — deterministic logic,
2. **integration tests** — real model/data pipeline,
3. **evaluation tests** — benchmark metrics on labelled datasets,
4. **golden mission tests** — complete end-to-end SIH scenarios.

---

# 🏗️ Development Roadmap

## Phase 1 — Foundation ✅

- [x] M4 router
- [x] tool schemas
- [x] planner
- [x] executor
- [x] evidence/confidence contracts
- [x] query parser
- [x] FastAPI foundation

## Phase 2 — Working Change Pipeline 🚧

- [x] executable bi-temporal image baseline
- [x] change mask generation
- [x] connected changed-region extraction
- [x] pixel bounding boxes
- [ ] GeoTIFF ingestion
- [ ] CRS-aware polygonization
- [ ] ground-coordinate area calculation
- [ ] real RCD model adapter

## Phase 3 — Earth Observation Intelligence 🚧

- [ ] real RS-VLM/VQA checkpoint
- [ ] text-to-region grounding model
- [ ] Sentinel-1 SAR ingestion
- [ ] Sentinel-2 optical ingestion
- [ ] learned Optical + SAR fusion
- [ ] stronger temporal alignment

## Phase 4 — Agentic Mission System 🚧

- [x] query → route → plan → execute foundation
- [x] execution trace foundation
- [x] evidence/confidence foundation
- [ ] dynamic multi-tool planning
- [ ] mission state management
- [ ] specialist failure recovery
- [ ] evidence fusion across models
- [ ] final answer synthesis

## Phase 5 — Judge Showcase 🚧

- [ ] Mission Workspace UI
- [ ] before/after/change visualization
- [ ] GIS map layers
- [ ] highlighted changed regions
- [ ] explanation panel
- [ ] evidence panel
- [ ] confidence/provenance panel
- [ ] one-click golden mission
- [ ] benchmark/evaluation report

---

# 🏆 Intended SIH Demonstration

The strongest demo scenario is a **natural-language infrastructure-change mission**.

### Judge enters

> **“Find newly constructed buildings in Lucknow between January 2024 and January 2026, verify the change using Optical + SAR evidence, and show the affected locations.”**

### SatQuery AI should automatically

```text
1. Understand the query
2. Resolve the location
3. Resolve the time window
4. Select before/after imagery
5. Select RCD/change analysis
6. Select grounding/localization
7. Run Optical + SAR corroboration
8. Extract changed regions
9. Convert regions to map coordinates
10. Calculate geographic measurements
11. Build evidence
12. Estimate confidence
13. Explain the detected changes
14. Display everything in the Mission Workspace
```

The key judge-facing message is:

> **“The user asks a question. SatQuery AI decides how to answer it.”**

---

# 🔐 Design Principles

### No hallucinated geometry

Coordinates must originate from image/GIS transformations, not generated text.

### Transparent orchestration

The agent should be able to explain which capability it selected and why.

### Evidence before confidence

Confidence is meaningful only when the underlying evidence is visible.

### Deterministic fallbacks

The project remains runnable even when heavyweight neural checkpoints are unavailable.

### Model-agnostic interfaces

Specialist models can be replaced without rewriting the entire orchestration layer.

### Reproducible missions

Inputs, query, model/tool decisions, and outputs should be traceable.

---

# 📌 Current Implementation Status

| Capability | Status |
|---|---|
| Natural-language query parsing | 🟢 Working foundation |
| M4 routing | 🟢 Working |
| Mission planning | 🟢 Working |
| Tool execution framework | 🟢 Working |
| Evidence/confidence contracts | 🟢 Working foundation |
| Bi-temporal image baseline | 🟢 Working |
| Changed-region extraction | 🟢 Working |
| FastAPI service | 🟢 Working foundation |
| Remote-sensing VLM inference | 🟡 Adapter ready; checkpoint integration pending |
| Neural RCD | 🟡 Adapter ready; checkpoint integration pending |
| Grounding model | 🟡 Adapter ready; checkpoint integration pending |
| Optical + SAR neural fusion | 🟡 Adapter ready; model integration pending |
| CRS-aware GIS pipeline | 🟡 Foundation present; full ingestion pending |
| Full Mission Workspace | 🟡 In development |

---

# 👥 M4 Role in the Team

M4 is the **brain of the system**.

```text
M1 → understands the image
M2 → detects/locates change
M3 → provides multi-modal evidence
M4 → decides what to run and combines the results
```

M4 is therefore responsible for:

- query understanding,
- intent classification,
- task decomposition,
- model/tool selection,
- execution ordering,
- failure handling,
- evidence aggregation,
- confidence calculation,
- execution trace,
- final mission synthesis.

---

# 📚 Reference Architecture

The project architecture was informed by the public SatQuery-Ai reference implementation and adapted into this repository rather than copied as a black box.

Reference project:

```text
https://github.com/theninthfoundry/SatQuery-Ai
```

Our implementation keeps the core idea of **agentic orchestration + specialist models + evidence + deterministic geospatial processing**, while building the SIH-specific workflow inside `SIH-SatQuery`.

---

# 📄 Documentation

- `README.md` — project overview, architecture, setup and demo
- `docs/ARCHITECTURE.md` — deeper architecture notes
- `agent/` — M4 orchestration implementation
- `models/` — specialist model contracts and executable baseline
- `query/` — mission/query understanding
- `geospatial/` — deterministic geospatial utilities
- `tests/` — automated verification

---

# ⚠️ Model & Hardware Notes

Remote-sensing VLMs, grounding models, RCD networks, and Optical + SAR fusion models can require substantial GPU memory and model checkpoints.

The repository therefore separates:

```text
LIGHTWEIGHT CORE
    ↓
query + routing + planning + baseline processing

OPTIONAL HEAVY MODELS
    ↓
VLM + RCD + grounding + multimodal fusion
```

This allows development and CI to continue without requiring a large GPU at every stage.

Do **not** interpret the deterministic baseline as equivalent to a trained RCD model. Its purpose is reproducible engineering validation and a runnable demo path while the learned models are integrated.

---

# 🤝 Contribution Workflow

Recommended branch strategy:

```text
main
 │
 ├── feature/query-understanding
 ├── feature/change-detection
 ├── feature/geospatial
 ├── feature/vlm
 ├── feature/optical-sar
 └── feature/mission-ui
```

Before merging:

```bash
pytest -q
```

Keep model-specific code behind stable specialist interfaces so M4 does not become tightly coupled to a particular neural architecture.

---

# 📜 License

See the repository license file for the applicable project terms.

---

## 🚀 SatQuery AI

**Ask the Earth-observation system a question. Let the agent decide how to answer it.**
