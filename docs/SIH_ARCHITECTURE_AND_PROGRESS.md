# SatQuery AI — SIH Architecture & Module Progress

## 1. System Architecture

```text
User / M6 Frontend
        |
        v
+-----------------------+
| M6 Presentation Layer  |
| Dashboard / GIS / UX   |
+-----------+-----------+
            |
            | HTTP JSON / multipart
            v
+-----------------------+
| M5 Backend/API Layer   |
| FastAPI / upload /     |
| artifact serving       |
+-----------+-----------+
            |
            v
+-----------------------+
| M4 Agentic Controller  |
| Parser -> Classifier   |
| -> Planner -> Executor |
| -> Synthesizer         |
+-----+---------+--------+
      |         |        \
      v         v         v
   +------+  +------+  +--------+
   |  M1  |  |  M2  |  |   M3   |
   | VQA  |  |Change|  |Optical |
   |Earth |  |+Ground| | + SAR  |
   |Dial   |  | ing  |  |Fusion  |
   +------+  +---+--+  +---+----+
                  |          |
                  +----+-----+
                       v
                +-------------+
                | M5 GIS Core |
                | CRS / raster|
                | polygons /  |
                | GeoJSON/map |
                +------+------+ 
                       |
                       v
                +-------------+
                | M4 Response |
                | evidence +  |
                | confidence +|
                | trace        |
                +------+------+ 
                       |
                       v
                  M6 rendering
```

### Runtime sequence

```text
Natural language query
        -> QueryRequest
        -> QueryParser
        -> IntentClassifier
        -> QueryPlanner
        -> AgentExecutor
        -> Specialist adapter(s)
        -> ToolResult(s)
        -> ResponseSynthesizer
        -> AgentResponse
        -> FastAPI / M6 frontend
```

The planner currently supports these specialist routes:

- VQA: M4 -> M1
- Change detection: M4 -> M2
- Change + localization + geospatial output: M4 -> M2 Change -> M2 Grounding -> M5 GIS
- Explicit grounding: M4 -> M2 Grounding -> optional M5 GIS
- Optical/SAR: M4 -> M3 -> optional M5 GIS

## 2. Module-by-module progress

> Progress below is an engineering assessment from the current repository state, not a claim about individual people's identities or effort.

| Module | Responsibility | Current implementation | Status | Main remaining work |
|---|---|---|---|---|
| **M1** | EarthDial / VQA | `m1_earthdial/` contains config, preprocessing, inference, server, adapter, examples and tests. A real M1->M4 integration test uses the supported offline backend. | **~90% — integrated** | Run/validate heavyweight real-model inference with the intended production weights/environment; improve deployment packaging if required. |
| **M2** | Change detection + grounding | Native change pipeline and native grounding adapter are present. Real M2->M4 integration and multi-step M2->Grounding->M5 acceptance coverage exist. | **~95% — integrated** | Validate against real mission imagery and the final trained model(s); tune quality metrics for SIH demo scenarios. |
| **M3** | Optical + SAR fusion | Optical/SAR pipeline, registration, fusion and model inference are present. Integration test exercises the real adapter and checks trained checkpoint inference. | **~90% — integrated** | Validate on final paired datasets, document model/checkpoint provenance, and verify GPU/CPU deployment expectations. |
| **M4** | Agentic routing/orchestration | Parser, classifier, planner, executor, registry, adapters, synthesizer, public API and acceptance tests are present. | **~95% — core complete** | Keep contracts stable, strengthen end-to-end CI, and add observability/authentication for deployment. |
| **M5** | GIS / geospatial evidence + API boundary | Native geospatial processing, CRS conversion, polygons, area, GeoJSON, Folium map and M4 adapter are implemented. FastAPI serves results and artifacts. | **~90% — integrated** | Harden production API, large-file handling, cleanup/retention, and final real-data validation. |
| **M6** | Frontend / judge dashboard | React/Vite dashboard, image upload, VQA/change/Optical-SAR views, GIS/evidence panels, history, report and team architecture UI are present. | **~85% — demo-ready** | Finish live-backend contract alignment, CORS/dev deployment validation, frontend E2E coverage, and final presentation polish. |

## 3. What is actually proven today

### Proven by repository tests

- M1 -> M4 real integration through the EarthDial adapter using the supported offline backend.
- M2 -> M4 real change-detection integration using deterministic synthetic imagery.
- M3 -> M4 real Optical/SAR adapter integration including trained checkpoint inference checks.
- M5 -> M4 native GIS integration generating evidence JSON, GeoJSON and an interactive map.
- M4 routing and multi-step orchestration with deterministic acceptance tests.

### Not yet equivalent to production validation

A passing contract/integration suite does **not** prove that a heavyweight model is available, GPU-compatible, accurate on the final SIH dataset, or operational under production load. Final model/data validation must still be performed in the intended runtime environment.

## 4. Judge-facing end-to-end workflow

### Scenario A — Image understanding

```text
M6 upload image
  -> M5 API
  -> M4 parser/classifier/planner
  -> M1 EarthDial
  -> M4 response synthesis
  -> M6 evidence/answer panel
```

### Scenario B — Change + location

```text
M6 upload BEFORE + AFTER
  -> M5 API
  -> M4
  -> M2 change detection
  -> M2 grounding
  -> M5 GIS
  -> M4 response
  -> M6 before/after + GIS evidence + report
```

### Scenario C — Optical + SAR

```text
M6 upload Optical + SAR
  -> M5 API
  -> M4
  -> M3 registration/fusion/inference
  -> optional M5 GIS
  -> M4 response
  -> M6 sensor comparison + confidence/evidence
```

## 5. Definition of "working"

SatQuery should be considered SIH-demo ready when all of these pass:

1. Root Python test suite passes.
2. M1 contract tests pass in CI.
3. M6 frontend `lint` passes.
4. M6 frontend `build` passes.
5. FastAPI `/health` (and the M6-compatible health route) returns success.
6. M6 Live mode can submit one real uploaded image and receives a structured M4 response.
7. M6 Live mode can submit a before/after pair and returns M2 + M5 evidence/artifacts.
8. M6 Live mode can submit Optical + SAR data and returns an M3 result.
9. Generated GeoJSON/map/evidence artifacts are reachable from the browser.
10. Demo Mode remains available as a deterministic offline fallback.

## 6. Final target architecture

```text
M6 UI
  |
  v
M5 FastAPI/API Gateway
  |
  v
M4 Agent Controller
  |
  +--> M1 VQA
  +--> M2 Change Detection
  +--> M2 Grounding
  +--> M3 Optical + SAR
  +--> M5 GIS
  |
  v
Structured AgentResponse
  |
  +--> answer
  +--> confidence
  +--> evidence
  +--> specialist results
  +--> execution trace
  +--> GIS artifacts
  |
  v
M6 Visualization / Report
```
