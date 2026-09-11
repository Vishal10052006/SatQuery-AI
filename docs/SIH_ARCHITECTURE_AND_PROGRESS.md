# SatQuery AI — SIH Architecture & Module Progress

## 1. System Architecture

```text
User / M6 Frontend
        |
        v
+-----------------------+
| M6 Presentation Layer |
| Dashboard / GIS / UX  |
+-----------+-----------+
            | HTTP JSON / multipart
            v
+-----------------------+
| M5 FastAPI / API      |
| upload / artifacts    |
+-----------+-----------+
            v
+-----------------------+
| M4 Agentic Controller |
| Parser -> Classifier  |
| -> Planner -> Executor|
| -> Synthesizer        |
+-----+---------+-------+
      |         |        \
      v         v         v
   +------+  +------+  +--------+
   |  M1  |  |  M2  |  |   M3   |
   | VQA  |  |Change|  |Optical |
   |Earth |  |+Ground| | + SAR  |
   |Dial  |  |ing   |  | Fusion |
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
                       v
                +-------------+
                | M4 Response |
                | evidence +  |
                | confidence + |
                | trace        |
                +------+------+ 
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

Supported specialist routes:

- VQA: M4 -> M1
- Change detection: M4 -> M2
- Change + localization + geospatial output: M4 -> M2 Change -> M2 Grounding -> M5 GIS
- Explicit grounding: M4 -> M2 Grounding -> optional M5 GIS
- Optical/SAR: M4 -> M3 -> optional M5 GIS

## 2. Module-by-module progress

> Progress is an engineering assessment of repository implementation. It is not a claim about individual identities or effort.

| Module | Responsibility | Current implementation | Status | Remaining validation/work |
|---|---|---|---|---|
| **M1** | EarthDial / VQA | Config, preprocessing, inference server, adapter, examples and tests; real adapter contract exists. | **~90% — integrated** | Validate heavyweight EarthDial weights on intended GPU/runtime; deployment packaging. |
| **M2** | Change detection + grounding | Validation, preprocessing, true geospatial reprojection, deterministic baseline, RCD adapter, region extraction, grounding and geospatial localization. | **~95% — integrated** | Validate final mission imagery/models and tune demo thresholds. |
| **M3** | Optical + SAR fusion | Optical/SAR preprocessing, cloud masking, SAR calibration + linear-domain speckle filtering, terrain adapter, CRS reprojection, fine alignment, feature/early fusion, trained-weight-only inference, confidence. | **~95% — integrated/hardened** | Validate paired mission datasets, final trained checkpoint provenance and GPU/CPU deployment. |
| **M4** | Agentic controller | Parser, classifier, planner, executor, registry, adapters, synthesizer, public API and acceptance tests. | **~95% — core complete** | Keep contracts stable; final end-to-end regression and deployment observability/auth. |
| **M5** | GIS / geospatial evidence + API | Native geospatial processing, CRS conversion, polygons, area, GeoJSON, map artifacts, M4 adapter and FastAPI boundary. | **~90% — integrated** | Large-file handling, cleanup/retention, hardened production API, real-data validation. |
| **M6** | Frontend / judge dashboard | React/Vite dashboard, uploads, VQA/change/Optical-SAR views, GIS/evidence, history/report/team UI, live API contract. | **~85% — demo-ready** | Final live-backend E2E validation, deployment/CORS checks and presentation polish. |

## 3. What is actually proven

### Proven by repository-level tests/contracts

- M1 -> M4 adapter/integration boundary.
- M2 -> M4 deterministic change-detection integration and multi-step M2 -> Grounding -> GIS wiring.
- M3 -> M4 native adapter path and model-wrapper architecture checks.
- M5 -> M4 native GIS integration and evidence artifact generation.
- M4 deterministic routing, planning, execution and synthesis.
- M6 frontend lint/build workflow and FastAPI multipart contract tests.

### Not equivalent to production validation

A passing contract/integration suite does **not** prove that a heavyweight checkpoint is available, GPU-compatible, accurate on the final SIH dataset, or operational under production load. Final model/data validation must be performed in the intended runtime environment.

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
5. FastAPI `/health` and `/api/health` return success.
6. M6 Live mode submits one real uploaded image and receives a structured M4 response.
7. M6 Live mode submits a before/after pair and returns M2 + M5 evidence/artifacts.
8. M6 Live mode submits Optical + SAR data and returns an M3 result.
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
