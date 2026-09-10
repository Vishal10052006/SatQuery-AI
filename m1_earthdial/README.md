# M1 — EarthDial VLM

**SatQuery AI Module M1** is the visual-perception specialist for optical Earth-observation imagery. It accepts a satellite image plus a natural-language question, runs EarthDial VLM inference, and returns a stable structured response consumed by M4.

## Definition of Done

M1 is complete when all of the following are true:

- [x] Image validation and corruption checks
- [x] RGB normalization and image metadata extraction
- [x] EarthDial 4B model integration
- [x] Direct CUDA backend
- [x] Remote GPU backend
- [x] Offline mock backend for CI/contract tests
- [x] Stable Pydantic request/response contract
- [x] Evidence metadata without fabricated confidence
- [x] M4 callable `analyze_image()` interface
- [x] CLI runner and JSON output
- [x] Reusable FastAPI GPU inference server
- [x] Colab T4 deployment notebook
- [x] Unit/integration tests for local, error, and remote transport paths
- [ ] Live EarthDial inference against an active T4 server — deployment/runtime gate

The last item is an operational verification step because model weights require a GPU and are not stored in this repository.

---

## 1. M1 Responsibility

```text
Satellite Image + Question
          ↓
Image validation / preprocessing
          ↓
EarthDial 4B VLM
          ↓
Visual Question Answering
          ↓
Structured M1 Response
          ↓
M4 Agent
```

### Supported tasks

- Visual Question Answering (VQA)
- Scene description
- Land-cover understanding
- Roads, buildings and infrastructure questions
- Water bodies, rivers, lakes and coastlines
- General optical satellite-scene understanding

### Explicitly outside M1

M1 does **not** perform change detection, SAR processing, optical-SAR fusion, GIS polygon creation, coordinate reprojection, or multi-agent orchestration. Those capabilities belong to other modules.

---

## 2. Model

Checkpoint:

```text
akshaydudhane/EarthDial_4B_RGB
```

EarthDial is a 4B-class vision-language model. The weights are downloaded by the GPU environment at runtime and are intentionally not committed to Git.

---

## 3. Architecture

```text
                     M4 Agent
                         │
                  analyze_image()
                         │
                         ▼
                 ┌───────────────┐
                 │ M1 EarthDial  │
                 │   Adapter     │
                 └───────┬───────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
          Direct GPU   Remote GPU   Mock
              │          │          │
              └──────────┼──────────┘
                         ▼
                  Structured JSON
                         │
                         ▼
                         M4
```

### Backend behavior

| Backend | Purpose | Real model? |
|---|---|---:|
| `direct` | Local CUDA workstation with sufficient VRAM | ✅ |
| `remote` | Local lightweight client → Colab/T4 or GPU server | ✅ |
| `mock` | Offline tests and CI | ❌ |
| `auto` | Direct CUDA → healthy remote → mock | Depends on availability |

**Important:** `mock` is never the production EarthDial path. It exists so the M1/M4 contract can be tested without downloading model weights.

---

## 4. Remote T4 Deployment

For a laptop without enough VRAM, use the included notebook:

```text
m1_earthdial/colab/EarthDial_Colab_Inference_Server.ipynb
```

### Colab

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the installation and model-loading cells.
4. Start the FastAPI server and Cloudflare tunnel.
5. Copy the generated `trycloudflare.com` URL.

The reusable server is also available as:

```text
m1_earthdial/server.py
```

For a normal CUDA host:

```bash
python -m m1_earthdial.server
```

### Local client

Linux/macOS:

```bash
export EARTHDIAL_BACKEND=remote
export EARTHDIAL_API_URL="https://YOUR-TUNNEL.trycloudflare.com"
```

PowerShell:

```powershell
$env:EARTHDIAL_BACKEND="remote"
$env:EARTHDIAL_API_URL="https://YOUR-TUNNEL.trycloudflare.com"
```

Verify the remote server first:

```bash
curl "$EARTHDIAL_API_URL/health"
```

Expected shape:

```json
{
  "status": "ready",
  "model": "akshaydudhane/EarthDial_4B_RGB",
  "cuda": true,
  "device": "Tesla T4"
}
```

---

## 5. M4 Integration Contract

M4 calls exactly one public function:

```python
from m1_earthdial import analyze_image

result = analyze_image(
    image_path="path/to/satellite_scene.jpg",
    question="What are the dominant land cover types visible in this scene?",
)
```

M1 returns:

```json
{
  "task": "image_vqa",
  "question": "What are the dominant land cover types visible in this scene?",
  "answer": "<EarthDial generated answer>",
  "model": "EarthDial",
  "confidence": null,
  "evidence": {
    "image_path": "...",
    "image_format": "JPEG",
    "image_size": [512, 512],
    "model_name": "akshaydudhane/EarthDial_4B_RGB",
    "backend": "remote_colab",
    "inference_time_seconds": 1.42,
    "timestamp": "...",
    "extra": {
      "num_beams": 5,
      "temperature": 0.0,
      "max_new_tokens": 128
    }
  },
  "success": true,
  "error": null
}
```

### Confidence policy

`confidence` is intentionally `null`. EarthDial's autoregressive token likelihoods are not treated as calibrated semantic confidence, so M1 does not fabricate a score.

---

## 6. CLI

```bash
python -m m1_earthdial.inference \
  --image m1_earthdial/examples/sample_satellite.jpg \
  --question "What can you see in this satellite image?" \
  --backend remote \
  --api-url "$EARTHDIAL_API_URL"
```

Save JSON:

```bash
python -m m1_earthdial.inference \
  --image m1_earthdial/examples/sample_satellite.jpg \
  --question "Are there any water bodies?" \
  --backend remote \
  --api-url "$EARTHDIAL_API_URL" \
  --output m1_earthdial/outputs/water_query.json
```

---

## 7. Tests

Run the complete M1 test suite from the repository root:

```bash
python -m unittest discover -s m1_earthdial/tests -p "test_*.py" -v
```

Run the demonstration:

```bash
python m1_earthdial/tests/run_e2e_demo.py
```

The test suite covers:

- general VQA
- land-cover questions
- water/river questions
- missing images
- invalid questions
- JSON serialization
- remote request payload construction
- remote empty-answer rejection
- backend validation

The offline tests do not require EarthDial weights or CUDA.

---

## 8. Installation

### Lightweight local client

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r m1_earthdial/requirements.txt
```

### GPU environment

```bash
pip install -r m1_earthdial/requirements-gpu.txt
```

---

## 9. Repository Layout

```text
m1_earthdial/
├── README.md
├── __init__.py
├── config.py
├── earthdial_adapter.py
├── inference.py
├── preprocessing.py
├── schemas.py
├── server.py
├── requirements.txt
├── requirements-gpu.txt
├── colab/
│   └── EarthDial_Colab_Inference_Server.ipynb
├── examples/
│   ├── generate_sample_image.py
│   ├── sample_query.json
│   └── sample_satellite.jpg
└── tests/
    ├── __init__.py
    ├── run_e2e_demo.py
    ├── test_inference.py
    ├── test_preprocessing.py
    └── test_schemas.py
```

## Final Runtime Gate

The final deployment proof is:

```text
Local M4
   ↓
M1 analyze_image()
   ↓
Remote EarthDial API
   ↓
T4 GPU
   ↓
Real EarthDial answer
   ↓
M1 structured response
   ↓
M4 synthesis
```

Once `/health` reports `ready` and the CLI returns `success: true` with a real generated answer, M1 has passed the complete runtime DoD.