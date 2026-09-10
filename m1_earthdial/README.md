# M1 — EarthDial VLM (Visual Question Answering & Image Understanding)

Module **M1** of **SatQuery AI** provides deep Earth observation visual reasoning and natural language question answering over high-resolution satellite imagery using **EarthDial** (CVPR 2025).

---

## 1. Overview

Within the SatQuery AI system, **M1** functions as the primary visual perception engine for optical satellite imagery. When a user submits a natural-language query about a satellite scene, the **M4 Agent** dispatches the image and query to M1. M1 validates and preprocesses the image, feeds it to the **EarthDial VLM** (`EarthDial_4B_RGB`), and returns a standardized, structured JSON response containing the generated answer and verifiable evidence metadata.

### Pipeline

```
Satellite Image (.jpg, .png, .tif)
         ↓
Image Preprocessing & Validation
         ↓
   EarthDial VLM
         ↓
Natural Language Question
         ↓
Visual Question Answering (VQA) / Image Understanding
         ↓
Structured Result (JSON Schema)
         ↓
     M4 Agent
```

---

## 2. Architecture & M4 Integration Flow

```
+-------------------------------------------------------------+
|                         User Query                          |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|                          M4 Agent                           |
|       (Orchestrator / Planning & Task Delegation)           |
+-------------------------------------------------------------+
                              │  analyze_image(img, question)
                              ▼
+-------------------------------------------------------------+
|                     M1 EarthDial Module                     |
|  - Preprocessing & format validation                        |
|  - Resolution and channel normalization                     |
|  - Backend routing (Direct GPU / Remote Colab / Mock)       |
+-------------------------------------------------------------+
                              │  Tensor / Base64 Payload
                              ▼
+-------------------------------------------------------------+
|                   EarthDial 4B VLM Engine                   |
|  - Model: akshaydudhane/EarthDial_4B_RGB                    |
|  - Architecture: InternVLChatModel                          |
|  - Autoregressive Chat Inference                            |
+-------------------------------------------------------------+
                              │  Raw Text Response
                              ▼
+-------------------------------------------------------------+
|                      Structured Result                      |
|  - task: "image_vqa"                                        |
|  - answer: Generated text                                   |
|  - confidence: null (not fabricated)                        |
|  - evidence: { path, size, format, model, time }           |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|                          M4 Agent                           |
|             (Synthesizes Final Response to User)            |
+-------------------------------------------------------------+
```

---

## 3. Supported Tasks & Scope Boundaries

### Supported Tasks in M1
- **Visual Question Answering (VQA)**: e.g. *"What can you see in this satellite image?"*
- **Scene Description**: e.g. *"Describe the major features visible in this scene."*
- **Earth Observation Understanding**:
  - Land cover categorization (agricultural, forest, barren, urban).
  - Infrastructure & object detection queries (presence of roads, buildings, runways).
  - Hydrological presence queries (water bodies, rivers, lakes, coastlines).

### M1 Scope Boundaries
- **M1 IS NOT responsible for**: Change detection, SAR radar processing, optical-SAR fusion, GIS polygon/shapefile creation, geospatial coordinate reprojection, or multi-agent orchestration. Those tasks are strictly relegated to other SatQuery AI modules.

---

## 4. Hardware & GPU Requirements

EarthDial is a 4.15-billion parameter Vision-Language Model requiring substantial VRAM for weight loading and activation tensors:

| Execution Mode | Weights Precision | Minimum GPU VRAM | Recommended Device |
|---|---|---|---|
| **Direct Full Precision** | bfloat16 / float16 | **10 – 12 GB** | NVIDIA T4 (16GB), V100, A10, A100 |
| **Direct Quantized** | 4-bit / 8-bit (bitsandbytes) | **5 – 6 GB** | NVIDIA RTX 3060/4060 (CUDA only) |
| **Remote Colab GPU** | bfloat16 | **16 GB (Cloud)** | Free Google Colab T4 GPU |
| **Local CPU (Non-GPU)** | - | *Not Feasible* | OOM crash; requires Remote Colab |

> **Note on Local Execution:** Machines without a dedicated NVIDIA CUDA GPU (such as integrated AMD Radeon or Intel Iris graphics with 8 GB system RAM) cannot load the 8.3 GB model weights into memory. For such machines, M1 provides a seamless **Remote Colab GPU Backend**.

---

## 5. Installation

### 5.1 Local Client Setup (Lightweight)
On your local development machine, install the lightweight client requirements (Pydantic, Pillow, Requests):

```bash
# Using Python 3.10 - 3.12
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install M1 client dependencies
pip install -r m1_earthdial/requirements.txt
```

### 5.2 GPU Environment Setup (For Colab or Local CUDA Workstation)
If running directly on a machine with a CUDA GPU:

```bash
pip install -r m1_earthdial/requirements-gpu.txt
```

---

## 6. Model Setup & Execution Backends

EarthDial weights are hosted on Hugging Face Hub:
- Repository ID: [`akshaydudhane/EarthDial_4B_RGB`](https://huggingface.co/akshaydudhane/EarthDial_4B_RGB)
- Size: ~8.29 GB (2 safetensors files)

M1 features a **Dual-Backend Adapter** controlled via environment variable `EARTHDIAL_BACKEND`:

### Option A: Free Google Colab T4 GPU (Recommended for non-GPU machines)
1. Open the included notebook in Google Colab:
   `m1_earthdial/colab/EarthDial_Colab_Inference_Server.ipynb`
2. Set Runtime to **T4 GPU** (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Run all cells. The notebook will start the FastAPI model server and output a public URL (e.g. `https://xxxx.trycloudflare.com`).
4. Set the environment variable on your local machine:
   ```powershell
   # PowerShell:
   $env:EARTHDIAL_API_URL="https://xxxx.trycloudflare.com"
   $env:EARTHDIAL_BACKEND="remote"
   ```
5. M1 now forwards inference queries to the Colab GPU transparently!

### Option B: Direct Local GPU
If your machine has a CUDA GPU with $\ge$10 GB VRAM:
```powershell
$env:EARTHDIAL_BACKEND="direct"
```

### Option C: Offline Mock Engine
For offline contract testing or CI/CD pipelines without network or GPU:
```powershell
$env:EARTHDIAL_BACKEND="mock"
```

---

## 7. Python API (How M4 Calls M1)

M4 can invoke M1 with a single function call:

```python
from m1_earthdial import analyze_image

# Invoke EarthDial analysis
result = analyze_image(
    image_path="path/to/satellite_scene.jpg",
    question="What are the dominant land cover types and major features visible in this scene?"
)

# Access structured fields
print("Answer:", result["answer"])
print("Model:", result["model"])
print("Evidence:", result["evidence"])
```

### Structured Output Schema
```json
{
  "task": "image_vqa",
  "question": "What are the dominant land cover types and major features visible in this scene?",
  "answer": "The image exhibits agricultural parcel divisions, a central meandering river channel, and a localized built-up urban cluster.",
  "model": "EarthDial",
  "confidence": null,
  "evidence": {
    "image_path": "C:\\Users\\...\\sample_satellite.jpg",
    "image_format": "JPEG",
    "image_size": [512, 512],
    "model_name": "akshaydudhane/EarthDial_4B_RGB",
    "backend": "remote_colab",
    "inference_time_seconds": 1.42,
    "timestamp": "2026-09-10T11:47:26.115835+00:00",
    "extra": {
      "num_beams": 5,
      "temperature": 0.0
    }
  },
  "success": true,
  "error": null
}
```

### Confidence Handling Notice
EarthDial generates textual answers autoregressively. Token probabilities in generative open-domain models do not correspond to calibrated semantic certainty. To maintain strict scientific integrity and avoid fabricating artificial scores, **`confidence` is strictly set to `null`**.

---

## 8. Command-Line Interface (CLI)

Run inference directly from your terminal:

```bash
# Basic query
python -m m1_earthdial.inference \
  --image m1_earthdial/examples/sample_satellite.jpg \
  --question "What can you see in this satellite image?"

# Custom output destination
python -m m1_earthdial.inference \
  --image m1_earthdial/examples/sample_satellite.jpg \
  --question "Are there any water bodies?" \
  --output m1_earthdial/outputs/water_query.json
```

---

## 9. Running Tests

Run the complete unit and integration test suite:

```bash
# Run all test modules
python -m unittest discover -s m1_earthdial/tests -p "test_*.py" -v

# Run interactive 3-query demonstration
python m1_earthdial/tests/run_e2e_demo.py
```

---

## 10. Troubleshooting

| Symptom / Error | Cause | Fix |
|---|---|---|
| `ImagePreprocessingError: Image file not found` | The provided path does not exist. | Verify path using `Path(img_path).exists()`. Use absolute or relative paths from workspace root. |
| `Unsupported image extension '.xyz'` | File extension not in supported list. | Supported extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.webp`. Convert file first. |
| `ConnectionError: Could not connect to EarthDial GPU server` | Remote Colab server is not running or URL is wrong. | Check that Colab notebook Step 5 is running, copy the `trycloudflare.com` URL, and set `$env:EARTHDIAL_API_URL`. |
| `OutOfMemoryError: CUDA out of memory` | Local GPU VRAM < 10 GB. | Use Google Colab T4 GPU server or launch with `--load-in-8bit` / `--load-in-4bit`. |
| `Torch not compiled with CUDA enabled` | Running direct GPU inference on CPU machine. | Switch backend to `remote` ($env:EARTHDIAL_BACKEND="remote") and run the Colab server. |

