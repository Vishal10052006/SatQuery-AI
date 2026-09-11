"""
Reusable FastAPI server for M1 EarthDial inference.

This service is intended for a CUDA-capable host such as Google Colab T4/A100.
It uses the official EarthDial model implementation and preprocessing path,
with optional 4-bit NF4 quantization for lower-VRAM GPU runtimes.
"""

import base64
import io
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoTokenizer

from earthdial.model.internvl_chat import InternVLChatModel
from m1_earthdial.preprocessing import get_image_tensor_transform

MODEL_ID = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB")
HOST = os.getenv("EARTHDIAL_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("EARTHDIAL_SERVER_PORT", "8000"))
LOAD_IN_4BIT = os.getenv("EARTHDIAL_LOAD_IN_4BIT", "true").lower() in {
    "1", "true", "yes", "on"
}
GPU_MEMORY_GIB = float(os.getenv("EARTHDIAL_GPU_MEMORY_GIB", "14"))
CPU_MEMORY_GIB = float(os.getenv("EARTHDIAL_CPU_MEMORY_GIB", "16"))
BNB_QUANT_TYPE = os.getenv("EARTHDIAL_BNB_QUANT_TYPE", "nf4")
BNB_DOUBLE_QUANT = os.getenv("EARTHDIAL_BNB_DOUBLE_QUANT", "true").lower() in {
    "1", "true", "yes", "on"
}

app = FastAPI(
    title="SatQuery AI — EarthDial API",
    version="1.0.0",
    description="GPU inference service for SatQuery AI M1 EarthDial VQA.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    """Validated request sent by the local M1 remote backend."""

    image_base64: str = Field(..., min_length=1)
    question: str = Field(..., min_length=2)
    num_beams: int = Field(default=1, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=64, ge=1, le=512)


if not torch.cuda.is_available():
    raise RuntimeError(
        "M1 EarthDial server requires a CUDA GPU. Start this service on a "
        "GPU runtime such as Google Colab T4/A100."
    )

MODEL_DTYPE = torch.bfloat16
if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
    MODEL_DTYPE = torch.float16

print(f"[M1] Loading EarthDial checkpoint: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
)

model_kwargs = {
    "low_cpu_mem_usage": True,
    "torch_dtype": MODEL_DTYPE,
    "device_map": "auto",
    "trust_remote_code": True,
}

if LOAD_IN_4BIT:
    model_kwargs.update(
        {
            "max_memory": {
                0: f"{GPU_MEMORY_GIB}GiB",
                "cpu": f"{CPU_MEMORY_GIB}GiB",
            },
            "load_in_4bit": True,
            "bnb_4bit_quant_type": BNB_QUANT_TYPE,
            "bnb_4bit_compute_dtype": MODEL_DTYPE,
            "bnb_4bit_use_double_quant": BNB_DOUBLE_QUANT,
        }
    )

model = InternVLChatModel.from_pretrained(MODEL_ID, **model_kwargs).eval()

image_size = getattr(model.config, "force_image_size", None) or getattr(
    model.config.vision_config, "image_size", 448
)
transform = get_image_tensor_transform(input_size=image_size)

print(
    f"[M1] EarthDial model loaded successfully using {MODEL_DTYPE}; "
    f"4-bit={LOAD_IN_4BIT}."
)


@app.get("/health")
def health():
    """Return readiness information used by M1 auto-routing and operators."""
    return {
        "status": "ready",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": str(MODEL_DTYPE),
        "load_in_4bit": LOAD_IN_4BIT,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Decode the image, run EarthDial VQA, and return the generated answer."""
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    question = request.question.strip()
    if len(question) < 2:
        raise HTTPException(status_code=422, detail="Question must contain at least 2 characters.")

    pixel_values = transform(image).unsqueeze(0).cuda().to(MODEL_DTYPE)
    generation_config = {
        "num_beams": request.num_beams,
        "max_new_tokens": request.max_new_tokens,
        "min_new_tokens": 1,
        "do_sample": request.temperature > 0.0,
        "temperature": request.temperature if request.temperature > 0.0 else 1.0,
    }

    try:
        with torch.no_grad():
            answer = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                verbose=False,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"EarthDial inference failed: {exc}") from exc

    answer = str(answer).strip()
    if not answer:
        raise HTTPException(status_code=500, detail="EarthDial returned an empty answer.")

    return {"answer": answer, "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
