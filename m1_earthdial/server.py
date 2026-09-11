"""
Reusable FastAPI server for M1 EarthDial inference.

This server is intended for a CUDA-capable host such as a Google Colab T4.
The local SatQuery M1 adapter sends only the image bytes and VQA request;
model execution remains on the GPU server.

Reference:
    EarthDial_4B_RGB: akshaydudhane/EarthDial_4B_RGB
"""

import base64
import io
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms

MODEL_ID = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB")
HOST = os.getenv("EARTHDIAL_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("EARTHDIAL_SERVER_PORT", "8000"))

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
    num_beams: int = Field(default=5, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=128, ge=1, le=512)


if not torch.cuda.is_available():
    raise RuntimeError(
        "M1 EarthDial server requires a CUDA GPU. Start this service on a "
        "GPU runtime such as Google Colab T4/A100."
    )

# EarthDial publishes BF16 weights. Use FP16 on CUDA devices without BF16 support.
MODEL_DTYPE = torch.bfloat16
if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
    MODEL_DTYPE = torch.float16

# Load once at process startup so every request reuses the same model.
print(f"[M1] Loading EarthDial checkpoint: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
)
model = AutoModel.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
    torch_dtype=MODEL_DTYPE,
    device_map="auto",
    trust_remote_code=True,
).eval()

vision_config = getattr(model.config, "vision_config", None)
image_size = getattr(model.config, "force_image_size", None) or getattr(
    vision_config, "image_size", 448
)
transform = transforms.Compose([
    transforms.Resize(
        (image_size, image_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

print(f"[M1] EarthDial model loaded successfully using {MODEL_DTYPE}.")


@app.get("/health")
def health():
    """Return readiness information used by M1 auto-routing and operators."""
    return {
        "status": "ready",
        "model": MODEL_ID,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": str(MODEL_DTYPE),
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

    return {
        "answer": answer,
        "model": MODEL_ID,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
