"""
Google Colab / Cloud GPU Inference Server for SatQuery AI M1 EarthDial.
Runs FastAPI with GPU-accelerated InternVLChatModel.
"""

import base64
import io
import os
import sys
import torch
from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("FastAPI and uvicorn are required to run this server.")
    print("Install via: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="SatQuery AI - EarthDial GPU Inference Server",
    version="1.0.0",
    description="GPU-accelerated EarthDial VLM inference endpoint for remote client integration."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
TOKENIZER = None
TRANSFORM = None
MODEL_ID = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB")


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image string")
    question: str = Field(..., description="Natural language question")
    num_beams: int = Field(default=5)
    temperature: float = Field(default=0.0)
    max_new_tokens: int = Field(default=128)


class AnalyzeResponse(BaseModel):
    answer: str
    model: str
    device: str
    tokens_generated: Optional[int] = None


def load_earthdial_model():
    global MODEL, TOKENIZER, TRANSFORM, MODEL_ID
    if MODEL is not None:
        return

    print(f"[*] Initializing EarthDial on device: {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    from transformers import AutoTokenizer, AutoModel
    from torchvision import transforms

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

    try:
        from earthdial.model.internvl_chat import InternVLChatModel
        model_cls = InternVLChatModel
    except ImportError:
        model_cls = AutoModel

    MODEL = model_cls.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    ).eval()

    if torch.cuda.is_available():
        MODEL = MODEL.cuda()

    image_size = getattr(MODEL.config, "force_image_size", None) or \
                 getattr(MODEL.config.vision_config, "image_size", 448)

    TRANSFORM = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"[+] EarthDial model successfully loaded on {MODEL.device}!")


@app.on_event("startup")
async def startup_event():
    load_earthdial_model()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model is still loading or unavailable.")

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    pixel_values = TRANSFORM(image).unsqueeze(0).to(device=MODEL.device, dtype=MODEL.dtype)

    generation_config = {
        "num_beams": req.num_beams,
        "max_new_tokens": req.max_new_tokens,
        "min_new_tokens": 1,
        "do_sample": req.temperature > 0.0,
        "temperature": req.temperature if req.temperature > 0.0 else 1.0,
    }

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            answer = MODEL.chat(
                tokenizer=TOKENIZER,
                pixel_values=pixel_values,
                question=req.question,
                generation_config=generation_config,
                verbose=False
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return AnalyzeResponse(
            answer=str(answer).strip(),
            model=MODEL_ID,
            device=str(MODEL.device)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

