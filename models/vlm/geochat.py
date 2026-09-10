"""Optional GeoChat-compatible VLM adapter.

The heavy dependencies are imported lazily so the lightweight SIH demo does
not require PyTorch/Transformers. Set SATQUERY_GEOCHAT_CHECKPOINT to a local
checkpoint directory to activate the learned path.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any


def run_geochat(image_path: str, question: str) -> dict[str, Any]:
    """Run an optional Hugging Face causal VLM checkpoint.

    The adapter intentionally fails closed: if the checkpoint or dependencies
    are missing, callers receive an explicit unavailable status instead of a
    fabricated answer.
    """
    checkpoint = Path(os.environ.get("SATQUERY_GEOCHAT_CHECKPOINT", ""))
    if not checkpoint.exists():
        return {
            "status": "unavailable",
            "reason": "SATQUERY_GEOCHAT_CHECKPOINT is not configured",
        }

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"missing optional dependency: {exc}"}

    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "CUDA is required for the configured GeoChat path"}

    processor = None
    model = None
    try:
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=question, return_tensors="pt")
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        answer = processor.decode(output[0], skip_special_tokens=True)
        return {
            "status": "success",
            "answer": answer,
            "checkpoint": str(checkpoint),
            "device": str(model.device),
        }
    except Exception as exc:  # noqa: BLE001 - adapter must expose runtime failures safely
        return {"status": "failed", "reason": str(exc), "checkpoint": str(checkpoint)}
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
