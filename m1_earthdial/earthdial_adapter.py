"""
Core adapter module for SatQuery AI Module M1 (EarthDial).
Implements direct GPU inference, remote Colab GPU client, and offline testing engine.
Provides the clean `analyze_image` interface for M4 Agent.
"""

import base64
import io
import time
from typing import Any, Dict, Optional
from PIL import Image

from .config import EarthDialConfig, default_config
from .preprocessing import load_and_preprocess_image, ImagePreprocessingError
from .schemas import EvidenceMetadata, ImageVQARequest, ImageVQAResponse


class EarthDialAdapter:
    """
    Adapter encapsulating EarthDial vision-language model execution.
    Supports:
      1. Direct execution on CUDA GPU workstations.
      2. Remote execution against a Google Colab GPU inference server.
      3. Offline mock execution for M4 Agent integration validation.
    """

    def __init__(self, config: Optional[EarthDialConfig] = None):
        self.config = config or default_config
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._resolved_backend = self._determine_backend()

    def _determine_backend(self) -> str:
        """Determines the appropriate backend based on configuration and hardware."""
        backend_choice = self.config.backend.lower()

        if backend_choice == "mock":
            return "mock"
        elif backend_choice == "direct":
            return "direct_gpu"
        elif backend_choice == "remote":
            return "remote_colab"
        elif backend_choice == "auto":
            # 1. Check if local CUDA GPU is available
            try:
                import torch
                if torch.cuda.is_available():
                    return "direct_gpu"
            except ImportError:
                pass

            # 2. Check if remote Colab server is reachable
            try:
                import requests
                health_url = self.config.api_url.rstrip("/") + "/health"
                r = requests.get(health_url, timeout=1.0)
                if r.status_code == 200:
                    return "remote_colab"
            except Exception:
                pass

            # 3. Fallback to mock engine for local testing
            return "mock"
        else:
            return "mock"

    def _ensure_direct_model_loaded(self):
        """Loads EarthDial model into GPU memory for direct local inference."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer
            from .preprocessing import get_image_tensor_transform
        except ImportError as e:
            raise RuntimeError(
                "Direct GPU inference requires PyTorch and transformers. "
                "Install requirements-gpu.txt or switch to remote Colab mode."
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Direct EarthDial inference requires a CUDA-enabled GPU with >=10GB VRAM. "
                "No CUDA GPU detected on this machine. Please use the Google Colab GPU backend."
            )

        print(f"[EarthDialAdapter] Loading checkpoint: {self.config.model_checkpoint}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint,
            trust_remote_code=True,
            use_fast=False
        )

        try:
            from earthdial.model.internvl_chat import InternVLChatModel
            model_class = InternVLChatModel
        except ImportError:
            from transformers import AutoModel
            model_class = AutoModel

        self._model = model_class.from_pretrained(
            self.config.model_checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        image_size = getattr(self._model.config, "force_image_size", None) or \
                     getattr(self._model.config.vision_config, "image_size", 448)
        self._transform = get_image_tensor_transform(input_size=image_size)
        print("[EarthDialAdapter] Model and tokenizer successfully loaded on GPU.")

    def _infer_direct(self, img: Image.Image, question: str, req: ImageVQARequest) -> str:
        """Executes inference directly on local GPU."""
        self._ensure_direct_model_loaded()
        import torch

        pixel_values = self._transform(img).unsqueeze(0).cuda().to(torch.bfloat16)

        generation_config = {
            "num_beams": req.num_beams,
            "max_new_tokens": req.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": req.temperature > 0.0,
            "temperature": req.temperature if req.temperature > 0.0 else 1.0,
        }

        answer = self._model.chat(
            tokenizer=self._tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            verbose=False
        )
        return str(answer).strip()

    def _infer_remote(self, image_path: str, question: str, req: ImageVQARequest) -> str:
        """Sends inference request to remote Google Colab GPU server."""
        import requests

        api_url = self.config.api_url.rstrip("/") + "/analyze"

        # Encode image as base64 for reliable REST transport
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "image_base64": image_b64,
            "question": question,
            "num_beams": req.num_beams,
            "temperature": req.temperature,
            "max_new_tokens": req.max_new_tokens
        }

        try:
            response = requests.post(api_url, json=payload, timeout=self.config.timeout_seconds)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to EarthDial GPU server at {self.config.api_url}. "
                "Ensure that the Google Colab server notebook is running and EARTHDIAL_API_URL is set."
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Request to EarthDial GPU server timed out after {self.config.timeout_seconds} seconds."
            ) from e

        if response.status_code != 200:
            raise RuntimeError(
                f"Remote EarthDial server returned error {response.status_code}: {response.text}"
            )

        data = response.json()
        return str(data.get("answer", "")).strip()

    def _infer_mock(self, img_meta: dict, question: str) -> str:
        """Simulates domain-accurate Earth observation responses for offline testing."""
        q_lower = question.lower()
        w, h = img_meta["width"], img_meta["height"]

        if any(term in q_lower for term in ["water", "river", "lake", "ocean", "sea"]):
            return (
                "Based on the visual features in this satellite imagery, dark absorption zones "
                "characteristic of open water bodies or river channels are discernible."
            )
        elif any(term in q_lower for term in ["building", "urban", "city", "built-up", "structure"]):
            return (
                "The scene exhibits high-density geometric patterns and high reflectance rooftops "
                "indicative of built-up urban infrastructure and road networks."
            )
        elif any(term in q_lower for term in ["land cover", "type", "vegetation", "forest", "field"]):
            return (
                "The dominant land cover consists of heterogeneous agricultural parcels, scattered tree "
                "canopies, and interconnected transport corridors."
            )
        else:
            return (
                f"The satellite image ({w}x{h} resolution) shows a composite Earth observation scene "
                "comprising surface terrain, mixed land-use patterns, and distinct spectral boundaries."
            )

    def analyze(
        self,
        image_path: str,
        question: str,
        task: str = "image_vqa",
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> ImageVQAResponse:
        """
        Processes a satellite image and natural-language question, returning a structured response.
        """
        start_time = time.perf_counter()

        # Step 1: Validate input parameters
        if not question or not isinstance(question, str) or len(question.strip()) < 2:
            return ImageVQAResponse(
                task=task,
                question=question or "",
                answer="",
                model="EarthDial",
                confidence=None,
                success=False,
                error="Invalid question: question must be a non-empty string with at least 2 characters.",
                evidence=EvidenceMetadata(
                    image_path=str(image_path),
                    model_name=self.config.model_checkpoint,
                    backend=self._resolved_backend,
                )
            )

        # Step 2: Validate and load image
        try:
            pil_img, img_metadata = load_and_preprocess_image(image_path)
        except ImagePreprocessingError as e:
            return ImageVQAResponse(
                task=task,
                question=question,
                answer="",
                model="EarthDial",
                confidence=None,
                success=False,
                error=f"Image preprocessing failed: {str(e)}",
                evidence=EvidenceMetadata(
                    image_path=str(image_path),
                    model_name=self.config.model_checkpoint,
                    backend=self._resolved_backend,
                )
            )

        # Step 3: Prepare request object
        req = ImageVQARequest(
            image_path=str(image_path),
            question=question.strip(),
            task=task,
            num_beams=num_beams if num_beams is not None else self.config.default_num_beams,
            temperature=temperature if temperature is not None else self.config.default_temperature,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.config.default_max_new_tokens,
        )

        # Step 4: Execute inference via resolved backend
        backend_used = self._resolved_backend
        try:
            if backend_used == "direct_gpu":
                answer = self._infer_direct(pil_img, question, req)
            elif backend_used == "remote_colab":
                answer = self._infer_remote(str(image_path), question, req)
            elif backend_used == "mock":
                answer = self._infer_mock(img_metadata, question)
            else:
                raise ValueError(f"Unknown backend: {backend_used}")

            elapsed = round(time.perf_counter() - start_time, 3)

            evidence = EvidenceMetadata(
                image_path=img_metadata["image_path"],
                image_format=img_metadata["original_format"],
                image_size=(img_metadata["width"], img_metadata["height"]),
                model_name=self.config.model_checkpoint,
                backend=backend_used,
                inference_time_seconds=elapsed,
                extra={
                    "num_beams": req.num_beams,
                    "temperature": req.temperature,
                }
            )

            return ImageVQAResponse(
                task=task,
                question=req.question,
                answer=answer,
                model="EarthDial",
                confidence=None,  # Not fabricated; EarthDial autoregressive text output
                evidence=evidence,
                success=True,
                error=None
            )

        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            evidence = EvidenceMetadata(
                image_path=img_metadata.get("image_path", str(image_path)),
                image_format=img_metadata.get("original_format"),
                image_size=(img_metadata.get("width", 0), img_metadata.get("height", 0)),
                model_name=self.config.model_checkpoint,
                backend=backend_used,
                inference_time_seconds=elapsed,
            )
            return ImageVQAResponse(
                task=task,
                question=req.question,
                answer="",
                model="EarthDial",
                confidence=None,
                evidence=evidence,
                success=False,
                error=f"Inference execution failed on backend '{backend_used}': {str(e)}"
            )


# Global adapter singleton
_default_adapter: Optional[EarthDialAdapter] = None


def get_adapter(config: Optional[EarthDialConfig] = None) -> EarthDialAdapter:
    """Returns or initializes the default EarthDialAdapter instance."""
    global _default_adapter
    if config is not None:
        return EarthDialAdapter(config=config)
    if _default_adapter is None:
        _default_adapter = EarthDialAdapter()
    return _default_adapter


def set_adapter(adapter: Optional[EarthDialAdapter]):
    """Sets or resets the default global adapter."""
    global _default_adapter
    _default_adapter = adapter


def analyze_image(
    image_path: str,
    question: str,
    adapter: Optional[EarthDialAdapter] = None,
    config: Optional[EarthDialConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Standard top-level interface for M4 Agent to invoke EarthDial VQA.
    
    Args:
        image_path: Path to the satellite image.
        question: Natural language question about the image.
        adapter: Optional custom EarthDialAdapter instance.
        config: Optional custom EarthDialConfig.
        **kwargs: Optional generation arguments (task, num_beams, temperature, max_new_tokens).
        
    Returns:
        Structured Python dictionary matching the required JSON format:
        {
            "task": "image_vqa",
            "question": "...",
            "answer": "...",
            "model": "EarthDial",
            "confidence": null,
            "evidence": { ... },
            "success": true,
            "error": null
        }
    """
    target_adapter = adapter or (get_adapter(config) if config else get_adapter())
    response = target_adapter.analyze(image_path=image_path, question=question, **kwargs)
    return response.to_dict()

