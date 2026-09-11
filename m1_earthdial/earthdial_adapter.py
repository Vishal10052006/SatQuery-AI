"""
Core adapter module for SatQuery AI Module M1 (EarthDial).

M1 is the visual-perception specialist. It validates satellite imagery,
selects the configured EarthDial execution backend, performs inference, and
returns the stable response contract consumed by M4.

Backends:
    - direct: EarthDial runs on a local CUDA GPU.
    - remote: EarthDial runs behind the supplied remote GPU API.
    - mock: deterministic offline simulator for tests only.
    - auto: choose direct CUDA, then a healthy remote endpoint, then mock.
"""

import base64
import time
from typing import Any, Dict, Optional

from PIL import Image

from .config import EarthDialConfig, default_config
from .preprocessing import ImagePreprocessingError, load_and_preprocess_image
from .schemas import EvidenceMetadata, ImageVQARequest, ImageVQAResponse


class EarthDialAdapter:
    """Adapter encapsulating EarthDial VLM execution and the M4 contract."""

    VALID_BACKENDS = {"auto", "remote", "direct", "mock"}

    def __init__(self, config: Optional[EarthDialConfig] = None):
        self.config = config or default_config
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._resolved_backend = self._determine_backend()

    def _determine_backend(self) -> str:
        """Resolve the requested backend without silently masking bad config."""
        backend_choice = self.config.backend.lower().strip()
        if backend_choice not in self.VALID_BACKENDS:
            raise ValueError(
                f"Unsupported EARTHDIAL_BACKEND '{self.config.backend}'. "
                f"Expected one of: {sorted(self.VALID_BACKENDS)}"
            )

        if backend_choice == "mock":
            return "mock"
        if backend_choice == "direct":
            return "direct_gpu"
        if backend_choice == "remote":
            return "remote_colab"

        # Auto mode prefers a real local GPU, then a reachable remote server.
        try:
            import torch
            if torch.cuda.is_available():
                return "direct_gpu"
        except ImportError:
            pass

        if self._remote_healthcheck():
            return "remote_colab"

        # Mock is deliberately the final auto-mode fallback for development/CI.
        return "mock"

    def _remote_healthcheck(self) -> bool:
        """Return True when the configured remote server exposes a healthy HTTP endpoint.

        Different FastAPI deployments use different health payloads (for example
        ``{"status": "ok"}``, ``{"status": "ready"}``, or an empty 200 response).
        For backend discovery, HTTP 200 is the reliable signal; explicit failure
        states are rejected without requiring one exact response schema.
        """
        try:
            import requests

            health_url = self.config.api_url.rstrip("/") + "/health"
            response = requests.get(health_url, timeout=2.0)
            if response.status_code != 200:
                return False

            # If JSON is supplied, reject explicit unhealthy states while accepting
            # common healthy values. Non-JSON 200 responses remain valid health checks.
            try:
                data = response.json()
            except ValueError:
                return True

            if not isinstance(data, dict):
                return True

            status = str(data.get("status", "")).lower().strip()
            if status in {"error", "failed", "failure", "unhealthy", "offline"}:
                return False
            return True
        except Exception:
            return False

    def _ensure_direct_model_loaded(self):
        """Load EarthDial once into a sufficiently capable local CUDA device."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer
            from .preprocessing import get_image_tensor_transform
        except ImportError as exc:
            raise RuntimeError(
                "Direct GPU inference requires PyTorch and transformers. "
                "Install m1_earthdial/requirements-gpu.txt or use remote mode."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Direct EarthDial inference requires a CUDA-enabled GPU with "
                "sufficient VRAM. No CUDA GPU was detected; use remote mode."
            )

        print(f"[EarthDialAdapter] Loading checkpoint: {self.config.model_checkpoint}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint,
            trust_remote_code=True,
            use_fast=False,
        )

        try:
            from earthdial.model.internvl_chat import InternVLChatModel
            model_class = InternVLChatModel
        except ImportError:
            from transformers import AutoModel
            model_class = AutoModel

        # EarthDial is published with BF16 weights. Fall back to FP16 on CUDA
        # devices that do not implement BF16 inference rather than failing during
        # model construction.
        model_dtype = torch.bfloat16
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            model_dtype = torch.float16

        self._model = model_class.from_pretrained(
            self.config.model_checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        image_size = getattr(self._model.config, "force_image_size", None) or getattr(
            self._model.config.vision_config, "image_size", 448
        )
        self._transform = get_image_tensor_transform(input_size=image_size)
        self._model_dtype = model_dtype
        print(
            f"[EarthDialAdapter] Model and tokenizer successfully loaded on GPU "
            f"using {model_dtype}."
        )

    def _infer_direct(self, img: Image.Image, question: str, req: ImageVQARequest) -> str:
        """Execute inference directly on the configured local CUDA GPU."""
        self._ensure_direct_model_loaded()
        import torch

        pixel_values = self._transform(img).unsqueeze(0).cuda().to(self._model_dtype)
        generation_config = {
            "num_beams": req.num_beams,
            "max_new_tokens": req.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": req.temperature > 0.0,
            "temperature": req.temperature if req.temperature > 0.0 else 1.0,
        }

        with torch.no_grad():
            answer = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                verbose=False,
            )
        return str(answer).strip()

    def _infer_remote(self, image_path: str, question: str, req: ImageVQARequest) -> str:
        """Send a validated image/question to the remote EarthDial API."""
        import requests

        api_url = self.config.api_url.rstrip("/") + "/analyze"
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("ascii")

        payload = {
            "image_base64": image_b64,
            "question": req.question,
            "num_beams": req.num_beams,
            "temperature": req.temperature,
            "max_new_tokens": req.max_new_tokens,
        }

        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not connect to EarthDial GPU server at {self.config.api_url}. "
                "Ensure the remote server is running and EARTHDIAL_API_URL is correct."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Request to EarthDial GPU server timed out after "
                f"{self.config.timeout_seconds} seconds."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ConnectionError(f"EarthDial remote request failed: {exc}") from exc

        if response.status_code != 200:
            detail = response.text[:1000]
            raise RuntimeError(
                f"Remote EarthDial server returned HTTP {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Remote EarthDial server returned invalid JSON.") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Remote EarthDial server returned a non-object JSON response.")

        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise RuntimeError("Remote EarthDial server returned an empty answer.")

        return answer

    def _infer_mock(self, img_meta: dict, question: str) -> str:
        """Deterministic simulator used only for offline contract testing."""
        q_lower = question.lower()
        w, h = img_meta["width"], img_meta["height"]

        if any(term in q_lower for term in ["water", "river", "lake", "ocean", "sea"]):
            return (
                "Based on the visual features in this satellite imagery, dark absorption zones "
                "characteristic of open water bodies or river channels are discernible."
            )
        if any(term in q_lower for term in ["building", "urban", "city", "built-up", "structure"]):
            return (
                "The scene exhibits high-density geometric patterns and high reflectance rooftops "
                "indicative of built-up urban infrastructure and road networks."
            )
        if any(term in q_lower for term in ["land cover", "type", "vegetation", "forest", "field"]):
            return (
                "The dominant land cover consists of heterogeneous agricultural parcels, scattered "
                "tree canopies, and interconnected transport corridors."
            )
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
        """Validate input, run EarthDial, and return the stable M4 response contract."""
        start_time = time.perf_counter()

        if not question or not isinstance(question, str) or len(question.strip()) < 2:
            return self._failure_response(
                task=task,
                question=question or "",
                image_path=image_path,
                error="Invalid question: question must be a non-empty string with at least 2 characters.",
            )

        try:
            pil_img, img_metadata = load_and_preprocess_image(image_path)
        except ImagePreprocessingError as exc:
            return self._failure_response(
                task=task,
                question=question,
                image_path=image_path,
                error=f"Image preprocessing failed: {exc}",
            )

        try:
            req = ImageVQARequest(
                image_path=str(image_path),
                question=question.strip(),
                task=task,
                num_beams=num_beams if num_beams is not None else self.config.default_num_beams,
                temperature=temperature if temperature is not None else self.config.default_temperature,
                max_new_tokens=(
                    max_new_tokens
                    if max_new_tokens is not None
                    else self.config.default_max_new_tokens
                ),
            )
        except Exception as exc:
            return self._failure_response(
                task=task,
                question=question,
                image_path=image_path,
                error=f"Invalid inference parameters: {exc}",
            )

        backend_used = self._resolved_backend
        try:
            if backend_used == "direct_gpu":
                answer = self._infer_direct(pil_img, req.question, req)
            elif backend_used == "remote_colab":
                answer = self._infer_remote(str(image_path), req.question, req)
            elif backend_used == "mock":
                answer = self._infer_mock(img_metadata, req.question)
            else:
                raise ValueError(f"Unknown resolved backend: {backend_used}")

            answer = answer.strip()
            if not answer:
                raise RuntimeError("EarthDial returned an empty answer.")

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
                    "max_new_tokens": req.max_new_tokens,
                },
            )
            return ImageVQAResponse(
                task=req.task,
                question=req.question,
                answer=answer,
                model="EarthDial",
                confidence=None,
                evidence=evidence,
                success=True,
                error=None,
            )
        except Exception as exc:
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
                task=req.task,
                question=req.question,
                answer="",
                model="EarthDial",
                confidence=None,
                evidence=evidence,
                success=False,
                error=f"Inference execution failed on backend '{backend_used}': {exc}",
            )

    def _failure_response(
        self,
        task: str,
        question: str,
        image_path: str,
        error: str,
    ) -> ImageVQAResponse:
        """Create a consistent failure response without raising across the M4 boundary."""
        return ImageVQAResponse(
            task=task,
            question=question,
            answer="",
            model="EarthDial",
            confidence=None,
            success=False,
            error=error,
            evidence=EvidenceMetadata(
                image_path=str(image_path),
                model_name=self.config.model_checkpoint,
                backend=self._resolved_backend,
            ),
        )


_default_adapter: Optional[EarthDialAdapter] = None


def get_adapter(config: Optional[EarthDialConfig] = None) -> EarthDialAdapter:
    """Return the global adapter, or create an adapter for an explicit config."""
    global _default_adapter
    if config is not None:
        return EarthDialAdapter(config=config)
    if _default_adapter is None:
        _default_adapter = EarthDialAdapter()
    return _default_adapter


def set_adapter(adapter: Optional[EarthDialAdapter]):
    """Set or reset the process-wide adapter used by M4 integration."""
    global _default_adapter
    _default_adapter = adapter


def analyze_image(
    image_path: str,
    question: str,
    adapter: Optional[EarthDialAdapter] = None,
    config: Optional[EarthDialConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Public M4 entry point returning a JSON-serializable M1 dictionary."""
    target_adapter = adapter or (get_adapter(config) if config else get_adapter())
    response = target_adapter.analyze(image_path=image_path, question=question, **kwargs)
    return response.to_dict()
