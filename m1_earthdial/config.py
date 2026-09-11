"""
Configuration settings for SatQuery AI Module M1 (EarthDial).
Settings can be overridden using environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EarthDialConfig:
    """Configuration parameters for EarthDial execution."""

    # Execution backend: 'auto', 'remote', 'direct', or 'mock'.
    # 'auto' prefers a local CUDA GPU, then a reachable EarthDial remote server.
    # If neither is available, auto mode uses the deterministic mock backend
    # so local development and CI remain usable without a GPU.
    backend: str = os.getenv("EARTHDIAL_BACKEND", "auto").lower()

    # EarthDial remote server URL (Google Colab FastAPI / ngrok endpoint).
    # Do NOT default this to the M5 SatQuery API on port 8000: M5 exposes its
    # own /health endpoint and would otherwise be mistaken for EarthDial.
    api_url: str = os.getenv("EARTHDIAL_API_URL", "http://localhost:7860")

    # Request timeout in seconds for remote inference
    timeout_seconds: int = int(os.getenv("EARTHDIAL_TIMEOUT", "90"))

    # Hugging Face model checkpoint ID
    model_checkpoint: str = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB")

    # Local checkpoint directory cache
    checkpoint_dir: Path = Path(os.getenv("EARTHDIAL_CACHE_DIR", "./checkpoints"))

    # Output directory for saving structured JSON responses
    output_dir: Path = Path(os.getenv("EARTHDIAL_OUTPUT_DIR", "./m1_earthdial/outputs"))

    # Low-VRAM local inference controls.
    load_in_4bit: bool = os.getenv(
        "EARTHDIAL_LOAD_IN_4BIT", "true"
    ).lower() in {"1", "true", "yes", "on"}
    gpu_memory_gib: float = float(os.getenv("EARTHDIAL_GPU_MEMORY_GIB", "3.0"))
    cpu_memory_gib: float = float(os.getenv("EARTHDIAL_CPU_MEMORY_GIB", "16.0"))
    bnb_quant_type: str = os.getenv("EARTHDIAL_BNB_QUANT_TYPE", "nf4")
    bnb_double_quant: bool = os.getenv(
        "EARTHDIAL_BNB_DOUBLE_QUANT", "true"
    ).lower() in {"1", "true", "yes", "on"}

    # Low-VRAM generation defaults validated with EarthDial 4B on RTX 2050.
    default_num_beams: int = int(os.getenv("EARTHDIAL_NUM_BEAMS", "1"))
    default_temperature: float = float(os.getenv("EARTHDIAL_TEMPERATURE", "0.0"))
    default_max_new_tokens: int = int(os.getenv("EARTHDIAL_MAX_NEW_TOKENS", "64"))


# Default configuration instance
default_config = EarthDialConfig()
