"""
Configuration settings for SatQuery AI Module M1 (EarthDial).
Settings can be overridden using environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EarthDialConfig:
    """Configuration parameters for EarthDial execution."""
    
    # Execution backend: 'auto', 'remote', 'direct', or 'mock'
    # - 'auto': Checks if local CUDA GPU is present; if yes uses 'direct', else checks for remote URL; falls back to mock if offline.
    # - 'remote': Sends requests to Colab GPU server via EARTHDIAL_API_URL
    # - 'direct': Loads weights directly into local GPU memory
    # - 'mock': Local offline simulator for agent integration tests
    backend: str = os.getenv("EARTHDIAL_BACKEND", "auto").lower()

    # Remote server URL (Google Colab FastAPI / ngrok endpoint)
    api_url: str = os.getenv("EARTHDIAL_API_URL", "http://localhost:8000")

    # Request timeout in seconds for remote inference
    timeout_seconds: int = int(os.getenv("EARTHDIAL_TIMEOUT", "90"))

    # Hugging Face model checkpoint ID
    model_checkpoint: str = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB")

    # Local checkpoint directory cache
    checkpoint_dir: Path = Path(os.getenv("EARTHDIAL_CACHE_DIR", "./checkpoints"))

    # Output directory for saving structured JSON responses
    output_dir: Path = Path(os.getenv("EARTHDIAL_OUTPUT_DIR", "./m1_earthdial/outputs"))

    # Inference generation defaults
    default_num_beams: int = 5
    default_temperature: float = 0.0
    default_max_new_tokens: int = 128


# Default configuration instance
default_config = EarthDialConfig()

