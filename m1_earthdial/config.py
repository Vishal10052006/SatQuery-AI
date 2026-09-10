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
    
    # Execution backend: 'auto', 'remote', 'direct', or 'mock'
    # - 'auto': Checks hardware; if local CUDA GPU has >=10GB VRAM, uses 'direct';
    #           if remote URL responds, uses 'remote'; falls back to 'mock' for local dev.
    # - 'remote': Sends requests to Colab GPU server via EARTHDIAL_API_URL
    # - 'direct': Loads weights directly into local GPU memory (requires >=10GB VRAM)
    # - 'mock': Local offline simulator for agent integration tests
    backend: str = os.getenv("EARTHDIAL_BACKEND", "auto").lower().strip()

    # Remote server URL (Google Colab FastAPI / Cloudflare tunnel endpoint)
    # Defaults to empty string to require explicit setting when backend='remote'
    api_url: str = os.getenv("EARTHDIAL_API_URL", "").strip()

    # Request timeout in seconds for remote inference
    timeout_seconds: int = int(os.getenv("EARTHDIAL_TIMEOUT", "90"))

    # Hugging Face model checkpoint ID
    model_checkpoint: str = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB").strip()

    # Local checkpoint directory cache
    checkpoint_dir: Path = Path(os.getenv("EARTHDIAL_CACHE_DIR", "./checkpoints"))

    # Output directory for saving structured JSON responses
    output_dir: Path = Path(os.getenv("EARTHDIAL_OUTPUT_DIR", "./m1_earthdial/outputs"))

    # Minimum VRAM in GB required for local direct inference in BF16
    min_vram_gb: float = 10.0

    # Inference generation defaults
    default_num_beams: int = 5
    default_temperature: float = 0.0
    default_max_new_tokens: int = 128


# Default configuration instance
default_config = EarthDialConfig()
