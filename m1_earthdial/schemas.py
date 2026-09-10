"""
Pydantic schemas and data contracts for SatQuery AI Module M1 (EarthDial).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class EvidenceMetadata(BaseModel):
    """
    Evidence metadata associated with the VLM inference result.
    Captures verifiable image attributes, inference timing, and execution backend.
    Note: Geographic coordinates and segmentation masks are NOT claimed here as
    EarthDial VQA generates textual responses without spatial coordinates.
    """
    image_path: str = Field(..., description="Path or URI of the processed satellite image")
    image_format: Optional[str] = Field(default=None, description="Image file format (e.g. JPEG, PNG, TIFF)")
    image_size: Optional[Tuple[int, int]] = Field(default=None, description="Image dimensions as (width, height) in pixels")
    model_name: str = Field(default="EarthDial_4B_RGB", description="Name of the underlying VLM checkpoint")
    backend: str = Field(default="direct", description="Execution backend: 'remote_colab', 'direct_gpu', or 'mock'")
    inference_time_seconds: Optional[float] = Field(default=None, description="Time taken to compute inference")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO 8601 UTC timestamp")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional model execution metadata")


class ImageVQARequest(BaseModel):
    """Request payload for satellite image visual question answering."""
    image_path: str = Field(..., description="Absolute or relative path to satellite image")
    question: str = Field(..., min_length=2, description="Natural language question about the satellite image")
    task: str = Field(default="image_vqa", description="Task identifier: 'image_vqa', 'scene_description'")
    num_beams: int = Field(default=5, ge=1, le=10, description="Beam search width for generation")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (0.0 for greedy/deterministic)")
    max_new_tokens: int = Field(default=128, ge=1, le=512, description="Maximum new tokens to generate")


class ImageVQAResponse(BaseModel):
    """
    Standardized, structured result contract returned by M1 to M4 Agent.
    
    CONFIDENCE HANDLING NOTICE:
    EarthDial generates answers autoregressively via open-ended text generation.
    Because raw token likelihoods do not represent calibrated semantic certainty,
    'confidence' is strictly set to None (null in JSON) rather than fabricated.
    """
    task: str = Field(default="image_vqa", description="Module task type")
    question: str = Field(..., description="Original input question")
    answer: str = Field(..., description="EarthDial generated answer / scene description")
    model: str = Field(default="EarthDial", description="Module / model identifier")
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score (null for EarthDial as autoregressive VLM does not emit calibrated confidence)"
    )
    evidence: EvidenceMetadata = Field(..., description="Verifiable execution and image metadata")
    success: bool = Field(default=True, description="Whether the analysis succeeded")
    error: Optional[str] = Field(default=None, description="Error message if inference failed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard Python dictionary matching the required JSON format."""
        return self.model_dump()

