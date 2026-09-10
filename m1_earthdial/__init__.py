"""
SatQuery AI - Module M1: Earth Observation VLM / Visual Question Answering using EarthDial
"""

from .schemas import ImageVQARequest, ImageVQAResponse, EvidenceMetadata
from .earthdial_adapter import EarthDialAdapter, analyze_image

__all__ = [
    "ImageVQARequest",
    "ImageVQAResponse",
    "EvidenceMetadata",
    "EarthDialAdapter",
    "analyze_image",
]

