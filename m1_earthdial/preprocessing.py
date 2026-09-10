"""
Satellite image preprocessing and validation utilities for SatQuery AI M1.
Handles file validation, format conversions, dimension extraction, and tensor transforms.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class ImagePreprocessingError(Exception):
    """Raised when an input satellite image fails validation or preprocessing."""
    pass


def validate_image_path(image_path: str) -> Path:
    """
    Validates that the provided path exists, is a file, and has a non-zero size.
    
    Args:
        image_path: Path to the satellite image file.
        
    Returns:
        Path object pointing to the validated image.
        
    Raises:
        ImagePreprocessingError: If file does not exist, is not a file, or is empty.
    """
    if not image_path or not isinstance(image_path, str):
        raise ImagePreprocessingError("Image path must be a non-empty string.")

    path = Path(image_path).expanduser().resolve()

    if not path.exists():
        raise ImagePreprocessingError(f"Image file not found: {image_path}")

    if not path.is_file():
        raise ImagePreprocessingError(f"Specified path is not a file: {image_path}")

    if path.stat().st_size == 0:
        raise ImagePreprocessingError(f"Image file is empty (0 bytes): {image_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImagePreprocessingError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported extensions are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return path


def load_and_preprocess_image(image_path: str) -> Tuple[Image.Image, dict]:
    """
    Loads a satellite image, verifies its integrity, converts it to 3-channel RGB,
    and extracts metadata for evidence recording.
    Ensures underlying file handles are cleanly closed.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Tuple of (PIL.Image in RGB mode, metadata dictionary).
        
    Raises:
        ImagePreprocessingError: If image cannot be read or is corrupted.
    """
    path = validate_image_path(image_path)

    # First pass: verify image structure integrity
    try:
        with Image.open(path) as test_img:
            test_img.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise ImagePreprocessingError(f"Corrupt or unreadable image file '{path.name}': {e}") from e

    # Second pass: read pixels and convert to RGB
    try:
        with Image.open(path) as raw_img:
            original_format = raw_img.format or path.suffix.replace(".", "").upper()
            width, height = raw_img.size
            img = raw_img.convert("RGB")

        metadata = {
            "image_path": str(path),
            "original_format": original_format,
            "width": width,
            "height": height,
            "mode": "RGB",
        }
        return img, metadata

    except Exception as e:
        raise ImagePreprocessingError(f"Failed to load image '{path.name}': {e}") from e


def get_image_tensor_transform(input_size: int = 448):
    """
    Returns torchvision transforms matching EarthDial's official build_transform.
    Only imported when running inside a GPU environment where PyTorch & torchvision are installed.
    """
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform
    except ImportError as e:
        raise ImagePreprocessingError(
            "PyTorch and torchvision are required for tensor transformation. "
            "Please ensure requirements-gpu.txt is installed in your GPU environment."
        ) from e

