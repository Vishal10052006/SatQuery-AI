"""
Integration interface for SatQuery-AI Module 5 (M5).
Provides clean entrypoints for upstream AI modules (M2: Change Detection,
M3: Object Detection / Classification) to invoke the geospatial processing engine.
"""

from pathlib import Path
from typing import Any, Dict, Union

from geospatial.pipeline import run_geospatial_pipeline
from geospatial.schema import M2M3Payload, EvidenceOutput


def process_m2_m3_result(
    payload: Union[Dict[str, Any], M2M3Payload, str, Path],
    output_dir: Union[str, Path] = "output",
) -> Dict[str, Any]:
    """
    Process an upstream M2/M3 result payload through the M5 geospatial pipeline.

    Expected Payload Format:
    {
        "target": "new building",
        "confidence": 0.91,
        "change_detected": True,
        "reference_image": "path/to/reference.tif",
        "change_mask": "path/to/change_mask.npy",
        "bounding_boxes": [
            [500, 300, 650, 450]
        ]
    }

    Args:
        payload: A dictionary matching the standardized M2/M3 schema, an M2M3Payload
                 dataclass instance, or a path to a JSON file.
        output_dir: Configurable destination directory for output artifacts
                    (GeoJSON, interactive map, evidence.json). Defaults to "output".

    Returns:
        Structured evidence dictionary containing verified paths, coordinates,
        areas, and geometries.

    Raises:
        ValueError: If reference image path is missing or invalid.
        FileNotFoundError: If referenced image or mask files do not exist.
    """
    # 1. Parse and validate input payload
    if isinstance(payload, M2M3Payload):
        parsed_payload = payload
    elif isinstance(payload, dict):
        parsed_payload = M2M3Payload.from_dict(payload)
    elif isinstance(payload, (str, Path)):
        parsed_payload = M2M3Payload.from_json(payload)
    else:
        raise TypeError(
            f"Unsupported payload type '{type(payload).__name__}'. "
            "Expected dict, M2M3Payload, or path to JSON file."
        )

    if not parsed_payload.reference_image:
        raise ValueError(
            "Payload must provide 'reference_image' pointing to the reference GeoTIFF."
        )

    ref_path = Path(parsed_payload.reference_image)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference GeoTIFF not found at: {ref_path}"
        )

    # 2. Delegate execution to existing, verified M5 pipeline
    evidence_dict = run_geospatial_pipeline(
        reference_geotiff=str(ref_path),
        change_mask=parsed_payload.change_mask,
        bounding_boxes=parsed_payload.bounding_boxes,
        target=parsed_payload.target,
        confidence=parsed_payload.confidence,
        output_dir=output_dir,
    )

    return evidence_dict
