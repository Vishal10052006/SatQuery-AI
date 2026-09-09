"""Visualization artifacts for M2 remote-sensing change analysis.

Generates:
1. BEFORE | AFTER | DETECTED CHANGE (or TARGET CHANGE) side-by-side composite
2. Change overlay on the after-image with optional bounding box annotations
3. Clean binary change mask PNG
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .mask_processing import ChangeRegion


def create_change_artifacts(
    before_path: str,
    after_path: str,
    output_dir: str | Path,
    *,
    threshold: float = 0.15,
) -> list[str]:
    """Create a side-by-side before/after/change visualization.

    Maintains 100% backward compatibility with existing tests and M4 contracts.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB").resize(before.size, Image.Resampling.BILINEAR)
    before_gray = np.asarray(before.convert("L"), dtype=np.float32) / 255.0
    after_gray = np.asarray(after.convert("L"), dtype=np.float32) / 255.0
    mask = np.abs(after_gray - before_gray) >= float(threshold)

    overlay = after.copy()
    pixels = np.asarray(overlay).copy()
    pixels[mask] = np.array([255, 40, 40], dtype=np.uint8)
    overlay = Image.fromarray(pixels, mode="RGB")

    width, height = before.size
    label_height = 34
    canvas = Image.new("RGB", (width * 3, height + label_height), "white")
    canvas.paste(before, (0, label_height))
    canvas.paste(after, (width, label_height))
    canvas.paste(overlay, (width * 2, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "BEFORE", fill="black")
    draw.text((width + 10, 10), "AFTER", fill="black")
    draw.text((width * 2 + 10, 10), "DETECTED CHANGE", fill="black")

    artifact_name = f"change-{uuid4().hex[:12]}.png"
    artifact_path = output / artifact_name
    canvas.save(artifact_path, format="PNG", optimize=True)
    return [artifact_name]


def generate_m2_visualizations(
    before_img: Image.Image,
    after_img: Image.Image,
    mask: np.ndarray,
    regions: Sequence[ChangeRegion],
    output_dir: str | Path,
    target: str | None = None,
) -> dict[str, str]:
    """Generate rich M2 visualization suite: composite, overlay with bboxes, and mask.

    Returns dict mapping artifact type to relative filename in output_dir.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Standardize size
    w, h = before_img.size
    after_resampled = after_img.resize((w, h), Image.Resampling.BILINEAR)

    # 1. Binary Mask image
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_pil = Image.fromarray(mask_uint8)
    mask_filename = f"mask-{uuid4().hex[:12]}.png"
    mask_pil.save(output / mask_filename, format="PNG")

    # 2. Overlay on after image with red highlight
    pixels = np.asarray(after_resampled.convert("RGB")).copy()
    if mask.shape == (h, w):
        pixels[mask] = np.array([255, 45, 45], dtype=np.uint8)
    overlay_pil = Image.fromarray(pixels)

    # Draw bounding boxes on overlay
    draw_overlay = ImageDraw.Draw(overlay_pil)
    for r in regions:
        box = r.bbox_pixel
        draw_overlay.rectangle(
            [box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
            outline=(50, 255, 80),
            width=2,
        )
        draw_overlay.text(
            (box["xmin"] + 2, max(0, box["ymin"] - 12)),
            f"R{r.region_id}",
            fill=(50, 255, 80),
        )

    overlay_filename = f"overlay-{uuid4().hex[:12]}.png"
    overlay_pil.save(output / overlay_filename, format="PNG", optimize=True)

    # 3. Side-by-side composite
    label_height = 34
    canvas = Image.new("RGB", (w * 3, h + label_height), "white")
    canvas.paste(before_img, (0, label_height))
    canvas.paste(after_resampled, (w, label_height))
    canvas.paste(overlay_pil, (w * 2, label_height))

    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((10, 10), "BEFORE", fill="black")
    draw_canvas.text((w + 10, 10), "AFTER", fill="black")
    change_label = f"TARGET: {target.upper()}" if target else "DETECTED CHANGE"
    draw_canvas.text((w * 2 + 10, 10), change_label, fill="black")

    composite_filename = f"change-{uuid4().hex[:12]}.png"
    canvas.save(output / composite_filename, format="PNG", optimize=True)

    return {
        "composite": composite_filename,
        "overlay": overlay_filename,
        "mask": mask_filename,
    }
