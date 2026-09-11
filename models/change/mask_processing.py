"""Change mask post-processing and connected region extraction.

Converts raw difference maps into clean binary masks and structured change
regions. All connected-component statistics are computed only from pixels
marked valid by the caller; invalid pixels must never become change evidence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChangeRegion:
    """A detected connected change region with spatial and statistical properties."""

    region_id: int
    pixel_count: int
    bbox_pixel: dict[str, int]
    centroid_pixel: dict[str, float]
    polygon_pixel: list[list[int]]
    confidence: float
    bbox_geo: dict[str, tuple[float, float]] | None = None
    centroid_geo: dict[str, float] | None = None
    polygon_geo: list[tuple[float, float]] | None = None
    area_sq_m: float | None = None
    target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "pixel_count": self.pixel_count,
            "bbox_pixel": self.bbox_pixel,
            "centroid_pixel": self.centroid_pixel,
            "polygon_pixel": self.polygon_pixel,
            "confidence": round(self.confidence, 4),
            "bbox_geo": self.bbox_geo,
            "centroid_geo": self.centroid_geo,
            "polygon_geo": self.polygon_geo,
            "area_sq_m": self.area_sq_m,
            "target": self.target,
        }


def _validate_inputs(raw_mask: np.ndarray, diff_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize component inputs and reject malformed/non-finite evidence."""
    mask = np.asarray(raw_mask, dtype=bool)
    diff = np.asarray(diff_map, dtype=np.float32)
    if mask.ndim != 2 or diff.ndim != 2 or mask.shape != diff.shape:
        raise ValueError("raw_mask and diff_map must be 2-D arrays with identical shapes")
    # Non-finite differences cannot be evidence. This also protects region scores.
    mask &= np.isfinite(diff)
    diff = np.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=0.0)
    return mask, diff


def _region_confidence(local_diff: np.ndarray) -> float:
    """Return an evidence-strength score, not a calibrated probability."""
    if local_diff.size == 0:
        return 0.0
    # Difference is expected to be normalized to [0, 1]. Keep this bounded and
    # explicitly interpret it as mean temporal signal strength.
    return float(np.clip(np.mean(local_diff), 0.0, 1.0))


def _extract_components_scipy(mask: np.ndarray, diff_map: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[ChangeRegion]]:
    """Extract connected components using scipy.ndimage (8-connectivity)."""
    from scipy.ndimage import find_objects, label  # type: ignore

    structure = np.ones((3, 3), dtype=int)
    labeled_mask, num_features = label(mask, structure=structure)
    slices = find_objects(labeled_mask)

    clean_mask = np.zeros_like(mask, dtype=bool)
    regions: list[ChangeRegion] = []

    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        region_label = idx + 1
        component = labeled_mask[slc] == region_label
        pixel_count = int(component.sum())
        if pixel_count < min_pixels:
            continue

        clean_mask[slc] |= component
        r_slice, c_slice = slc
        ymin, ymax = r_slice.start, r_slice.stop - 1
        xmin, xmax = c_slice.start, c_slice.stop - 1
        coords = np.argwhere(component)
        cy = float(coords[:, 0].mean() + ymin)
        cx = float(coords[:, 1].mean() + xmin)
        conf = _region_confidence(diff_map[slc][component])

        polygon = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]
        regions.append(ChangeRegion(
            region_id=len(regions) + 1,
            pixel_count=pixel_count,
            bbox_pixel={"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            centroid_pixel={"x": round(cx, 2), "y": round(cy, 2)},
            polygon_pixel=polygon,
            confidence=conf,
        ))

    regions.sort(key=lambda r: (-r.pixel_count, r.bbox_pixel["ymin"], r.bbox_pixel["xmin"]))
    for idx, region in enumerate(regions, start=1):
        region.region_id = idx
    return clean_mask, regions


def _extract_components_bfs(mask: np.ndarray, diff_map: np.ndarray, min_pixels: int) -> tuple[np.ndarray, list[ChangeRegion]]:
    """Deterministic fallback connected-components extraction using 8-connectivity."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clean_mask = np.zeros_like(mask, dtype=bool)
    regions: list[ChangeRegion] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            pixels: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                pixels.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

            if len(pixels) < min_pixels:
                continue

            for r, c in pixels:
                clean_mask[r, c] = True
            rows = np.fromiter((p[0] for p in pixels), dtype=np.int64)
            cols = np.fromiter((p[1] for p in pixels), dtype=np.int64)
            xmin, xmax = int(cols.min()), int(cols.max())
            ymin, ymax = int(rows.min()), int(rows.max())
            conf = _region_confidence(np.asarray([diff_map[r, c] for r, c in pixels], dtype=np.float32))
            regions.append(ChangeRegion(
                region_id=len(regions) + 1,
                pixel_count=len(pixels),
                bbox_pixel={"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                centroid_pixel={"x": round(float(cols.mean()), 2), "y": round(float(rows.mean()), 2)},
                polygon_pixel=[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]],
                confidence=conf,
            ))

    regions.sort(key=lambda r: (-r.pixel_count, r.bbox_pixel["ymin"], r.bbox_pixel["xmin"]))
    for idx, region in enumerate(regions, start=1):
        region.region_id = idx
    return clean_mask, regions


def extract_change_regions(raw_mask: np.ndarray, diff_map: np.ndarray, min_pixels: int = 8) -> tuple[np.ndarray, list[ChangeRegion]]:
    """Extract clean change mask and connected ChangeRegion objects."""
    mask, diff = _validate_inputs(raw_mask, diff_map)
    min_pix = max(1, int(min_pixels))
    try:
        import scipy.ndimage  # type: ignore
        return _extract_components_scipy(mask, diff, min_pix)
    except ImportError:
        return _extract_components_bfs(mask, diff, min_pix)
