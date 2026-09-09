"""Change mask post-processing and connected region extraction.

Converts raw difference or probability maps into clean binary masks and
structured, attribute-rich change regions.
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
    bbox_pixel: dict[str, int]               # {'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int}
    centroid_pixel: dict[str, float]          # {'x': float, 'y': float}
    polygon_pixel: list[list[int]]            # list of [x, y] polygon points
    confidence: float                         # region quality / mean difference intensity
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


def _extract_components_scipy(
    mask: np.ndarray,
    diff_map: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, list[ChangeRegion]]:
    """Extract connected components using scipy.ndimage for high efficiency."""
    from scipy.ndimage import label, find_objects  # type: ignore

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_mask, num_features = label(mask, structure=structure)
    slices = find_objects(labeled_mask)

    clean_mask = np.zeros_like(mask, dtype=bool)
    regions: list[ChangeRegion] = []
    region_id_counter = 1

    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        region_label = idx + 1
        component = (labeled_mask[slc] == region_label)
        pixel_count = int(np.sum(component))

        if pixel_count < min_pixels:
            continue

        clean_mask[slc] |= component

        # Bounding box
        r_slice, c_slice = slc
        ymin, ymax = r_slice.start, r_slice.stop - 1
        xmin, xmax = c_slice.start, c_slice.stop - 1

        # Centroid
        comp_coords = np.argwhere(component)
        cy = float(np.mean(comp_coords[:, 0]) + ymin)
        cx = float(np.mean(comp_coords[:, 1]) + xmin)

        # Region confidence from local difference intensity
        local_diff = diff_map[slc][component]
        conf = float(np.mean(local_diff)) if local_diff.size > 0 else 0.5
        conf = min(1.0, max(0.1, conf))

        # Approximate polygon (boundary coordinates: corners and midpoints)
        polygon = [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]

        regions.append(
            ChangeRegion(
                region_id=region_id_counter,
                pixel_count=pixel_count,
                bbox_pixel={"xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)},
                centroid_pixel={"x": round(cx, 2), "y": round(cy, 2)},
                polygon_pixel=polygon,
                confidence=conf,
            )
        )
        region_id_counter += 1

    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return clean_mask, regions


def _extract_components_bfs(
    mask: np.ndarray,
    diff_map: np.ndarray,
    min_pixels: int,
) -> tuple[np.ndarray, list[ChangeRegion]]:
    """Fallback connected-components extraction using breadth-first search."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clean_mask = np.zeros_like(mask, dtype=bool)
    regions: list[ChangeRegion] = []
    region_id_counter = 1

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
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            if len(pixels) < min_pixels:
                continue

            for r, c in pixels:
                clean_mask[r, c] = True

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            xmin, xmax = min(cols), max(cols)
            ymin, ymax = min(rows), max(rows)
            cx = float(np.mean(cols))
            cy = float(np.mean(rows))

            diff_vals = [diff_map[r, c] for r, c in pixels]
            conf = float(np.mean(diff_vals)) if diff_vals else 0.5
            conf = min(1.0, max(0.1, conf))

            polygon = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin],
            ]

            regions.append(
                ChangeRegion(
                    region_id=region_id_counter,
                    pixel_count=len(pixels),
                    bbox_pixel={"xmin": int(xmin), "ymin": int(ymin), "xmax": int(xmax), "ymax": int(ymax)},
                    centroid_pixel={"x": round(cx, 2), "y": round(cy, 2)},
                    polygon_pixel=polygon,
                    confidence=conf,
                )
            )
            region_id_counter += 1

    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return clean_mask, regions


def extract_change_regions(
    raw_mask: np.ndarray,
    diff_map: np.ndarray,
    min_pixels: int = 8,
) -> tuple[np.ndarray, list[ChangeRegion]]:
    """Extract clean change mask and connected ChangeRegion objects."""
    min_pix = max(1, int(min_pixels))
    try:
        import scipy.ndimage  # type: ignore
        return _extract_components_scipy(raw_mask, diff_map, min_pix)
    except ImportError:
        return _extract_components_bfs(raw_mask, diff_map, min_pix)
