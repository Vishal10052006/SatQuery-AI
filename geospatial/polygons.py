"""
Polygon extraction and conversion module for SatQuery-AI M5.
Converts change masks into Shapely polygons reprojected to EPSG:4326.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union
from affine import Affine
import numpy as np
from pyproj import CRS, Transformer
import rasterio
import rasterio.features
import shapely.geometry
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid


def _ensure_affine(transform: Union[Affine, Sequence[float]]) -> Affine:
    if isinstance(transform, Affine):
        return transform
    if len(transform) == 9:
        return Affine(
            transform[0], transform[1], transform[2],
            transform[3], transform[4], transform[5]
        )
    elif len(transform) == 6:
        return Affine(*transform)
    raise ValueError(f"Invalid affine transform format: {transform}")


def load_mask(mask_input: Union[np.ndarray, str, Path]) -> np.ndarray:
    """
    Load a change mask from numpy array or file path.

    Supports:
        - numpy ndarray (2D or 3D)
        - .npy file
        - GeoTIFF / PNG / raster image file
    """
    if isinstance(mask_input, np.ndarray):
        arr = mask_input
    else:
        path = Path(mask_input)
        if not path.exists():
            raise FileNotFoundError(f"Change mask file not found: {path}")

        if path.suffix.lower() == ".npy":
            arr = np.load(path)
        else:
            with rasterio.open(path) as src:
                arr = src.read(1)

    # If 3D, take first band or squeeze
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] < arr.shape[2] else arr[:, :, 0]

    # Convert to 2D binary uint8 (0 or 1)
    binary_mask = (arr > 0).astype(np.uint8)
    return binary_mask


def mask_to_polygons(
    mask: Union[np.ndarray, str, Path],
    transform: Union[Affine, Sequence[float]],
    crs: Union[str, CRS, Any] = None,
    to_crs: str = "EPSG:4326",
    min_pixel_area: int = 1,
    connectivity: int = 8,
) -> List[Union[Polygon, MultiPolygon]]:
    """
    Extract polygons from a binary change mask and reproject to target geographic CRS.

    Args:
        mask: 2D binary numpy array or path to mask file (.npy or raster).
        transform: Affine transform matrix of the raster.
        crs: Native coordinate reference system of the raster.
        to_crs: Destination CRS (default: EPSG:4326).
        min_pixel_area: Filter out shapes with pixel count below this threshold.
        connectivity: Pixel connectivity (4 or 8).

    Returns:
        List of Shapely Polygon or MultiPolygon objects in to_crs (EPSG:4326).
    """
    binary_mask = load_mask(mask)
    aff = _ensure_affine(transform)

    # Check if there are any changed pixels
    if not np.any(binary_mask):
        return []

    # Prepare coordinate transformer if CRS conversion is needed
    transformer: Optional[Transformer] = None
    if crs is not None:
        source_crs = CRS.from_user_input(crs)
        target_crs = CRS.from_user_input(to_crs)
        if source_crs != target_crs:
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    polygons: List[Union[Polygon, MultiPolygon]] = []

    # rasterio.features.shapes yields (geojson_geom, val)
    # where mask specifies pixels to vectorize
    shapes_gen = rasterio.features.shapes(
        binary_mask,
        mask=(binary_mask == 1),
        transform=aff,
        connectivity=connectivity,
    )

    for geom_dict, val in shapes_gen:
        if val != 1:
            continue

        geom = shapely.geometry.shape(geom_dict)

        if geom.is_empty:
            continue

        # Ensure geometry is valid
        if not geom.is_valid:
            geom = make_valid(geom)

        # Reproject geometry if transformer is set
        if transformer is not None:
            geom = shapely_transform(transformer.transform, geom)
            if not geom.is_valid:
                geom = make_valid(geom)

        if isinstance(geom, (Polygon, MultiPolygon)):
            polygons.append(geom)
        elif isinstance(geom, GeometryCollection):
            # If GeometryCollection, collect all polygon components
            for sub_geom in geom.geoms:
                if isinstance(sub_geom, (Polygon, MultiPolygon)):
                    polygons.append(sub_geom)

    return polygons


def bbox_to_polygon(bbox_geo: Sequence[float]) -> Polygon:
    """
    Convert geographic bounding box [min_lon, min_lat, max_lon, max_lat]
    into a Shapely Polygon.
    """
    if len(bbox_geo) != 4:
        raise ValueError("bbox_geo must have 4 elements: [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = bbox_geo
    return box(min_lon, min_lat, max_lon, max_lat)
