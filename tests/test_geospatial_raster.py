"""Tests for the optional geospatial raster layer."""
from pathlib import Path

from PIL import Image

from geospatial.raster import bbox_pixel_to_geo, inspect_raster


def test_inspect_plain_image_has_no_fabricated_georeference(tmp_path: Path):
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (20, 10)).save(image_path)

    metadata = inspect_raster(str(image_path))

    assert metadata["width"] == 20
    assert metadata["height"] == 10
    assert metadata["georeferenced"] is False
    assert metadata["crs"] is None
    assert metadata["transform"] is None


def test_affine_bbox_conversion():
    # x_geo = x + 100, y_geo = -y + 200
    transform = (1.0, 0.0, 100.0, 0.0, -1.0, 200.0)
    result = bbox_pixel_to_geo(
        {"xmin": 10, "ymin": 20, "xmax": 19, "ymax": 29},
        transform,
    )

    assert result["top_left"] == (110.0, 180.0)
    assert result["top_right"] == (120.0, 180.0)
    assert result["bottom_right"] == (120.0, 170.0)
    assert result["bottom_left"] == (110.0, 170.0)
