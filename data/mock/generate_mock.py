"""
Synthetic mock data generator for SatQuery-AI M5 testing and validation.
Creates a valid sample GeoTIFF, change mask, and upstream M2 detection JSON.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def create_sample_mock_data(output_dir: str | Path | None = None):
    if output_dir is None:
        out_path = Path(__file__).resolve().parent
    else:
        out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    geotiff_path = out_path / "sample.tif"
    mask_npy_path = out_path / "change_mask.npy"
    m2_result_path = out_path / "m2_result.json"

    # 1. Create realistic multi-band GeoTIFF
    # Dimensions: 256 x 256, 3 bands (RGB)
    # CRS: UTM Zone 43N (EPSG:32643) covering parts of India / Central Asia
    # Resolution: 10 meters per pixel
    width, height = 256, 256
    west = 700000.0   # UTM Easting in meters
    north = 3100000.0  # UTM Northing in meters
    x_res, y_res = 10.0, 10.0
    transform = from_origin(west, north, x_res, y_res)
    crs = "EPSG:32643"

    # Generate synthetic satellite pixel values (e.g. forest / land / water textures)
    np.random.seed(42)
    band1 = np.random.randint(40, 120, size=(height, width), dtype=np.uint8)
    band2 = np.random.randint(60, 160, size=(height, width), dtype=np.uint8)
    band3 = np.random.randint(30, 90, size=(height, width), dtype=np.uint8)

    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(band1, 1)
        dst.write(band2, 2)
        dst.write(band3, 3)

    # 2. Create synthetic change mask
    # A 256x256 binary mask with 2 distinct change clusters
    mask = np.zeros((height, width), dtype=np.uint8)
    # Cluster 1: e.g. Deforestation patch (col: 40-90, row: 50-100)
    mask[50:100, 40:90] = 1
    # Cluster 2: e.g. New road / clearing (col: 140-200, row: 160-185)
    mask[160:185, 140:200] = 1

    np.save(mask_npy_path, mask)

    # 3. Create mock M2 module output payload
    m2_payload = {
        "module": "M2_Change_Detection",
        "reference_geotiff": str(geotiff_path.resolve().as_posix()),
        "change_mask_path": str(mask_npy_path.resolve().as_posix()),
        "target": "deforestation",
        "confidence": 0.94,
        "change_detected": True,
        "bounding_boxes_pixel": [
            [40, 50, 90, 100],     # [xmin, ymin, xmax, ymax]
            [140, 160, 200, 185],  # [xmin, ymin, xmax, ymax]
        ],
        "metadata": {
            "satellite": "Sentinel-2",
            "resolution_meters": 10.0,
            "bands": ["B04", "B03", "B02"],
        },
    }

    with open(m2_result_path, "w", encoding="utf-8") as f:
        json.dump(m2_payload, f, indent=2)

    print(f"Sample GeoTIFF created at: {geotiff_path}")
    print(f"Sample change mask created at: {mask_npy_path}")
    print(f"Mock M2 result saved at: {m2_result_path}")
    return geotiff_path, mask_npy_path, m2_result_path


if __name__ == "__main__":
    create_sample_mock_data()
