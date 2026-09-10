"""PyTorch Dataset and DataLoaders for real multimodal Optical + SAR satellite rasters."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from modules.optical_sar.optical.loader import load_optical
from modules.optical_sar.optical.normalization import normalize_optical
from modules.optical_sar.registration.reprojection import reproject_to_reference
from modules.optical_sar.sar.calibration import calibrate_sar
from modules.optical_sar.sar.loader import load_sar
from modules.optical_sar.sar.normalization import normalize_sar
from modules.optical_sar.sar.speckle import apply_speckle_filter

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Vegetation", "Bare Soil", "Water"]
SCL_TO_CLASS: Dict[int, int] = {
    4: 0,  # Vegetation
    5: 1,  # Bare soil
    6: 2,  # Water
}


class RealOpticalSARDataset(Dataset):
    """PyTorch Dataset extracting real paired Optical and SAR patches with ground-truth labels."""

    def __init__(
        self,
        optical_patches: List[np.ndarray],
        sar_patches: List[np.ndarray],
        labels: List[int],
    ):
        self.optical_patches = optical_patches
        self.sar_patches = sar_patches
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        opt_arr = torch.from_numpy(self.optical_patches[idx]).float()
        sar_arr = torch.from_numpy(self.sar_patches[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return opt_arr, sar_arr, label


def prepare_real_multimodal_patches(
    optical_path: Union[str, Path],
    sar_path: Union[str, Path],
    scl_path: Union[str, Path],
    patch_size: int = 32,
    stride: int = 16,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Extract coincident real spatial patches from Optical, SAR, and SCL ground-truth rasters.

    Args:
        optical_path: Path to real multi-band Sentinel-2 GeoTIFF.
        sar_path: Path to real dual-pol Sentinel-1 GeoTIFF.
        scl_path: Path to real Sentinel-2 Scene Classification Layer GeoTIFF.
        patch_size: Square window edge in pixels (e.g. 32x32).
        stride: Step size between adjacent patch origins.

    Returns:
        (optical_patches, sar_patches, labels)
    """
    # 1. Preprocess Optical Data
    optical_raw = load_optical(optical_path, band_mapping={"B02": 1, "B03": 2, "B04": 3, "B08": 4})
    optical_norm = normalize_optical(optical_raw, method="percentile", percentile_bounds=(1.0, 99.0))
    opt_data = optical_norm.data # (4, H, W)
    h, w = opt_data.shape[1], opt_data.shape[2]

    # 2. Preprocess SAR Data
    sar_raw = load_sar(sar_path, polarizations=["VV", "VH"])
    sar_cal = calibrate_sar(sar_raw, is_already_calibrated=True, to_db=True, min_db=-35.0, max_db=5.0)
    sar_cal.data = apply_speckle_filter(sar_cal.data, method="lee", kernel_size=5)
    sar_norm = normalize_sar(sar_cal, method="percentile", percentile_bounds=(1.0, 99.0))
    sar_aligned = reproject_to_reference(sar_norm, optical_norm, resampling_method="bilinear")
    sar_data = sar_aligned.data # (2, H, W)

    # 3. Read and Align Ground-Truth SCL
    scl_file = Path(scl_path)
    with rasterio.open(scl_file) as scl_src:
        if scl_src.shape != (h, w):
            scl_arr = np.empty((h, w), dtype=np.uint8)
            reproject(
                source=rasterio.band(scl_src, 1),
                destination=scl_arr,
                src_transform=scl_src.transform,
                src_crs=scl_src.crs,
                dst_transform=optical_norm.transform,
                dst_crs=optical_norm.crs,
                resampling=Resampling.nearest,
            )
        else:
            scl_arr = scl_src.read(1)

    opt_patches: List[np.ndarray] = []
    sar_patches: List[np.ndarray] = []
    labels: List[int] = []

    # 4. Extract sliding window patches
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            opt_p = opt_data[:, y : y + patch_size, x : x + patch_size]
            sar_p = sar_data[:, y : y + patch_size, x : x + patch_size]
            scl_p = scl_arr[y : y + patch_size, x : x + patch_size]

            # Skip if any NaN exists in the patch
            if np.isnan(opt_p).any() or np.isnan(sar_p).any():
                continue

            # Determine dominant class from SCL in this patch
            vals, counts = np.unique(scl_p, return_counts=True)
            valid_candidates = [(v, c) for v, c in zip(vals, counts) if v in SCL_TO_CLASS]
            if not valid_candidates:
                continue

            dominant_scl = max(valid_candidates, key=lambda item: item[1])[0]
            target_class = SCL_TO_CLASS[dominant_scl]

            opt_patches.append(opt_p)
            sar_patches.append(sar_p)
            labels.append(target_class)

    logger.info(
        f"Extracted {len(labels)} paired patches ({patch_size}x{patch_size}) from real satellite data. "
        f"Class breakdown: {[labels.count(c) for c in range(len(CLASS_NAMES))]}"
    )

    return opt_patches, sar_patches, labels


def create_real_dataloaders(
    optical_path: Union[str, Path],
    sar_path: Union[str, Path],
    scl_path: Union[str, Path],
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Generate train, validation, and test PyTorch DataLoaders from real satellite rasters.

    Returns:
        (train_loader, val_loader, test_loader, class_counts)
    """
    opt_patches, sar_patches, labels = prepare_real_multimodal_patches(
        optical_path=optical_path,
        sar_path=sar_path,
        scl_path=scl_path,
        patch_size=patch_size,
        stride=stride,
    )

    full_dataset = RealOpticalSARDataset(opt_patches, sar_patches, labels)
    total_samples = len(full_dataset)

    n_val = int(total_samples * val_split)
    n_test = int(total_samples * test_split)
    n_train = total_samples - n_val - n_test

    generator = torch.Generator().manual_seed(random_seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    class_counts = {name: labels.count(i) for i, name in enumerate(CLASS_NAMES)}

    return train_loader, val_loader, test_loader, class_counts
