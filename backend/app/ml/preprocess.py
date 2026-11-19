"""Input preparation that reproduces the training pipeline bit-for-bit.

Training (notebook "GeoLitterNet New V2", MARIDAVisualizationDataset._load_patch) did:

    raw_11    = nan_to_num(raw, nan=0, posinf=10000, neginf=0)      # MARIDA patches: ACOLITE reflectance, ~0..1
    scaled_11 = clip(raw_11 / 10000, 0, 2)
    physics_5 = [FDI, PI, NDVI, MNDWI, SWR] computed on scaled_11
    x_16      = concat(scaled_11, physics_5)

MARIDA patches already hold reflectance (0..1), so the network was trained on reflectance / 10 000.
The checkpoint confirms it: the first BatchNorm has running_var ~1e-11. Production imagery must therefore be
converted to physical reflectance first (provider's job) and then go through exactly this function.
`RAW_DIVISOR` is configurable only for a future retrained model; changing it for this checkpoint breaks it.
"""
from __future__ import annotations

import numpy as np

# Band order expected everywhere in the product (MARIDA order, no B9/B10).
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
INDEX_NAMES = ["FDI", "PI", "NDVI", "MNDWI", "SWR"]
RAW_DIVISOR = 10000.0


def compute_non_redundant_indices(raw_11: np.ndarray) -> np.ndarray:
    """Exact copy of the training implementation (operates on the *scaled* bands)."""
    eps = 1e-6
    raw_11 = np.nan_to_num(raw_11, nan=0.0, posinf=2.0, neginf=0.0)

    B03, B04, B06 = raw_11[2], raw_11[3], raw_11[5]
    B08, B8A, B11, B12 = raw_11[7], raw_11[8], raw_11[9], raw_11[10]

    fdi = np.clip(B8A - (B06 + (B11 - B06) * ((865.0 - 740.0) / (1610.0 - 740.0))), -0.05, 0.05) / 0.05
    pi = np.clip(B08 / (B08 + B04 + eps), 0.0, 2.0) / 2.0
    ndvi = np.clip((B08 - B04) / (B08 + B04 + eps), -1.0, 1.0)
    mndwi = np.clip((B03 - B11) / (B03 + B11 + eps), -1.0, 1.0)
    swr = np.clip(B11 / (B12 + eps), 0.0, 5.0) / 5.0

    indices = np.stack([fdi, pi, ndvi, mndwi, swr], axis=0)
    return np.nan_to_num(indices, nan=0.0).astype(np.float32)


def build_input_stack(reflectance_11: np.ndarray, raw_divisor: float = RAW_DIVISOR) -> np.ndarray:
    """reflectance_11: (11, H, W) physical surface reflectance (0..~1), NaN where no data.

    Returns the (16, H, W) float32 tensor the network expects.
    """
    if reflectance_11.ndim != 3 or reflectance_11.shape[0] != 11:
        raise ValueError(f"expected (11, H, W) reflectance, got {reflectance_11.shape}")
    raw_11 = np.nan_to_num(reflectance_11.astype(np.float32), nan=0.0, posinf=10000.0, neginf=0.0)
    scaled_11 = np.clip(raw_11 / raw_divisor, 0.0, 2.0)
    physics_5 = compute_non_redundant_indices(scaled_11)
    return np.concatenate([scaled_11, physics_5], axis=0).astype(np.float32)
