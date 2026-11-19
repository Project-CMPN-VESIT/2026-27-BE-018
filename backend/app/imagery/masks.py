"""Analysis masks: which pixels are open water we can trust a debris prediction on.

The model was trained on coastal/ocean MARIDA patches; it has no "land" class, so land, cloud and
no-data pixels must be removed before the heatmap and detections are produced.

Source of truth is the Sen2Cor Scene Classification (SCL) when available, with one important fix:
floating debris / Sargassum is often labelled "vegetation" (SCL 4) by Sen2Cor because it has an NIR
signal. Small non-water holes fully enclosed by water are therefore put back into the analysis area
(`max_hole_fill_px`). Without SCL (e.g. some uploads) a spectral water test (NDWI/MNDWI) is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

SCL_NODATA, SCL_SATURATED, SCL_DARK, SCL_SHADOW, SCL_VEG, SCL_SOIL, SCL_WATER = 0, 1, 2, 3, 4, 5, 6
SCL_UNCLASSIFIED, SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS, SCL_SNOW = 7, 8, 9, 10, 11
CLOUD_CLASSES = (SCL_SHADOW, SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS)
LAND_CLASSES = (SCL_VEG, SCL_SOIL, SCL_SNOW)


@dataclass
class AnalysisMask:
    valid: np.ndarray        # pixels analysed (water, cloud-free, has data, inside AOI)
    nodata: np.ndarray
    cloud: np.ndarray
    land: np.ndarray
    method: str

    def fractions(self, inside: np.ndarray) -> dict[str, float]:
        n = max(int(inside.sum()), 1)
        return {k: round(float((getattr(self, k) & inside).sum()) / n, 4) for k in ("valid", "nodata", "cloud", "land")}


def _fill_small_holes(water: np.ndarray, max_px: int) -> np.ndarray:
    if max_px <= 0:
        return water
    filled = ndimage.binary_fill_holes(water)
    holes = filled & ~water
    if not holes.any():
        return water
    lab, n = ndimage.label(holes)
    sizes = ndimage.sum(holes, lab, index=np.arange(1, n + 1))
    small = np.zeros(n + 1, bool)
    small[1:] = sizes <= max_px
    return water | small[lab]


def spectral_water(refl: np.ndarray) -> np.ndarray:
    """Fallback water test on reflectance (11, H, W): NDWI > 0 or MNDWI > 0.2, with NIR not too bright."""
    g, nir, swir = refl[2], refl[7], refl[9]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (g - nir) / (g + nir)
        mndwi = (g - swir) / (g + swir)
    return np.nan_to_num(((ndwi > 0) | (mndwi > 0.2)) & (nir < 0.25), nan=False).astype(bool)


def spectral_cloud(refl: np.ndarray) -> np.ndarray:
    """Conservative bright-cloud test for sources without SCL (blue and SWIR both bright)."""
    blue, swir = refl[1], refl[9]
    return np.nan_to_num((blue > 0.20) & (swir > 0.15), nan=False).astype(bool)


def build_mask(refl: np.ndarray, scl: Optional[np.ndarray], inside_aoi: np.ndarray, *, mask_clouds: bool = True,
               mask_land: bool = True, max_hole_fill_px: int = 2500) -> AnalysisMask:
    nodata = np.isnan(refl).any(0)
    if scl is not None and (scl > 0).any():
        cloud = np.isin(scl, CLOUD_CLASSES) | (scl == SCL_SATURATED)
        nodata |= scl == SCL_NODATA
        water = np.isin(scl, (SCL_WATER, SCL_DARK, SCL_UNCLASSIFIED))
        method = "scl"
    else:
        cloud = spectral_cloud(refl)
        water = spectral_water(refl)
        method = "spectral"
    water = _fill_small_holes(water | cloud, max_hole_fill_px) & ~cloud | water
    land = ~water & ~nodata & ~cloud
    valid = inside_aoi & ~nodata
    if mask_clouds:
        valid &= ~cloud
    if mask_land:
        valid &= ~land
    else:
        land = np.zeros_like(land)
    if not mask_clouds:
        cloud = np.zeros_like(cloud)
    return AnalysisMask(valid=valid, nodata=nodata & inside_aoi, cloud=cloud & inside_aoi & ~nodata,
                        land=land & inside_aoi, method=method)
