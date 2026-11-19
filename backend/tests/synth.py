"""Synthetic Sentinel-2 scenes for tests (no network needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

# rough surface-reflectance spectra in MARIDA band order (B1..B8, B8A, B11, B12)
WATER = [0.045, 0.040, 0.032, 0.016, 0.011, 0.008, 0.007, 0.006, 0.005, 0.002, 0.001]
LAND = [0.030, 0.040, 0.070, 0.050, 0.100, 0.250, 0.300, 0.320, 0.330, 0.200, 0.100]
DEBRIS = [0.050, 0.050, 0.052, 0.046, 0.052, 0.062, 0.074, 0.085, 0.084, 0.040, 0.020]
CLOUD = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.50, 0.51, 0.50, 0.32, 0.22]

CRS = "EPSG:32616"                      # Bay of Honduras (a MARIDA region)
ORIGIN = (380000.0, 1780000.0)          # UTM x, y (top-left)


def scene(h: int = 300, w: int = 400, seed: int = 0):
    """Returns reflectance (11,H,W), SCL (H,W), truth debris mask."""
    rng = np.random.default_rng(seed)
    refl = np.empty((11, h, w), np.float32)
    scl = np.full((h, w), 6, np.uint8)
    for b in range(11):
        refl[b] = WATER[b]
    land = np.zeros((h, w), bool)
    land[:, : w // 6] = True                        # coastline on the west
    cloud = np.zeros((h, w), bool)
    cloud[h - 60:, w - 80:] = True                  # cloud bank SE
    debris = np.zeros((h, w), bool)
    for (r, c, s) in [(60, 200, 4), (150, 300, 6), (220, 150, 3), (100, 120, 5)]:
        debris[r:r + s, c:c + 2 * s] = True          # floating filaments
    for m, spec, code in [(land, LAND, 4), (debris, DEBRIS, 4), (cloud, CLOUD, 9)]:
        for b in range(11):
            refl[b][m] = spec[b]
        scl[m] = code                                # Sen2Cor often calls floating debris "vegetation"
    refl *= rng.normal(1.0, 0.03, refl.shape).astype(np.float32)
    return refl, scl, debris


def transform(res: float = 10.0) -> Affine:
    return Affine(res, 0, ORIGIN[0], 0, -res, ORIGIN[1])


def write_tif(path: Path, arr: np.ndarray, crs: str = CRS, tr: Affine | None = None, descriptions=None,
              dtype: str | None = None, nodata=None) -> Path:
    a = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                       dtype=dtype or a.dtype.name, crs=crs, transform=tr or transform(), nodata=nodata,
                       tiled=a.shape[1] >= 256 and a.shape[2] >= 256, compress="deflate") as dst:
        dst.write(a.astype(dtype or a.dtype.name))
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)
    return path


def bbox_lonlat(h: int = 300, w: int = 400, inset_px: int = 5) -> dict:
    from pyproj import Transformer
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    x0, y1 = ORIGIN[0] + inset_px * 10, ORIGIN[1] - inset_px * 10
    x1, y0 = ORIGIN[0] + (w - inset_px) * 10, ORIGIN[1] - (h - inset_px) * 10
    (lo0, la0), (lo1, la1) = tr.transform(x0, y0), tr.transform(x1, y1)
    return {"type": "Polygon", "coordinates": [[[lo0, la0], [lo1, la0], [lo1, la1], [lo0, la1], [lo0, la0]]]}
