"""Provider interface shared by STAC, Google Earth Engine and uploaded GeoTIFFs."""
from __future__ import annotations

import abc
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from shapely.geometry.base import BaseGeometry

from .geo import Grid

log = logging.getLogger(__name__)
ProgressFn = Callable[[float, str], None]

# Analysis band order (MARIDA): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
BAND_IDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

GDAL_ENV = dict(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.jp2",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="1",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
)


class ProviderError(RuntimeError):
    """User-facing failure (bad scene id, no coverage, provider not configured...)."""


@dataclass
class SceneInfo:
    id: str
    provider: str
    datetime: str                 # ISO 8601 UTC acquisition time
    cloud_cover: Optional[float]  # scene-level %, from metadata
    platform: str = ""
    tiles: list[str] = field(default_factory=list)
    coverage: Optional[float] = None   # fraction of the AOI covered by the scene footprint (0..1)
    thumbnail: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneData:
    reflectance: np.ndarray       # (11, H, W) float32 surface reflectance, NaN = no data
    scl: Optional[np.ndarray]     # (H, W) uint8 Sen2Cor scene classification, if the source has it
    grid: Grid
    scene: SceneInfo


class ImageryProvider(abc.ABC):
    name: str = ""
    label: str = ""

    @abc.abstractmethod
    def status(self) -> tuple[bool, str]:
        """(available, human-readable reason)."""

    @abc.abstractmethod
    def search(self, aoi: BaseGeometry, start: date, end: date, max_cloud: float, limit: int) -> list[SceneInfo]:
        ...

    @abc.abstractmethod
    def load(self, scene_id: str, aoi: BaseGeometry, progress: Optional[ProgressFn] = None) -> SceneData:
        ...


def read_to_grid(href: str, grid: Grid, resampling: Resampling, band: int = 1, dtype="float32",
                 fill: float = 0) -> np.ndarray:
    """Read one band of a (possibly remote, possibly differently projected/resolved) raster onto `grid`.

    Same CRS: a windowed (boundless) read with resampling - only the needed COG blocks are fetched.
    Other CRS: GDAL warp into the grid (also block-wise; reads only the source area it needs).
    """
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            if src.crs is not None and src.crs.to_string() == grid.crs:
                win = from_bounds(*grid.bounds, transform=src.transform)
                return src.read(band, window=win, out_shape=grid.shape, resampling=resampling,
                                boundless=True, fill_value=fill, out_dtype=dtype)
            out = np.full(grid.shape, fill, dtype=dtype)
            reproject(rasterio.band(src, band), out, dst_transform=grid.transform, dst_crs=grid.crs,
                      dst_nodata=fill, resampling=resampling)
            return out


def mosaic_fill(dst: np.ndarray, src: np.ndarray) -> None:
    """In-place: fill NaN pixels of dst (C,H,W) from src where src has data."""
    hole = np.isnan(dst[0])
    if hole.any():
        dst[:, hole] = src[:, hole]
