"""Geometry helpers: AOI validation, area, UTM selection and 10 m analysis grids."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Geod, Transformer
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
import shapely
from shapely.ops import unary_union

GEOD = Geod(ellps="WGS84")
RES_M = 10.0


class GeometryError(ValueError):
    pass


def parse_geometry(obj: Any) -> BaseGeometry:
    """Accept a GeoJSON geometry, Feature or FeatureCollection in EPSG:4326 and return one valid polygon(al) shape."""
    if not isinstance(obj, dict) or "type" not in obj:
        raise GeometryError("geometry must be a GeoJSON object")
    t = obj["type"]
    if t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in obj.get("features", []) if f.get("geometry")]
        if not geoms:
            raise GeometryError("empty FeatureCollection")
        g = unary_union(geoms)
    elif t == "Feature":
        if not obj.get("geometry"):
            raise GeometryError("Feature has no geometry")
        g = shape(obj["geometry"])
    else:
        g = shape(obj)
    if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
        raise GeometryError("geometry must be a Polygon or MultiPolygon")
    if not g.is_valid:
        g = g.buffer(0)
    minx, miny, maxx, maxy = g.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise GeometryError("coordinates must be WGS84 longitude/latitude")
    return g


def area_km2(g: BaseGeometry) -> float:
    return abs(GEOD.geometry_area_perimeter(g)[0]) / 1e6


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) % 60) + 1
    return (32600 if lat >= 0 else 32700) + zone


def to_crs(g: BaseGeometry, dst_crs: str | int, src_crs: str | int = 4326) -> BaseGeometry:
    tr = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)
    return shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


@dataclass
class Grid:
    crs: str            # e.g. "EPSG:32616"
    transform: Affine   # north-up, RES_M pixels
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x0, y1 = self.transform.c, self.transform.f
        return x0, y1 - self.height * RES_M, x0 + self.width * RES_M, y1

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def footprint_wgs84(self) -> BaseGeometry:
        return to_crs(box(*self.bounds), 4326, self.crs)


def grid_for(g_wgs84: BaseGeometry, crs: str | int | None = None, res: float = RES_M) -> Grid:
    """A pixel grid snapped to multiples of `res` covering the AOI in `crs` (default: local UTM)."""
    if crs is None:
        c = g_wgs84.centroid
        crs = f"EPSG:{utm_epsg(c.x, c.y)}"
    crs = CRS.from_user_input(crs).to_string()
    minx, miny, maxx, maxy = to_crs(g_wgs84, crs).bounds
    minx, miny = math.floor(minx / res) * res, math.floor(miny / res) * res
    maxx, maxy = math.ceil(maxx / res) * res, math.ceil(maxy / res) * res
    w, h = max(1, int(round((maxx - minx) / res))), max(1, int(round((maxy - miny) / res)))
    return Grid(crs=crs, transform=Affine(res, 0, minx, 0, -res, maxy), width=w, height=h)


def aoi_mask(g_wgs84: BaseGeometry, grid: Grid) -> np.ndarray:
    """Boolean (H, W) mask of grid pixels whose centre lies inside the AOI."""
    from rasterio.features import geometry_mask
    return geometry_mask([mapping(to_crs(g_wgs84, grid.crs))], out_shape=grid.shape, transform=grid.transform,
                         invert=True, all_touched=False)
