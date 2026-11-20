"""Export real Sentinel-2 data for the demo video:   python -m app.demo_export

Downloads (with the app's own STAC provider) the exact scenes used in two finished runs and writes, per site,
into ../demo_data/:
  <site>_bands.tif    11 bands (B1..B12, MARIDA order) + SCL, uint16 DN = reflectance x 10000, 10 m, native UTM
  <site>_context.tif  true-colour (TCI) context around the site at 40 m, for an offline basemap
  <site>.json         STAC metadata of the scene and of all scenes found in the run's date window
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import box

from .config import BACKEND_DIR
from .imagery.base import read_to_grid
from .imagery.geo import grid_for
from .imagery.registry import get_provider

SITES = [
    {"site": "chetumal", "scene": "S2B_16QCE_20261030_0_L2A", "bbox": [-88.1286, 17.7056, -88.0407, 18.0036],
     "start": "2026-10-05", "end": "2026-11-04", "context_deg": 0.30},
    {"site": "cuba", "scene": "S2B_16QGK_20261024_0_L2A", "bbox": [-84.5227, 21.6875, -84.3732, 21.7954],
     "start": "2026-10-05", "end": "2026-11-04", "context_deg": 0.25},
]
OUT = BACKEND_DIR.parent / "demo_data"


def say(m: str) -> None:
    print(f"[demo-export] {m}", flush=True)


def export(site: dict) -> None:
    prov = get_provider("stac")
    x0, y0, x1, y1 = site["bbox"]
    aoi = box(x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01)
    t = time.time()
    say(f"{site['site']}: searching scenes {site['start']} .. {site['end']}")
    scenes = prov.search(box(x0, y0, x1, y1), date.fromisoformat(site["start"]), date.fromisoformat(site["end"]), 100, 30)
    items_json = [i.to_dict() for i in prov._items_seen.values()]
    say(f"{site['site']}: downloading 11 bands + SCL of {site['scene']} (this is the slow part)...")
    data = prov.load(site["scene"], aoi, lambda f, m: print(f"\r  {f:5.0%} {m}      ", end="", flush=True))
    print()
    dn = np.where(np.isnan(data.reflectance), 0, np.clip(np.round(data.reflectance * 10000), 1, 65535)).astype(np.uint16)
    g = data.grid
    bands = dict(driver="GTiff", height=g.height, width=g.width, count=12, dtype="uint16", crs=g.crs,
                 transform=g.transform, nodata=0, compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(OUT / f"{site['site']}_bands.tif", "w", **bands) as dst:
        dst.write(dn, list(range(1, 12)))
        dst.write(data.scl if data.scl is not None else np.zeros(g.shape, np.uint8), 12)
        for i, n in enumerate(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"], 1):
            dst.set_band_description(i, n)
    say(f"{site['site']}: bands saved ({g.width}x{g.height} px, {time.time() - t:.0f}s)")

    # true-colour context for an offline basemap (uses the COG overviews, so it is a small download)
    item = next(i for i in prov._items_seen.values() if i.id == site["scene"])
    c = site["context_deg"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cgrid = grid_for(box(cx - c, cy - c, cx + c, cy + c), g.crs, res=40.0)
    if "visual" in item.assets:
        href = item.assets["visual"].href
        rgb = np.stack([read_to_grid(href, cgrid, Resampling.average, band=b, dtype="uint8") for b in (1, 2, 3)])
    else:   # no TCI asset: stretch red/green/blue reflectance ourselves
        chans = []
        for key in ("red", "green", "blue"):
            v = read_to_grid(item.assets[key].href, cgrid, Resampling.average)
            chans.append(np.clip(v / 2500.0 * 255, 0, 255).astype("uint8"))
        rgb = np.stack(chans)
    ctx = dict(driver="GTiff", height=cgrid.height, width=cgrid.width, count=3, dtype="uint8", crs=cgrid.crs,
               transform=cgrid.transform, nodata=0, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(OUT / f"{site['site']}_context.tif", "w", **ctx) as dst:
        dst.write(rgb)
    (OUT / f"{site['site']}.json").write_text(json.dumps({
        "site": site, "scene": data.scene.to_dict(), "scenes_found": [s.to_dict() for s in scenes],
        "items": items_json}, default=str))
    say(f"{site['site']}: done in {time.time() - t:.0f}s")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for s in SITES:
        for attempt in (1, 2, 3):
            try:
                export(s)
                break
            except Exception as e:
                say(f"{s['site']}: attempt {attempt} failed: {e}")
                if attempt == 3:
                    ok = False
                time.sleep(3)
    say(f"{'ALL DONE' if ok else 'FINISHED WITH ERRORS'} - files are in {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
