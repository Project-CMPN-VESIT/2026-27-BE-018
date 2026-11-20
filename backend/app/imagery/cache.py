"""On-disk cache of downloaded scene pixels.

Downloading Sentinel-2 bands is the slowest step (minutes for large areas on a long-distance link), while the
pixels of a given scene over a given area never change. Re-running the same area/scene - a new threshold,
other mask settings, a demo, a second look - is served from here in about a second.

Entries are keyed by (provider, collection, scene id, exact AOI geometry) and stored as raw .npy arrays (lossless,
fast to read). The cache is size-capped; least recently used entries are evicted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from affine import Affine
from shapely.geometry.base import BaseGeometry

from .base import SceneData, SceneInfo
from .geo import Grid

log = logging.getLogger(__name__)
_lock = threading.Lock()


class SceneCache:
    def __init__(self, root: Path, max_gb: float = 5.0, enabled: bool = True):
        self.root = Path(root)
        self.max_bytes = int(max_gb * 1024 ** 3)
        self.enabled = enabled and max_gb > 0
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(provider: str, collection: str, scene_id: str, aoi: BaseGeometry) -> str:
        h = hashlib.sha1()
        for part in (provider, collection, scene_id, aoi.wkb_hex):
            h.update(part.encode())
            h.update(b"|")
        return h.hexdigest()[:24]

    def get(self, key: str) -> Optional[SceneData]:
        if not self.enabled:
            return None
        d = self.root / key
        meta_p = d / "meta.json"
        if not meta_p.exists():
            return None
        try:
            m = json.loads(meta_p.read_text())
            refl = np.load(d / "refl.npy")
            scl = np.load(d / "scl.npy") if (d / "scl.npy").exists() else None
            grid = Grid(crs=m["crs"], transform=Affine(*m["transform"]), width=m["width"], height=m["height"])
            if refl.shape != (11, grid.height, grid.width):
                raise ValueError("shape mismatch")
            meta_p.touch()                      # LRU bookkeeping
            return SceneData(reflectance=refl, scl=scl, grid=grid, scene=SceneInfo(**m["scene"]))
        except Exception as e:                  # corrupt / partial entry: drop it and download again
            log.warning("scene cache entry %s unreadable (%s); ignoring", key, e)
            shutil.rmtree(d, ignore_errors=True)
            return None

    def put(self, key: str, data: SceneData) -> None:
        if not self.enabled:
            return
        d = self.root / key
        tmp = self.root / f".{key}.tmp{int(time.time() * 1000)}"
        try:
            tmp.mkdir(parents=True, exist_ok=True)
            np.save(tmp / "refl.npy", data.reflectance, allow_pickle=False)
            if data.scl is not None:
                np.save(tmp / "scl.npy", data.scl, allow_pickle=False)
            g = data.grid
            (tmp / "meta.json").write_text(json.dumps({
                "crs": g.crs, "transform": list(g.transform)[:6], "width": g.width, "height": g.height,
                "scene": data.scene.to_dict(), "stored": time.time()}))
            with _lock:
                shutil.rmtree(d, ignore_errors=True)
                tmp.rename(d)
            self._evict()
        except Exception as e:                  # caching is best-effort; never fail a run because of it
            log.warning("could not cache scene %s: %s", key, e)
            shutil.rmtree(tmp, ignore_errors=True)

    def _evict(self) -> None:
        with _lock:
            entries = []
            total = 0
            for d in self.root.iterdir():
                if not d.is_dir() or d.name.startswith("."):
                    continue
                size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
                mt = (d / "meta.json").stat().st_mtime if (d / "meta.json").exists() else 0
                entries.append((mt, size, d))
                total += size
            for _, size, d in sorted(entries):
                if total <= self.max_bytes:
                    break
                shutil.rmtree(d, ignore_errors=True)
                total -= size

    def stats(self) -> dict:
        n, size = 0, 0
        for d in self.root.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                n += 1
                size += sum(f.stat().st_size for f in d.iterdir() if f.is_file())
        return {"entries": n, "size_mb": round(size / 1e6, 1), "max_gb": round(self.max_bytes / 1024 ** 3, 1),
                "enabled": self.enabled}

    def clear(self) -> None:
        with _lock:
            for d in self.root.iterdir():
                shutil.rmtree(d, ignore_errors=True)
