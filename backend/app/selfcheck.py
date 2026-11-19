"""First-run diagnostics:  python -m app.selfcheck [--live]

Checks the model, storage, and each imagery provider. With --live it also runs a real detection on a
small area of the Bay of Honduras (a MARIDA site) using the default provider, end to end.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

from shapely.geometry import box

OK, FAIL, WARN = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="download a real scene and run the model")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    bad = 0

    from .config import get_settings
    s = get_settings()
    print(f"data dir     : {s.data_dir}")

    from .ml.engine import ModelEngine
    try:
        eng = ModelEngine(s.model_path, s.device, s.torch_threads, s.batch_size, s.tile_overlap)
        eng.load()
        print(f"model        : {OK}  {eng.n_params:,} params, sha256 {eng.sha256[:16]}..., device {eng.device}")
    except Exception as e:
        print(f"model        : {FAIL} {e}")
        return 1

    from .imagery.registry import providers
    aoi = box(-88.10, 16.05, -88.02, 16.12)   # ~ 8 x 8 km, Gulf of Honduras
    end = date.today()
    for name, p in providers().items():
        ok, why = p.status()
        if not ok:
            print(f"{name:<13}: {WARN} {why}")
            continue
        if name == "upload":
            print(f"{name:<13}: {OK}  {why}")
            continue
        try:
            t = time.time()
            sc = p.search(aoi, end - timedelta(days=60), end, 60, 10)
            print(f"{name:<13}: {OK}  {len(sc)} scenes in the last 60 days ({time.time() - t:.1f}s)")
        except Exception as e:
            bad += 1
            print(f"{name:<13}: {FAIL} {e}")

    if args.live:
        from .imagery.masks import build_mask
        from .imagery.geo import aoi_mask
        from .imagery.registry import get_provider
        from .pipeline.runner import pick_best_scene
        name = args.provider or s.default_provider
        p = get_provider(name)
        best = pick_best_scene(p.search(aoi, end - timedelta(days=90), end, 40, 30))
        if best is None:
            print(f"live         : {WARN} no scene <= 40% cloud in 90 days")
            return bad
        t = time.time()
        data = p.load(best.id, aoi)
        t_fetch = time.time() - t
        res = eng.predict(data.reflectance)
        m = build_mask(data.reflectance, data.scl, aoi_mask(aoi, data.grid))
        print(f"live         : {OK}  scene {best.datetime[:10]} cloud {best.cloud_cover}% | grid {data.grid.width}x{data.grid.height} "
              f"| fetch {t_fetch:.1f}s, model {res.seconds:.1f}s | water {m.valid.mean():.0%} "
              f"| max P(debris) {float(res.debris_prob[m.valid].max()) if m.valid.any() else 0:.2f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
