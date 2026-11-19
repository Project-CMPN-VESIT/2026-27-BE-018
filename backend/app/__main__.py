"""Start the server with visible progress:   python -m app [--host 127.0.0.1] [--port 8000]

Plain `uvicorn app.main:app` prints nothing until PyTorch, rasterio and the model are loaded; on Windows the first
start can take 30-60 s (antivirus scans the DLLs), which looks like a hang. This launcher reports each step,
writes everything to data/logs/server.log, and exits cleanly on Ctrl+C.
"""
from __future__ import annotations

import argparse
import sys
import time


def _say(msg: str) -> None:
    print(f"[gln] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    t0 = time.time()
    try:
        _say("starting GeoLitterNet Watch - loading libraries (first start on Windows can take up to a minute)...")
        import numpy  # noqa: F401
        _say(f"  numpy ok ({time.time() - t0:.1f}s)")
        import torch
        _say(f"  torch {torch.__version__} ok ({time.time() - t0:.1f}s)")
        import rasterio
        _say(f"  rasterio {rasterio.__version__} ok ({time.time() - t0:.1f}s)")
        from app.main import app
        _say(f"  application imported ({time.time() - t0:.1f}s) - loading model and starting server...")
        import uvicorn
        _say(f"open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}  (Ctrl+C to stop)")
        uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
        return 0
    except KeyboardInterrupt:
        _say("stopped (Ctrl+C).")
        return 0
    except Exception:
        import traceback
        _say("FAILED to start:")
        traceback.print_exc()
        try:
            from app.config import get_settings
            p = get_settings().data_dir / "logs" / "server.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(f"\n--- startup failure {time.ctime()} ---\n")
                traceback.print_exc(file=f)
            _say(f"details written to {p}")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
