"""GeoLitterNet Watch - FastAPI application entry point.

Run:  python -m app   (or: uvicorn app.main:app --host 0.0.0.0 --port 8000)
If ../frontend/dist exists (or GLN_STATIC_DIR points to a built UI), the web app is served at "/".
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import public, router
from .config import BACKEND_DIR, get_settings
from .db import init_db
from .imagery.base import ProviderError
from .imagery.geo import GeometryError
from .jobs import JobQueue
from .ml.engine import ModelEngine

log = logging.getLogger("gln")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=s.log_level, format=fmt)
    (s.data_dir / "logs").mkdir(parents=True, exist_ok=True)
    root_log = logging.getLogger()
    if not any(getattr(h, "_gln_file", False) for h in root_log.handlers):
        fh = RotatingFileHandler(s.data_dir / "logs" / "server.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        fh._gln_file = True  # type: ignore[attr-defined]
        for name in ("", "uvicorn"):   # uvicorn's logger does not propagate to root
            logging.getLogger(name).addHandler(fh)
    logging.getLogger().setLevel(s.log_level)
    init_db()
    engine = ModelEngine(s.model_path, s.device, s.torch_threads, s.batch_size, s.tile_overlap)
    engine.load()
    jobs = JobQueue(engine)
    jobs.start()
    app.state.engine, app.state.jobs = engine, jobs
    log.info("GeoLitterNet Watch %s ready (data dir: %s)", __version__, s.data_dir)
    yield
    jobs.stop()


app = FastAPI(title="GeoLitterNet Watch API", version=__version__, lifespan=lifespan,
              description="Marine-debris heatmaps from Sentinel-2 using the GeoLitterNet segmentation model.")
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"],
                   allow_headers=["*"])


@app.exception_handler(ProviderError)
async def _provider_error(_: Request, exc: ProviderError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(GeometryError)
async def _geom_error(_: Request, exc: GeometryError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(public)
app.include_router(router)

# ---- optional: serve the built web UI from the same process (no Node needed at runtime) ----
_static = Path(os.environ.get("GLN_STATIC_DIR", BACKEND_DIR.parent / "frontend" / "dist"))
if (_static / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith("api/") or path == "api":
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        f = _static / path
        if path and f.is_file() and _static in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(_static / "index.html")
