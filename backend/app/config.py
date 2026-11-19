"""Runtime configuration (environment variables prefixed GLN_, or a .env file)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GLN_", env_file=(BACKEND_DIR.parent / ".env", BACKEND_DIR / ".env"),
                                      env_file_encoding="utf-8", extra="ignore")

    # --- storage ---
    data_dir: Path = BACKEND_DIR / "data"
    model_path: Path = BACKEND_DIR / "models" / "geolitternet_v3_seed42.pth"

    # --- inference ---
    device: Literal["cpu", "cuda"] = "cpu"
    torch_threads: int = 0            # 0 = let PyTorch decide
    batch_size: int = 4
    tile_overlap: int = 64            # px of overlap between 256 px tiles (blended)

    # --- detection defaults ---
    debris_threshold: float = Field(0.5, ge=0.05, le=0.99)
    min_cluster_pixels: int = 3       # 3 px = 300 m^2 at 10 m
    heatmap_floor: float = 0.15       # probabilities below this are drawn transparent
    mask_clouds: bool = True
    mask_land: bool = True
    max_hole_fill_px: int = 2500      # floating objects inside water up to 0.25 km^2 are kept as "water"

    # --- request limits ---
    max_aoi_km2: float = 400.0
    max_upload_mb: int = 1024
    scene_search_limit: int = 30

    # --- imagery providers ---
    default_provider: Literal["stac", "gee", "upload"] = "stac"
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    stac_collection: str = "sentinel-2-l2a"
    stac_timeout_s: int = 60
    gee_enabled: bool = False
    gee_project: str = ""
    gee_service_account: str = ""     # e.g. gln@my-project.iam.gserviceaccount.com
    gee_key_file: Path | None = None  # path to the service-account JSON key
    gee_collection: str = "COPERNICUS/S2_SR_HARMONIZED"

    # --- API ---
    api_key: str = ""                 # if set, every /api call (except /api/health) needs X-API-Key
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'gln.sqlite3').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.runs_dir, s.uploads_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
