"""Builds the configured imagery providers once per process."""
from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from .base import ImageryProvider, ProviderError
from .gee import GeeProvider
from .stac import StacProvider
from .upload import UploadProvider


@lru_cache
def providers() -> dict[str, ImageryProvider]:
    s = get_settings()
    return {
        "stac": StacProvider(s.stac_url, s.stac_collection, s.stac_timeout_s),
        "gee": GeeProvider(s.gee_enabled, s.gee_project, s.gee_service_account, s.gee_key_file, s.gee_collection),
        "upload": UploadProvider(s.uploads_dir),
    }


def get_provider(name: str) -> ImageryProvider:
    p = providers().get(name)
    if p is None:
        raise ProviderError(f"unknown imagery provider '{name}' (use one of: {', '.join(providers())})")
    ok, why = p.status()
    if not ok:
        raise ProviderError(f"provider '{name}' unavailable: {why}")
    return p


def upload_provider() -> UploadProvider:
    p = providers()["upload"]
    assert isinstance(p, UploadProvider)
    return p
