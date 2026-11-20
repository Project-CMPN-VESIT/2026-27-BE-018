"""Request bodies for the public API."""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class SceneSearchIn(BaseModel):
    provider: Optional[str] = None
    geometry: dict[str, Any]
    start_date: date
    end_date: date
    max_cloud: float = Field(30, ge=0, le=100)
    limit: int = Field(20, ge=1, le=100)

    @model_validator(mode="after")
    def _dates(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class RunIn(BaseModel):
    geometry: Optional[dict[str, Any]] = None   # optional when aoi_id is given, or for an uploaded image
    aoi_id: Optional[str] = None
    provider: Optional[str] = None
    scene_id: Optional[str] = None              # omit to auto-select the best scene in the date range
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    max_cloud: float = Field(30, ge=0, le=100)
    threshold: Optional[float] = Field(None, ge=0.05, le=0.99)
    min_cluster_pixels: Optional[int] = Field(None, ge=1, le=10000)
    mask_land: Optional[bool] = None
    mask_clouds: Optional[bool] = None

    @model_validator(mode="after")
    def _dates(self):
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class AOIIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    geometry: dict[str, Any]
    notes: str = Field("", max_length=2000)


class AOIPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    notes: Optional[str] = Field(None, max_length=2000)


class AOIRunIn(BaseModel):
    provider: Optional[str] = None
    days_back: int = Field(30, ge=1, le=365)
    max_cloud: float = Field(30, ge=0, le=100)
    threshold: Optional[float] = Field(None, ge=0.05, le=0.99)


class ReprocessIn(BaseModel):
    threshold: float = Field(..., ge=0.05, le=0.99)
    min_cluster_pixels: Optional[int] = Field(None, ge=1, le=10000)
