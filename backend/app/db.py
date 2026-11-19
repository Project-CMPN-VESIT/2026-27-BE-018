"""SQLite persistence for saved areas (AOIs) and detection runs."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Base(DeclarativeBase):
    pass


class AOI(Base):
    __tablename__ = "aois"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120))
    geometry: Mapped[dict] = mapped_column(JSON)
    area_km2: Mapped[float] = mapped_column(Float)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    def to_dict(self, last_run: Optional["Run"] = None) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "geometry": self.geometry, "area_km2": round(self.area_km2, 3),
                "notes": self.notes, "created_at": self.created_at.isoformat(),
                "last_run": last_run.summary() if last_run else None}


RUN_STATES = ("queued", "searching", "fetching", "inferring", "rendering", "done", "failed")


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    aoi_id: Mapped[Optional[str]] = mapped_column(ForeignKey("aois.id", ondelete="SET NULL"), nullable=True, index=True)
    geometry: Mapped[dict] = mapped_column(JSON)
    provider: Mapped[str] = mapped_column(String(16))
    scene_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    scene: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stats: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    layers: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    def summary(self) -> dict[str, Any]:
        st = self.stats or {}
        return {"id": self.id, "status": self.status, "created_at": self.created_at.isoformat(),
                "scene_datetime": (self.scene or {}).get("datetime"), "debris_area_m2": st.get("debris_area_m2"),
                "clusters": st.get("clusters"), "max_prob": st.get("max_prob")}

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "aoi_id": self.aoi_id, "geometry": self.geometry, "provider": self.provider,
                "scene_id": self.scene_id, "scene": self.scene, "params": self.params, "status": self.status,
                "progress": round(self.progress, 3), "message": self.message, "error": self.error,
                "stats": self.stats, "layers": self.layers, "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None}


_engine = None
_Session: Optional[sessionmaker] = None


def init_db() -> None:
    global _engine, _Session
    s = get_settings()
    _engine = create_engine(s.db_url, connect_args={"check_same_thread": False, "timeout": 30},
                            json_serializer=lambda o: json.dumps(o, default=str))

    @event.listens_for(_engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)


def session() -> Session:
    if _Session is None:
        init_db()
    assert _Session is not None
    return _Session()


def get_db() -> Iterator[Session]:
    db = session()
    try:
        yield db
    finally:
        db.close()


def latest_run_for(db: Session, aoi_id: str) -> Optional[Run]:
    return db.scalars(select(Run).where(Run.aoi_id == aoi_id).order_by(Run.created_at.desc()).limit(1)).first()


__all__ = ["AOI", "Run", "RUN_STATES", "init_db", "session", "get_db", "latest_run_for", "now"]
