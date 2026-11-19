"""Background job queue: one worker thread runs detections in submission order.

Inference is CPU-bound and the model holds a lock, so a single worker is the right default for one
machine. Queue state lives in SQLite, so queued runs survive a restart and interrupted runs are marked
failed instead of hanging forever.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from sqlalchemy import select

from .db import Run, now, session
from .ml.engine import ModelEngine
from .pipeline.runner import execute

log = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.current: Optional[str] = None

    def start(self) -> None:
        with session() as db:
            for r in db.scalars(select(Run).where(Run.status.in_(("searching", "fetching", "inferring", "rendering")))):
                r.status, r.error, r.finished_at = "failed", "interrupted by a server restart - run it again", now()
            queued = [r.id for r in db.scalars(select(Run).where(Run.status == "queued").order_by(Run.created_at))]
            db.commit()
        for rid in queued:
            self.q.put(rid)
        self.thread = threading.Thread(target=self._loop, name="gln-worker", daemon=True)
        self.thread.start()
        log.info("job worker started (%d queued runs recovered)", len(queued))

    def submit(self, run_id: str) -> None:
        self.q.put(run_id)

    def position(self, run_id: str) -> int:
        items = list(self.q.queue)
        return items.index(run_id) + 1 if run_id in items else 0

    def stop(self) -> None:
        self.q.put(None)

    def _loop(self) -> None:
        while True:
            rid = self.q.get()
            if rid is None:
                return
            self.current = rid
            try:
                execute(rid, self.engine)
            except Exception:  # execute() already records failures; never let the worker die
                log.exception("worker error on run %s", rid)
            finally:
                self.current = None
                self.q.task_done()
