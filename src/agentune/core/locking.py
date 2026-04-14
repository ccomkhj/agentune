"""Lease-based concurrency control for campaign execution."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import timedelta

logger = logging.getLogger(__name__)

from agentune.core.db import Database

LEASE_DURATION = timedelta(minutes=15)
REFRESH_INTERVAL = timedelta(minutes=5)


class LeaseError(Exception):
    pass


class LeaseManager:
    def __init__(self, db: Database, worker_id: str | None = None) -> None:
        self._db = db
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"

    def acquire(self, campaign_id: int) -> None:
        with self._db.connection() as conn:
            cur = conn.execute(
                "UPDATE campaigns SET claimed_by = %s, "
                "claim_expires_at = now() + %s "
                "WHERE id = %s AND (claimed_by IS NULL OR claim_expires_at < now())",
                (self.worker_id, LEASE_DURATION, campaign_id),
            )
            if cur.rowcount == 0:
                raise LeaseError(
                    f"Campaign {campaign_id} is already claimed by another worker"
                )

    def refresh(self, campaign_id: int) -> None:
        with self._db.connection() as conn:
            cur = conn.execute(
                "UPDATE campaigns SET claim_expires_at = now() + %s "
                "WHERE id = %s AND claimed_by = %s",
                (LEASE_DURATION, campaign_id, self.worker_id),
            )
            if cur.rowcount == 0:
                raise LeaseError(
                    f"Cannot refresh lease: campaign {campaign_id} not claimed by {self.worker_id}"
                )

    def release(self, campaign_id: int) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE campaigns SET claimed_by = NULL, claim_expires_at = NULL "
                "WHERE id = %s AND claimed_by = %s",
                (campaign_id, self.worker_id),
            )


class LeaseRefresher:
    """Context manager that refreshes a lease in a background thread."""

    def __init__(self, lease: LeaseManager, campaign_id: int,
                 interval_seconds: float = REFRESH_INTERVAL.total_seconds()) -> None:
        self._lease = lease
        self._campaign_id = campaign_id
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                self._lease.refresh(self._campaign_id)
            except Exception:
                logger.warning("Lease refresh failed for campaign %s",
                             self._campaign_id, exc_info=True)

    def __enter__(self) -> LeaseRefresher:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
