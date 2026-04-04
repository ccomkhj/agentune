"""Lease-based concurrency control for campaign execution."""

from __future__ import annotations

import uuid
from datetime import timedelta

from agent_hpo.core.db import Database

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
