"""Database connection pool and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.rows import dict_row

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS campaigns (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    state           TEXT NOT NULL DEFAULT 'CREATED',
    pause_requested BOOLEAN NOT NULL DEFAULT false,
    metric_name     TEXT NOT NULL,
    objective_direction TEXT NOT NULL,
    backend         TEXT NOT NULL,
    sampler_config  JSONB NOT NULL,
    initial_search_space JSONB NOT NULL,
    improvement_criteria JSONB NOT NULL,
    stop_conditions JSONB NOT NULL,
    trials_per_round INT NOT NULL,
    claimed_by      TEXT,
    claim_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS study_rounds (
    id              SERIAL PRIMARY KEY,
    campaign_id     INT NOT NULL REFERENCES campaigns(id),
    round_number    INT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'PROPOSED',
    failed_from     TEXT,
    parent_round_id INT REFERENCES study_rounds(id),
    optuna_study_name TEXT NOT NULL,
    search_space    JSONB NOT NULL,
    budget          INT NOT NULL,
    trial_offset    INT NOT NULL,
    trial_end       INT,
    summary         JSONB,
    summary_schema_version INT,
    retry_count     INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(campaign_id, round_number)
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    id              SERIAL PRIMARY KEY,
    campaign_id     INT NOT NULL REFERENCES campaigns(id),
    round_id        INT NOT NULL REFERENCES study_rounds(id),
    action          TEXT NOT NULL,
    justification   TEXT NOT NULL,
    proposed_search_space JSONB,
    proposed_budget INT,
    reference_round_ids JSONB NOT NULL,
    accepted        BOOLEAN NOT NULL,
    rejection_reason TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Database:
    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: psycopg.Connection | None = None

    @property
    def optuna_storage_url(self) -> str:
        """Return the DB URL formatted for Optuna's RDBStorage."""
        return self._db_url

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._db_url, row_factory=dict_row)
        return self._conn

    @contextmanager
    def connection(self) -> Generator[psycopg.Connection, None, None]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def setup_schema(self) -> None:
        with self.connection() as conn:
            conn.execute(SCHEMA_SQL)

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
