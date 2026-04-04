import pytest
from agent_hpo.core.db import Database
from agent_hpo.core.locking import LeaseManager, LeaseError


@pytest.fixture
def db(test_db_url):
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


@pytest.fixture
def campaign_id(db):
    """Insert a minimal campaign row and return its id."""
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO campaigns (name, metric_name, objective_direction, backend, "
            "sampler_config, initial_search_space, improvement_criteria, stop_conditions, trials_per_round) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            ("test", "accuracy", "maximize", "xgboost", "{}", "[]", "{}", "{}", 50),
        )
        return cur.fetchone()["id"]


class TestLeaseManager:
    def test_acquire_and_release(self, db, campaign_id):
        lm = LeaseManager(db, worker_id="worker-1")
        lm.acquire(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claimed_by FROM campaigns WHERE id = %s", (campaign_id,))
            assert cur.fetchone()["claimed_by"] == "worker-1"
        lm.release(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claimed_by FROM campaigns WHERE id = %s", (campaign_id,))
            assert cur.fetchone()["claimed_by"] is None

    def test_acquire_fails_if_already_claimed(self, db, campaign_id):
        lm1 = LeaseManager(db, worker_id="worker-1")
        lm2 = LeaseManager(db, worker_id="worker-2")
        lm1.acquire(campaign_id)
        with pytest.raises(LeaseError):
            lm2.acquire(campaign_id)
        lm1.release(campaign_id)

    def test_acquire_succeeds_if_lease_expired(self, db, campaign_id):
        lm1 = LeaseManager(db, worker_id="worker-1")
        lm1.acquire(campaign_id)
        with db.connection() as conn:
            conn.execute(
                "UPDATE campaigns SET claim_expires_at = now() - interval '1 minute' WHERE id = %s",
                (campaign_id,),
            )
        lm2 = LeaseManager(db, worker_id="worker-2")
        lm2.acquire(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claimed_by FROM campaigns WHERE id = %s", (campaign_id,))
            assert cur.fetchone()["claimed_by"] == "worker-2"
        lm2.release(campaign_id)

    def test_refresh_extends_lease(self, db, campaign_id):
        lm = LeaseManager(db, worker_id="worker-1")
        lm.acquire(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claim_expires_at FROM campaigns WHERE id = %s", (campaign_id,))
            first_expiry = cur.fetchone()["claim_expires_at"]
        lm.refresh(campaign_id)
        with db.connection() as conn:
            cur = conn.execute("SELECT claim_expires_at FROM campaigns WHERE id = %s", (campaign_id,))
            second_expiry = cur.fetchone()["claim_expires_at"]
        assert second_expiry > first_expiry
        lm.release(campaign_id)
