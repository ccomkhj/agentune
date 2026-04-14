import pytest
from agentune.core.db import Database


@pytest.fixture
def db(test_db_url):
    """Uses the test_db_url fixture from conftest.py."""
    database = Database(test_db_url)
    database.setup_schema()
    yield database
    database.close()


class TestDatabase:
    def test_setup_schema_creates_tables(self, db):
        with db.connection() as conn:
            cur = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN ('campaigns', 'study_rounds', 'agent_decisions') "
                "ORDER BY table_name"
            )
            tables = [row["table_name"] for row in cur.fetchall()]
        assert tables == ["agent_decisions", "campaigns", "study_rounds"]

    def test_setup_schema_is_idempotent(self, db):
        db.setup_schema()
        db.setup_schema()
        with db.connection() as conn:
            cur = conn.execute(
                "SELECT count(*) as cnt FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'campaigns'"
            )
            assert cur.fetchone()["cnt"] == 1
