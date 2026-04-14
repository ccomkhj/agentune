import os
import pytest
import psycopg

TEST_DB_URL = os.environ.get(
    "AGENTUNE_TEST_DB_URL",
    "postgresql://agentune:agentune@localhost:5432/agentune_test",
)


@pytest.fixture(scope="session")
def test_db_url():
    """Provide test database URL."""
    base_url = TEST_DB_URL.rsplit("/", 1)[0] + "/postgres"
    db_name = TEST_DB_URL.rsplit("/", 1)[1].split("?")[0]
    try:
        with psycopg.connect(base_url, autocommit=True) as conn:
            cur = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
            )
            if not cur.fetchone():
                conn.execute(f"CREATE DATABASE {db_name}")
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL not available")
    return TEST_DB_URL


@pytest.fixture(autouse=True)
def clean_db(request):
    """Truncate all tables before each test (only when DB is used)."""
    yield
    if "test_db_url" not in request.fixturenames:
        return
    try:
        with psycopg.connect(TEST_DB_URL) as conn:
            conn.execute("TRUNCATE agent_decisions, study_rounds, campaigns CASCADE")
            conn.commit()
    except Exception:
        pass
