# Contributing to agentune

Thanks for your interest in contributing!

## Setup

```bash
git clone https://github.com/huijokim/agentune.git
cd agentune
docker compose up -d   # Postgres + MLflow
uv sync                # install all dependencies including dev
```

## Running Tests

```bash
# Create test database (once)
PGPASSWORD=agentune psql -h localhost -U agentune -c "CREATE DATABASE agentune_test;"

# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_report.py -v
```

Tests that need Postgres will use `AGENTUNE_TEST_DB_URL` (defaults to `postgresql://agentune:agentune@localhost:5432/agentune_test`).

## Project Structure

```
src/agentune/
  backends/       # XGBoost, LightGBM, CatBoost objective functions + tuning guides
  core/           # Campaign service, DB, state machines, data models
  cli.py          # Click CLI commands
  datasets.py     # Dataset loaders (tabular + time-series via mlforecast)
  mcp_server.py   # MCP tools for Claude Code integration
  runner.py       # Round orchestration + MLflow logging
  report.py       # HTML report generation
  scheduler.py    # Budget clipping + stop conditions
  summarizer.py   # Signal extraction from Optuna studies
```

## Making Changes

1. Create a branch: `git checkout -b my-feature`
2. Write tests first when possible
3. Run the full test suite before submitting
4. Keep commits focused -- one logical change per commit

## Adding a Backend

1. Create `src/agentune/backends/my_backend.py` following the `XGBoostBackend` pattern
2. Implement the `ObjectiveBackend` protocol (see `backends/base.py`)
3. Register in `backends/__init__.py`
4. Add tests in `tests/test_backend_my_backend.py`

## Adding a Dataset

1. Add an entry to `DATASETS` in `src/agentune/datasets.py`
2. For time-series: add `"temporal": True, "file": "my_dataset.parquet"` and create a prep script in `scripts/`
3. Add tests in `tests/test_datasets_*.py`
4. Update the dataset table in `README.md`

## Pull Requests

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update README.md if adding user-facing features
- CI must pass before merge
