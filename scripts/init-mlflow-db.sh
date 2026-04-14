#!/bin/bash
# Creates the mlflow database if it doesn't exist.
# Mounted into /docker-entrypoint-initdb.d/ so Postgres runs it on first start.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
EOSQL
