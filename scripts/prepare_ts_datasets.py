"""One-time script to prepare time-series datasets as parquet files.

Requires Kaggle API credentials (~/.kaggle/kaggle.json).
Run: uv run python scripts/prepare_ts_datasets.py
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import zipfile

import numpy as np
import pandas as pd


def prepare_store_sales(output_path: str) -> None:
    """Download and prepare Corporacion Favorita Store Sales dataset.

    Source: kaggle competitions download -c store-sales-time-series-forecasting
    We keep only the train.csv, subsample stores, and produce a flat parquet.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "store-sales-time-series-forecasting",
             "-f", "train.csv.zip",
             "-p", tmpdir],
            check=True,
        )
        zip_path = os.path.join(tmpdir, "train.csv.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        df = pd.read_csv(os.path.join(tmpdir, "train.csv"), parse_dates=["date"])

    # Subsample: keep 5 stores to get ~50K rows
    top_stores = df.groupby("store_nbr")["sales"].sum().nlargest(5).index.tolist()
    df = df[df["store_nbr"].isin(top_stores)].copy()

    # Aggregate to store-level daily sales (sum across families)
    df = df.groupby(["store_nbr", "date"]).agg({"sales": "sum"}).reset_index()
    df = df.rename(columns={"store_nbr": "unique_id", "date": "ds", "sales": "y"})
    df["unique_id"] = df["unique_id"].astype(str)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    print(f"Store Sales: {len(df)} rows, {df['unique_id'].nunique()} stores")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


def prepare_rossmann(output_path: str) -> None:
    """Download and prepare Rossmann Store Sales dataset.

    Source: kaggle competitions download -c rossmann-store-sales
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", "rossmann-store-sales",
             "-f", "train.csv.zip",
             "-p", tmpdir],
            check=True,
        )
        zip_path = os.path.join(tmpdir, "train.csv.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        df = pd.read_csv(os.path.join(tmpdir, "train.csv"), parse_dates=["Date"])

    # Filter to open days with sales > 0
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Subsample: top 5 stores by total sales
    top_stores = df.groupby("Store")["Sales"].sum().nlargest(5).index.tolist()
    df = df[df["Store"].isin(top_stores)].copy()

    df = df.rename(columns={"Store": "unique_id", "Date": "ds", "Sales": "y"})
    df["unique_id"] = df["unique_id"].astype(str)

    # Keep exogenous features
    df = df[["unique_id", "ds", "y", "DayOfWeek", "Promo", "SchoolHoliday"]].copy()
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    print(f"Rossmann: {len(df)} rows, {df['unique_id'].nunique()} stores")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    prepare_store_sales("data/store_sales.parquet")
    prepare_rossmann("data/rossmann.parquet")
    print("Done!")
