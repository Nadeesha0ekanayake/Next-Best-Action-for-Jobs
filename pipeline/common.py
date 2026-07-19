"""Shared config, paths, and helpers for the NBA pipeline (sanitised example).

The original ran on Databricks against warehouse tables. This version reads a
SYNTHETIC dataset from local CSVs so the whole pipeline runs end-to-end with no
warehouse access and no real data. Production table names are documented as
generic placeholders below (they do not point at any real system).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed "as of" date so the pipeline is fully reproducible.
TODAY = datetime(2026, 3, 23)
TODAY_STR = "2026-03-23"

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
CHARTS = ROOT / "charts"

# In production, stage 01 reads these sources from the data warehouse.
# Names below are GENERIC PLACEHOLDERS — swap for your own tables to run for real.
SOURCE_TABLES = {
    "jobs":         "your_catalog.silver.jobs_changed",
    "invoices":     "your_catalog.silver.invoices_changed",
    "quotes":       "your_catalog.silver.quotes_changed_events",
    "appointments": "your_catalog.silver.appointments_changed_events",
    "estimates":    "your_catalog.silver.estimates_changed_events",
    "fe_actions":   "your_catalog.gold.accounts_and_events_modular",
}


def load_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv")


def load_interim(name: str) -> pd.DataFrame:
    return pd.read_csv(INTERIM / f"{name}.csv")


def save_interim(df: pd.DataFrame, name: str) -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM / f"{name}.csv", index=False)
    print(f"  saved interim/{name}.csv ({len(df):,} rows)")


def days_ago_str(days: float) -> str:
    """A 'YYYY-MM-DD' date string `days` before TODAY."""
    return (TODAY - timedelta(days=float(days))).strftime("%Y-%m-%d")


def safe_days_since(date_series, reference=TODAY):
    """Vectorised: days between each date in a series and the reference date."""
    parsed = pd.to_datetime(date_series, errors="coerce", utc=True).dt.tz_localize(None)
    return (reference - parsed).dt.days
