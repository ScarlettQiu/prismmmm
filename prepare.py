"""Data loading and preprocessing for auto-MMM.

Supports three data sources via config.json:
  - CSV:       {"source": "csv",       "data_path": "./data.csv"}
  - BigQuery:  {"source": "bigquery",  "bigquery_query": "SELECT ...", "bigquery_project": "my-proj"}
  - GSheet:    {"source": "gsheet",    "gsheet_id": "1BxiM...", "gsheet_sheet": "Sheet1"}
  - (default)  no "source" key → reads "data_path" as CSV (backwards compatible)

Run discover.py first to auto-generate config.json and metadata.json from any source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_config(path: str = "config.json") -> dict:
    return json.loads(Path(path).read_text())


def _load_csv(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["data_path"]).expanduser()
    return pd.read_csv(path)


def _load_bigquery(cfg: dict) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
    except ImportError:
        print("ERROR: google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        sys.exit(1)
    project = cfg.get("bigquery_project")
    query   = cfg["bigquery_query"]
    client  = bigquery.Client(project=project)
    print(f"  Querying BigQuery: {query[:80]}...")
    return client.query(query).to_dataframe()


def _load_gsheet(cfg: dict) -> pd.DataFrame:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("ERROR: gspread not installed. Run: pip install gspread google-auth")
        sys.exit(1)
    sheet_id   = cfg["gsheet_id"]
    sheet_name = cfg.get("gsheet_sheet", "Sheet1")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds  = Credentials.from_service_account_file("service_account.json", scopes=scopes)
    gc     = gspread.authorize(creds)
    ws     = gc.open_by_key(sheet_id).worksheet(sheet_name)
    print(f"  Reading Google Sheet: {sheet_id} / {sheet_name}")
    return pd.DataFrame(ws.get_all_records())


def load_raw(cfg: dict) -> pd.DataFrame:
    source = cfg.get("source", "csv")
    if source == "bigquery":
        df = _load_bigquery(cfg)
    elif source == "gsheet":
        df = _load_gsheet(cfg)
    else:
        df = _load_csv(cfg)

    # Parse date
    df["date"] = pd.to_datetime(df[cfg["date_column"]], format=cfg.get("date_format"))
    df = df.sort_values("date").reset_index(drop=True)
    return df


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int) -> np.ndarray:
    """Apply geometric adstock: carryover effect with exponential decay."""
    result = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag in range(min(t + 1, max_lag + 1)):
            result[t] += x[t - lag] * (decay ** lag)
    return result


def hill_saturation(x: np.ndarray, slope: float, ec: float) -> np.ndarray:
    """Apply Hill saturation function: diminishing returns."""
    x_scaled = x / (x.max() + 1e-8)
    return x_scaled ** slope / (x_scaled ** slope + ec ** slope)


def preprocess(cfg: dict, adstock_decays: dict | None = None) -> pd.DataFrame:
    """Load, clean, apply transforms, return model-ready DataFrame."""
    df = load_raw(cfg)

    channels = cfg["media_channels"]
    kpi = cfg["kpi_column"]
    controls = cfg["control_variables"]
    max_lag = cfg["adstock_max_lag"]
    slope = cfg["hill_slope"]
    ec = cfg["hill_ec"]

    # Default adstock decay per channel (can be tuned)
    default_decays = {
        "TV": 0.7,
        "Digital": 0.3,
        "Sponsorship": 0.5,
        "Content.Marketing": 0.4,
        "Online.marketing": 0.3,
        "Affiliates": 0.2,
        "SEM": 0.2,
        "Radio": 0.5,
    }
    decays = {**default_decays, **(adstock_decays or {})}

    out = pd.DataFrame()
    out["date"] = df["date"]
    out["kpi"] = df[kpi].values.astype(float)

    # Apply adstock then Hill saturation per channel
    for ch in channels:
        raw = df[ch].fillna(0).values.astype(float)
        adstocked = geometric_adstock(raw, decays.get(ch, 0.4), max_lag)
        out[f"{ch}_adstock"] = adstocked
        out[f"{ch}_saturated"] = hill_saturation(adstocked, slope, ec)

    # Control variables (normalised)
    for ctrl in controls:
        if ctrl in df.columns:
            vals = df[ctrl].fillna(df[ctrl].median()).values.astype(float)
            out[ctrl] = (vals - vals.mean()) / (vals.std() + 1e-8)

    # Trend + seasonality
    out["trend"] = np.arange(len(out)) / len(out)
    out["month_num"] = df["date"].dt.month

    return out


def train_test_split(df: pd.DataFrame, holdout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-holdout].copy(), df.iloc[-holdout:].copy()


def summary(df: pd.DataFrame, cfg: dict) -> None:
    channels = cfg["media_channels"]
    print(f"\n{'='*60}")
    print(f"Dataset: {len(df)} periods  |  KPI: {cfg['kpi_column']}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nKPI stats:")
    print(f"  Mean: {df['kpi'].mean():,.0f}  |  Std: {df['kpi'].std():,.0f}")
    print(f"\nMedia spend (adstocked):")
    for ch in channels:
        col = f"{ch}_adstock"
        if col in df.columns:
            print(f"  {ch:25s}: mean={df[col].mean():12,.0f}  max={df[col].max():12,.0f}")
    print("="*60)


if __name__ == "__main__":
    cfg = load_config()
    df = preprocess(cfg)
    summary(df, cfg)
    print("\nFirst 3 rows:")
    print(df[["date", "kpi"] + [f"{ch}_saturated" for ch in cfg["media_channels"]]].head(3).to_string())
