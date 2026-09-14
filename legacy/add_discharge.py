"""
add_discharge.py
================
One-shot script: adds surma_discharge column to real_training_data_v3.csv
using the Open-Meteo GloFAS Flood API (free, no authentication required).

Fetches the 7-day mean river discharge ending on each event date.
Takes ~2-3 minutes for 101 events (no GEE required).

Usage:
    python add_discharge.py
"""

import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT     = Path(__file__).parent
CSV_PATH = ROOT / "data" / "real_training_data_v3.csv"
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
DEFAULT_Q = 20.0   # pre-monsoon dry-season baseline (m³/s)


def fetch_discharge(date_str: str, days: int = 7) -> float:
    end_d   = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_d = end_d - timedelta(days=days)
    try:
        r = requests.get(
            FLOOD_API,
            params={
                "latitude":  24.87,
                "longitude": 91.40,
                "daily":     "river_discharge",
                "start_date": str(start_d),
                "end_date":   str(end_d),
            },
            timeout=15,
        )
        vals = [v for v in r.json()["daily"]["river_discharge"] if v is not None]
        return round(float(sum(vals) / len(vals)), 2) if vals else DEFAULT_Q
    except Exception:
        return DEFAULT_Q


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} not found. Run collect_real_data_v3.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    print(f"Adding surma_discharge to {total} rows in {CSV_PATH.name} ...")
    print(f"API: Open-Meteo GloFAS — 7-day mean discharge at Sunamganj (24.87°N, 91.40°E)\n")

    discharges = []
    for i, row in df.iterrows():
        date_str = str(row["date"])
        q = fetch_discharge(date_str)
        discharges.append(q)
        label = "FLOOD" if row["flood_label"] == 1 else "dry"
        print(f"  [{i+1:3d}/{total}] {date_str}  {label:5s}  discharge = {q:6.1f} m³/s")
        time.sleep(0.3)   # be polite to the API

    df["surma_discharge"] = discharges
    df.to_csv(CSV_PATH, index=False)

    # Quick sanity check
    flood_q = df[df["flood_label"] == 1]["surma_discharge"].mean()
    dry_q   = df[df["flood_label"] == 0]["surma_discharge"].mean()
    print(f"\nDone. {CSV_PATH.name} updated.")
    print(f"  Mean discharge — FLOOD events: {flood_q:.1f} m³/s")
    print(f"  Mean discharge — DRY events:   {dry_q:.1f} m³/s")
    print(f"  Separation (flood − dry):       {flood_q - dry_q:+.1f} m³/s")
    print()
    print("Next step:")
    print("  python train_honest.py --data real_training_data_v3.csv --real-only")


if __name__ == "__main__":
    main()
