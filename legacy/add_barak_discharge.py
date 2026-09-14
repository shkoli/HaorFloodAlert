"""
add_barak_discharge.py — Backfill Barak river discharge for training data.
==========================================================================
Adds 'barak_discharge_cumecs' column to honest_training_data.csv using
GloFAS reanalysis data from the Open-Meteo Flood API.

Usage
-----
    python add_barak_discharge.py

Output
------
- Overwrites data/honest_training_data.csv with new column added
- Prints correlation with existing upstream_vv feature (multicollinearity check)

IMPORTANT BEFORE RETRAINING
----------------------------
If Pearson r(barak_discharge_cumecs, upstream_vv) > 0.70, the feature will
likely be auto-dropped by train_honest.py (zero-variance check + RF importance).
In that case, do NOT add it to config.FEATURES — use it as a monitoring
dashboard indicator only (current implementation).

To add to the ML model after running this script:
1. Check correlation output below
2. If r < 0.70: add "barak_discharge_cumecs" to config.FEATURES
3. Add "barak_discharge_cumecs": 450.0 to config.DEFAULTS
4. Run: python train_honest.py --data honest_training_data.csv
5. Compare new LOOCV accuracy with old (88.9%)
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_FILE = Path("data/honest_training_data.csv")
BARAK_LAT = 24.82
BARAK_LON = 92.79
FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} rows from {DATA_FILE}")
print(f"Columns: {list(df.columns)}")

if "barak_discharge_cumecs" in df.columns:
    print("Column 'barak_discharge_cumecs' already exists — re-fetching to update.")

def fetch_discharge_for_date(date_str: str, days_back: int = 7) -> float:
    """Fetch 7-day mean Barak discharge ending on date_str."""
    d_end   = datetime.strptime(date_str, "%Y-%m-%d").date()
    d_start = d_end - timedelta(days=days_back)
    try:
        r = requests.get(
            FLOOD_API,
            params={
                "latitude":   BARAK_LAT,
                "longitude":  BARAK_LON,
                "daily":      "river_discharge",
                "start_date": str(d_start),
                "end_date":   str(d_end),
            },
            timeout=20,
        )
        vals = [v for v in r.json().get("daily", {}).get("river_discharge", [])
                if v is not None]
        if vals:
            return round(sum(vals) / len(vals), 1)
    except Exception as exc:
        print(f"  API error for {date_str}: {exc}")
    return 450.0   # dry-season fallback


date_col = None
for c in ("date", "Date", "event_date", "date_str"):
    if c in df.columns:
        date_col = c
        break

if date_col is None:
    raise ValueError(f"No date column found. Columns: {list(df.columns)}")

print(f"\nFetching Barak discharge for {len(df)} events (date column: '{date_col}') ...")
discharges = []
for i, row in df.iterrows():
    d_str = str(row[date_col])[:10]
    val   = fetch_discharge_for_date(d_str)
    discharges.append(val)
    print(f"  [{i+1:3d}/{len(df)}]  {d_str}  →  {val:,.1f} m³/s")
    time.sleep(0.25)   # gentle rate limit

df["barak_discharge_cumecs"] = discharges

# Flood vs dry comparison
if "flood_label" in df.columns:
    flood_mean = df[df["flood_label"] == 1]["barak_discharge_cumecs"].mean()
    dry_mean   = df[df["flood_label"] == 0]["barak_discharge_cumecs"].mean()
    print(f"\nFlood mean discharge : {flood_mean:,.1f} m³/s")
    print(f"Dry   mean discharge : {dry_mean:,.1f} m³/s")
    print(f"Separation           : {flood_mean - dry_mean:+,.1f} m³/s")

# Multicollinearity check
if "upstream_vv" in df.columns:
    r_upvv = df["barak_discharge_cumecs"].corr(df["upstream_vv"])
    print(f"\nPearson r(barak_discharge_cumecs, upstream_vv): {r_upvv:.3f}")
    if abs(r_upvv) > 0.70:
        print("  ⚠️  HIGH CORRELATION — do NOT add to config.FEATURES (multicollinear)")
        print("     Use as monitoring dashboard indicator only.")
    else:
        print("  ✅  Low correlation — safe to add to config.FEATURES for retraining")
        print("     Add 'barak_discharge_cumecs' to config.FEATURES + config.DEFAULTS")
        print("     Then run: python train_honest.py")

df.to_csv(DATA_FILE, index=False)
print(f"\n[OK] Saved updated CSV → {DATA_FILE}")
print(f"     New column: barak_discharge_cumecs  ({len(df)} rows)")
