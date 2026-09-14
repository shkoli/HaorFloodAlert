"""
generate_synthetic_v2.py
=========================
Generates a physics-calibrated synthetic real_training_data_v2.csv
for all 45 events (2017-2025) with all 13 features in exact FEATURES order.

Use this when:
  - You haven't run collect_real_haor_data.py yet (no GEE auth / no time)
  - You want to test the full pipeline immediately

Values are calibrated from literature:
  Flood SAR:  VV ~-20 dB, VH ~-25 dB  (Uddin et al. 2019, Singha et al. 2020)
  Dry SAR:    VV ~-10 dB, VH ~-18 dB
  Flood NDWI: > 0 (positive = water)
  Dry NDWI:   < 0 (negative = vegetation/land)
  Flood soil_moisture: 55-75% (ERA5 saturated haor soils)
  Dry soil_moisture:   20-40%
  Flood TWI:  14-20 (haor bowl centre)
  Upstream VV flood signal: < -16 dB (open water upstream)

Run:  python generate_synthetic_v2.py
Then: python validate_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import FEATURES, DATA_DIR, DEFAULTS

np.random.seed(42)

OUT_FILE = DATA_DIR / "real_training_data_v2.csv"

# ── 45 events (same as collect_real_haor_data.py) ─────────────────────────────
LABELED_EVENTS = [
    # date,         label, type
    ("2017-04-28", 1, "upstream"),
    ("2017-06-01", 1, "seasonal"),
    ("2017-07-15", 1, "seasonal"),
    ("2017-08-10", 1, "seasonal"),
    ("2017-09-05", 0, "dry"),
    ("2018-03-20", 1, "upstream"),   # hard FN candidate
    ("2018-04-15", 1, "seasonal"),
    ("2018-05-20", 1, "seasonal"),
    ("2018-06-01", 1, "seasonal"),
    ("2018-07-20", 0, "dry"),
    ("2019-04-05", 1, "upstream"),
    ("2019-05-20", 1, "seasonal"),
    ("2019-06-10", 1, "seasonal"),
    ("2019-07-05", 0, "dry"),
    ("2019-08-15", 0, "dry"),        # hard FP candidate
    ("2020-06-15", 0, "dry"),
    ("2020-07-10", 0, "dry"),
    ("2020-08-05", 0, "dry"),
    ("2020-09-15", 0, "dry"),        # hard FP candidate
    ("2020-10-01", 0, "dry"),
    ("2021-05-01", 1, "upstream"),   # hard FN candidate
    ("2021-06-05", 1, "seasonal"),
    ("2021-07-01", 1, "seasonal"),
    ("2021-08-10", 1, "seasonal"),
    ("2021-09-10", 0, "dry"),
    ("2022-03-25", 1, "upstream"),
    ("2022-04-20", 1, "seasonal"),
    ("2022-05-15", 1, "seasonal"),
    ("2022-06-05", 0, "dry"),
    ("2022-07-20", 1, "seasonal"),
    ("2023-04-18", 0, "dry"),
    ("2023-05-10", 0, "dry"),
    ("2023-06-01", 0, "dry"),
    ("2023-07-01", 1, "seasonal"),
    ("2023-08-10", 0, "dry"),
    ("2024-04-15", 1, "upstream"),
    ("2024-05-12", 1, "seasonal"),
    ("2024-06-01", 1, "seasonal"),
    ("2024-07-10", 1, "seasonal"),
    ("2024-08-01", 0, "dry"),
    ("2025-04-10", 1, "upstream"),
    ("2025-05-15", 1, "seasonal"),
    ("2025-06-05", 1, "seasonal"),
    ("2025-07-20", 0, "dry"),
    ("2025-08-15", 0, "dry"),
]

assert len(LABELED_EVENTS) == 45

# Static terrain features (haor-specific constants from literature)
SLOPE_MEAN = 1.9    # degrees — very flat haor bowl (SRTM avg)
TWI_MEAN   = 15.8   # HydroSHEDS TWI haor bowl centre (12-20 range)

# ── Physics-calibrated value generator ────────────────────────────────────────

def r(lo, hi):
    """Uniform random in [lo, hi]."""
    return round(float(np.random.uniform(lo, hi)), 3)

def rn(mean, std):
    """Gaussian random."""
    return round(float(np.random.normal(mean, std)), 3)


def generate_row(date_str: str, label: int, etype: str) -> dict:
    flood = label == 1

    # ── Hard cases — deliberately ambiguous values ─────────────────────
    hard_fn = date_str in ("2018-03-20", "2021-05-01")  # upstream, low local rain
    hard_fp = date_str in ("2019-08-15", "2020-09-15")  # post-monsoon residual

    # ── SAR backscatter ────────────────────────────────────────────────
    if flood and not hard_fn:
        vv = rn(-20.0, 1.5)      # flood signal
        vh = rn(-25.0, 1.5)
    elif hard_fn:
        # Upstream flood: local SAR not as depressed (barrage not yet broken)
        vv = rn(-14.5, 1.2)      # borderline — will likely misclassify
        vh = rn(-20.5, 1.2)
    elif hard_fp:
        vv = rn(-16.5, 1.0)      # high residual moisture → looks like flood
        vh = rn(-22.0, 1.0)
    else:
        vv = rn(-10.0, 1.5)      # dry signal
        vh = rn(-18.0, 1.5)

    vv = round(vv, 2)
    vh = round(vh, 2)
    vv_vh_ratio = round(vv / vh if vh != 0 else DEFAULTS["vv_vh_ratio"], 4)

    # ── CHIRPS rainfall (7-day sum) ───────────────────────────────────
    if flood and not hard_fn:
        rainfall = r(45, 180)
    elif hard_fn:
        rainfall = r(8, 25)      # very low local rain (barrage-only event)
    else:
        rainfall = r(2, 40)

    # ── ERA5 soil moisture (%) ────────────────────────────────────────
    if flood and not hard_fn:
        soil_moisture = r(55, 78)
    elif hard_fn:
        soil_moisture = r(38, 52)  # moderate — below flood threshold
    elif hard_fp:
        soil_moisture = r(52, 68)  # high residual post-monsoon moisture
    else:
        soil_moisture = r(18, 42)

    # ── Open-Meteo temperature + wind ─────────────────────────────────
    month = int(date_str[5:7])
    base_temp = 25 + 5 * abs(month - 6) / 6  # peaks in Apr-Jun
    temp = rn(base_temp, 2.0)
    wind = r(8, 35) if flood else r(5, 20)

    # ── Slope (static) ────────────────────────────────────────────────
    slope = round(rn(SLOPE_MEAN, 0.1), 2)

    # ── Forecast proxies ──────────────────────────────────────────────
    forecast_rain_next_12h = round(rainfall * r(0.04, 0.12), 1)
    forecast_rain_72h      = round(rainfall * r(0.35, 0.65), 1)

    # ── Sentinel-2 NDWI ──────────────────────────────────────────────
    if flood and not hard_fn:
        ndwi = r(0.05, 0.45)
    elif hard_fn:
        ndwi = r(-0.10, 0.08)   # borderline
    elif hard_fp:
        ndwi = r(0.0, 0.15)     # slightly positive (residual water)
    else:
        ndwi = r(-0.45, -0.05)

    # ── TWI (static-ish, slight variation) ───────────────────────────
    twi = round(rn(TWI_MEAN, 0.8), 3)

    # ── Upstream VV (Barak river proxy) ──────────────────────────────
    if etype == "upstream":
        upstream_vv = rn(-19.5, 1.2)   # strong upstream flood signal
    elif flood:
        upstream_vv = rn(-17.0, 1.5)   # moderate upstream signal
    else:
        upstream_vv = rn(-10.5, 1.5)   # dry upstream

    upstream_vv = round(upstream_vv, 2)

    return {
        "date":                   date_str,
        "flood_label":            label,
        "event_note":             f"{etype} event",
        "VV":                     vv,
        "VH":                     vh,
        "vv_vh_ratio":            vv_vh_ratio,
        "rainfall":               round(rainfall, 1),
        "soil_moisture":          round(soil_moisture, 1),
        "temp":                   round(temp, 1),
        "wind":                   round(wind, 1),
        "slope":                  slope,
        "forecast_rain_next_12h": forecast_rain_next_12h,
        "ndwi":                   round(ndwi, 4),
        "twi":                    twi,
        "upstream_vv":            upstream_vv,
        "forecast_rain_72h":      forecast_rain_72h,
    }


def main():
    print("=" * 60)
    print("  HaorFloodAlert — Synthetic v2 Data Generator")
    print("=" * 60)
    print(f"  Generating {len(LABELED_EVENTS)} events with 13 features ...")

    rows = [generate_row(d, l, t) for d, l, t in LABELED_EVENTS]
    df   = pd.DataFrame(rows)

    # Enforce column order: meta + FEATURES
    meta_cols = ["date", "flood_label", "event_note"]
    df = df[meta_cols + FEATURES]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    n_flood = int(df["flood_label"].sum())
    n_dry   = len(df) - n_flood

    print(f"  ✓ Saved {len(df)} rows → {OUT_FILE}")
    print(f"    Flood events : {n_flood}")
    print(f"    Dry events   : {n_dry}")
    print(f"    Columns      : {list(df.columns)}")
    print()
    print("  Feature means (flood vs dry):")
    print(f"  {'Feature':<25} {'Flood mean':>12} {'Dry mean':>10}")
    print(f"  {'─'*50}")
    for f in FEATURES:
        fm = df[df["flood_label"]==1][f].mean()
        dm = df[df["flood_label"]==0][f].mean()
        print(f"  {f:<25} {fm:>12.3f} {dm:>10.3f}")

    print()
    print("  ✓ Done.  Now run:")
    print("      python validate_models.py")
    print()
    print("  When ready for REAL GEE data, run:")
    print("      python collect_real_haor_data.py")
    print("  (This will overwrite this file with actual satellite data)")


if __name__ == "__main__":
    main()
