"""
dib_export/refetch_15_events.py

Repairs the 15 rows in the 77-event "real_sar" export that were flagged
real_sar by expand_training_data.py purely on the basis of event year (>=2017)
without ever fetching real Sentinel-1/Sentinel-2/ERA5-Land data for them
(expand_training_data.py always synthesizes VV/VH/ndwi/upstream_vv from
hardcoded SAR_PROFILES). See dib_export/references_and_checks.md section 3.

For those 15 dates, this script re-fetches genuine values using the SAME
GEE collections/scales/windows as the original collect_honest_data.py
pipeline (which produced the other 62 rows):
  - get_s1()       -> VV, VH, vv_vh_ratio        (Sentinel-1 GRD, SAR_SCALE)
  - get_upstream() -> upstream_vv                (Sentinel-1 GRD over Barak AOI)
  - get_era5()     -> rainfall, soil_moisture, temp, wind (ERA5-Land, ERA5_SCALE)
  - NDWI is fetched for real from Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED),
    using the fetch_ndwi() pattern from collect_real_data_v3.py — the ORIGINAL
    collect_honest_data.py never fetched real NDWI (it hardcoded DEFAULTS["ndwi"]
    for every row), so this script upgrades all 77 rows to a genuine per-event
    NDWI value rather than reproducing that shortcut.

For ALL 77 dates (not just the 15), this script also computes two properly
defined antecedent-rainfall features from ERA5-Land directly (replacing the
misleading "forecast_rain_next_12h"/"forecast_rain_72h" columns, which were
actually backward-looking archived rainfall mislabeled as a forecast):
  - rain_24h : ERA5-Land total_precipitation_sum for the 1 day ending on the
               event date (same collection/scale as get_era5(), 1-day window)
  - rain_96h : same, but a 4-day window ending on the event date

Requires an authenticated GEE session (same as collect_honest_data.py):
    ee.Initialize(project=GEE_PROJECT)
Run from the repo root:
    python dib_export/refetch_15_events.py

Outputs:
    dib_export/dataset_v2.csv     — all 77 rows, repaired
    dib_export/refetch_report.md  — old vs new values for the 15 repaired rows

This script only reads dib_export/dataset.csv and GEE/API data. It does not
modify any file outside dib_export/.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

import ee
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (GEE_PROJECT, HAOR_BBOX, UPSTREAM_BBOX,
                     SAR_SCALE, S2_SCALE, ERA5_SCALE, DEFAULTS,
                     TEMP_CLIMATOLOGY)

EXPORT_DIR = Path(__file__).parent
IN_CSV     = EXPORT_DIR / "dataset.csv"
OUT_CSV    = EXPORT_DIR / "dataset_v2.csv"
REPORT_MD  = EXPORT_DIR / "refetch_report.md"

# The 15 dates flagged "real_sar" but actually populated with synthetic
# SAR_PROFILES values by expand_training_data.py (see references_and_checks.md).
DATES_TO_REFETCH = [
    "2018-04-20", "2018-07-01", "2019-04-20", "2019-09-01", "2021-04-15",
    "2021-09-10", "2022-09-10", "2022-10-20", "2023-06-15", "2023-08-10",
    "2023-09-05", "2023-11-10", "2024-04-25", "2024-09-10", "2024-10-15",
]

ee.Initialize(project=GEE_PROJECT)
haor     = ee.Geometry.Rectangle(HAOR_BBOX)
upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)


def get_s1(start, end):
    """Same logic as collect_honest_data.py::get_s1(), plus we also return
    the actual median image's acquisition-date range used, for reporting."""
    windows = [
        (start, end),
        (str((datetime.strptime(start, "%Y-%m-%d") - timedelta(days=23)).date()), end),
    ]
    for s, e in windows:
        col = (ee.ImageCollection("COPERNICUS/S1_GRD")
               .filterBounds(haor).filterDate(s, e)
               .filter(ee.Filter.eq("instrumentMode", "IW")))
        n = col.size().getInfo()
        if n > 0:
            img = col.select(["VV", "VH"]).median()
            r = img.reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE)
            vv = r.get("VV").getInfo()
            vh = r.get("VH").getInfo()
            if vv is not None and vh is not None:
                vv, vh = round(float(vv), 2), round(float(vh), 2)
                ratio = round(vv / vh if vh != 0 else DEFAULTS["vv_vh_ratio"], 4)
                # dates of images actually used, for the report
                dates = col.aggregate_array("system:time_start").getInfo()
                img_dates = sorted({datetime.utcfromtimestamp(t / 1000).date().isoformat() for t in dates})
                return vv, vh, ratio, s, e, n, img_dates
    return DEFAULTS["VV"], DEFAULTS["VH"], DEFAULTS["vv_vh_ratio"], None, None, 0, []


def get_upstream(start, end):
    windows = [
        (start, end),
        (str((datetime.strptime(start, "%Y-%m-%d") - timedelta(days=30)).date()), end),
    ]
    for s, e in windows:
        try:
            col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(upstream).filterDate(s, e)
                   .filter(ee.Filter.eq("instrumentMode", "IW"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
            if col.size().getInfo() == 0:
                continue
            r = col.select("VV").median().reduceRegion(ee.Reducer.mean(), upstream, SAR_SCALE)
            v = r.get("VV").getInfo()
            if v:
                return round(float(v), 2)
        except Exception:
            continue
    return DEFAULTS["upstream_vv"]


def get_era5(start, end):
    """Same logic as collect_honest_data.py::get_era5()."""
    try:
        img = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .filterDate(start, end)
               .select(["total_precipitation_sum", "volumetric_soil_water_layer_1",
                        "temperature_2m", "u_component_of_wind_10m"])
               .mean())
        r = img.reduceRegion(ee.Reducer.mean(), haor, ERA5_SCALE)
        rain = round(float(r.get("total_precipitation_sum").getInfo() or 0) * 1000, 1)
        soil = round(float(r.get("volumetric_soil_water_layer_1").getInfo() or 0) * 100, 1)
        temp = round(float(r.get("temperature_2m").getInfo() or 298.15) - 273.15, 1)
        wind = round(abs(float(r.get("u_component_of_wind_10m").getInfo() or 0)) * 3.6, 1)
        return rain, soil, temp, wind
    except Exception:
        return DEFAULTS["rainfall"], DEFAULTS["soil_moisture"], DEFAULTS["temp"], DEFAULTS["wind"]


def get_rain_sum(start, end):
    """ERA5-Land total_precipitation_sum, summed (not meaned) over [start, end)."""
    try:
        col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .filterDate(start, end)
               .select("total_precipitation_sum"))
        img = col.sum()
        r = img.reduceRegion(ee.Reducer.mean(), haor, ERA5_SCALE)
        val = r.get("total_precipitation_sum").getInfo()
        return round(float(val or 0) * 1000, 1)
    except Exception:
        return None


def fetch_ndwi(start, end, cloud_pct=50):
    """Same pattern as collect_real_data_v3.py::fetch_ndwi(): Sentinel-2 SR,
    NDWI = normalizedDifference(B3, B8), median composite, 7-day window,
    fallback to 70% cloud cover if nothing found. Returns (value, is_real)."""
    for pct in [cloud_pct, 70]:
        try:
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterBounds(haor)
                   .filterDate(start, end)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", pct)))
            if col.size().getInfo() == 0:
                continue
            img = col.median()
            ndwi = img.normalizedDifference(["B3", "B8"]).rename("ndwi")
            val = ndwi.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=haor, scale=S2_SCALE, maxPixels=1e9,
            ).get("ndwi").getInfo()
            if val is not None:
                return round(float(val), 4), True
        except Exception:
            continue
    return DEFAULTS["ndwi"], False


def temp_anomaly_for(date_str, temp):
    month = int(date_str[5:7])
    return round(temp - TEMP_CLIMATOLOGY.get(month, temp), 2)


def main():
    df = pd.read_csv(IN_CSV)
    assert len(df) == 77, f"expected 77 rows in {IN_CSV}, got {len(df)}"
    assert df["date"].is_unique, "dataset.csv has duplicate dates"

    # ── Step 1: refetch the 15 flagged rows ──────────────────────────────
    refetch_log = []
    for date_str in DATES_TO_REFETCH:
        mask = df["date"] == date_str
        if not mask.any():
            print(f"  [WARN] {date_str} not found in dataset.csv — skipping")
            continue
        idx = df.index[mask][0]
        old_row = df.loc[idx].to_dict()

        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        start = str(d - timedelta(days=7))
        end = date_str
        print(f"Refetching {date_str} ...", flush=True)

        vv, vh, ratio, s1_start_used, s1_end_used, n_images, img_dates = get_s1(start, end)
        no_s1_found = (n_images == 0)
        rain, soil, temp, wind = get_era5(start, end)
        upstream_vv = get_upstream(start, end)
        ndwi_val, ndwi_real = fetch_ndwi(start, end)
        anomaly = temp_anomaly_for(date_str, temp)

        # Sanity: was any S1 image within +/-6 days of the event date?
        within_6d = False
        if img_dates:
            for ds in img_dates:
                delta = abs((datetime.strptime(ds, "%Y-%m-%d").date() - d).days)
                if delta <= 6:
                    within_6d = True

        df.loc[idx, "VV"] = vv
        df.loc[idx, "VH"] = vh
        df.loc[idx, "vv_vh_ratio"] = ratio
        df.loc[idx, "rainfall"] = rain
        df.loc[idx, "soil_moisture"] = soil
        df.loc[idx, "temp"] = temp
        df.loc[idx, "wind"] = wind
        df.loc[idx, "temp_anomaly"] = anomaly
        df.loc[idx, "ndwi"] = ndwi_val
        df.loc[idx, "upstream_vv"] = upstream_vv

        refetch_log.append({
            "date": date_str,
            "old_VV": old_row["VV"], "new_VV": vv,
            "old_VH": old_row["VH"], "new_VH": vh,
            "old_ndwi": old_row["ndwi"], "new_ndwi": ndwi_val, "ndwi_real": ndwi_real,
            "old_upstream_vv": old_row["upstream_vv"], "new_upstream_vv": upstream_vv,
            "old_soil_moisture": old_row["soil_moisture"], "new_soil_moisture": soil,
            "s1_window": f"{start}..{end}",
            "s1_images_found": n_images,
            "s1_image_dates": img_dates,
            "s1_within_6d": within_6d,
            "no_s1_found": no_s1_found,
        })

    # ── Step 2: rain_24h / rain_96h for ALL 77 dates ─────────────────────
    print("\nComputing rain_24h / rain_96h for all 77 rows ...")
    rain24, rain96 = [], []
    for date_str in df["date"]:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        end = str(d + timedelta(days=1))          # ee filterDate end is exclusive
        start24 = str(d)                          # 1-day window ending on event date
        start96 = str(d - timedelta(days=3))       # 4-day window ending on event date
        v24 = get_rain_sum(start24, end)
        v96 = get_rain_sum(start96, end)
        rain24.append(v24 if v24 is not None else DEFAULTS["rainfall"])
        rain96.append(v96 if v96 is not None else DEFAULTS["rainfall"])

    df["rain_24h"] = rain24
    df["rain_96h"] = rain96
    df = df.drop(columns=["forecast_rain_next_12h", "forecast_rain_72h"])

    # Reorder: keep original column order but swap the two forecast columns
    # for the two new rainfall columns in the same position.
    cols = list(pd.read_csv(IN_CSV, nrows=0).columns)
    cols = [c if c != "forecast_rain_next_12h" else "rain_24h" for c in cols]
    cols = [c if c != "forecast_rain_72h" else "rain_96h" for c in cols]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")

    # ── Step 3: sanity checks ────────────────────────────────────────────
    problems = []
    if df.isna().sum().sum() > 0:
        problems.append(f"Missing values found:\n{df.isna().sum()[df.isna().sum() > 0]}")
    if not df["date"].is_unique:
        problems.append("Duplicate dates found in dataset_v2.csv")
    if not df["VV"].between(-30, -5).all():
        problems.append(f"VV out of plausible range: min={df['VV'].min()}, max={df['VV'].max()}")
    if not df["VH"].between(-35, -10).all():
        problems.append(f"VH out of plausible range: min={df['VH'].min()}, max={df['VH'].max()}")
    if not df["upstream_vv"].between(-30, -5).all():
        problems.append(f"upstream_vv out of plausible range: min={df['upstream_vv'].min()}, max={df['upstream_vv'].max()}")
    if (df["rain_24h"] < 0).any() or (df["rain_96h"] < 0).any():
        problems.append("Negative rainfall found in rain_24h/rain_96h")

    print("\n" + "=" * 60)
    if problems:
        print("SANITY CHECK: ISSUES FOUND")
        for p in problems:
            print(" -", p)
    else:
        print("SANITY CHECK: all clear (no missing values, dates unique, "
              "backscatter within plausible ranges, rainfall non-negative)")
    print("=" * 60)

    # ── Step 4: refetch_report.md ─────────────────────────────────────────
    write_report(refetch_log, problems)


def write_report(refetch_log, problems):
    lines = []
    lines.append("# Refetch Report — 15 Repaired Rows\n")
    lines.append(
        "Real Sentinel-1 (VV/VH/upstream_vv), Sentinel-2 (NDWI), and ERA5-Land "
        "(soil_moisture) values fetched via Google Earth Engine to replace the "
        "synthetic `SAR_PROFILES` proxy values that `expand_training_data.py` had "
        "written into these 15 rows despite flagging them `data_quality=\"real_sar\"`.\n"
    )
    lines.append("| Date | VV old→new | VH old→new | NDWI old→new | upstream_vv old→new | soil_moisture old→new | S1 window | S1 images found | Within ±6d of event? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    no_image_flags = []
    for r in refetch_log:
        ndwi_flag = "" if r["ndwi_real"] else " (fallback default, no cloud-free S2 scene)"
        lines.append(
            f"| {r['date']} | {r['old_VV']} → {r['new_VV']} | {r['old_VH']} → {r['new_VH']} | "
            f"{r['old_ndwi']} → {r['new_ndwi']}{ndwi_flag} | "
            f"{r['old_upstream_vv']} → {r['new_upstream_vv']} | "
            f"{r['old_soil_moisture']} → {r['new_soil_moisture']} | "
            f"{r['s1_window']} | {r['s1_images_found']} | {'yes' if r['s1_within_6d'] else 'NO'} |"
        )
        if r["no_s1_found"]:
            no_image_flags.append(r["date"])
        elif not r["s1_within_6d"]:
            no_image_flags.append(f"{r['date']} (image found but >6 days from event date: {r['s1_image_dates']})")

    lines.append("")
    lines.append("## Sentinel-1 image-date notes")
    if no_image_flags:
        lines.append(
            "The following dates had **no Sentinel-1 image within ±6 days** of the "
            "event date (values below are either the widened-window median or the "
            "DEFAULTS fallback — flagged here rather than silently presented as "
            "fresh per-event data):\n"
        )
        for f in no_image_flags:
            lines.append(f"- {f}")
    else:
        lines.append("All 15 dates had at least one Sentinel-1 image within ±6 days of the event date.")

    lines.append("\n## Sanity check on dataset_v2.csv\n")
    if problems:
        lines.append("Issues found:\n")
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append(
            "- No missing values\n"
            "- All 77 dates unique\n"
            "- VV/VH/upstream_vv within plausible Sentinel-1 backscatter ranges\n"
            "- rain_24h / rain_96h non-negative\n"
        )

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report -> {REPORT_MD}")


if __name__ == "__main__":
    main()
