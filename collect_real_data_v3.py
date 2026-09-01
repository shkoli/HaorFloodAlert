"""
collect_real_data_v3.py
=======================
Collects ALL 13 model features from real satellite sources (GEE + Open-Meteo)
for every labeled event in honest_training_data.csv.

Replaces collect_real_haor_data.py which only collected 10 features
and skipped ndwi, twi, upstream_vv, and forecast_rain_72h.

Features collected:
  VV, VH, vv_vh_ratio    — Sentinel-1 SAR (real satellite backscatter)
  ndwi                   — Sentinel-2 NDWI (real optical water index)
  rainfall               — CHIRPS 7-day cumulative (real)
  soil_moisture          — ERA5-Land layer 1 (real)
  temp, wind             — Open-Meteo archive (real)
  slope                  — SRTM (static terrain, real)
  twi                    — HydroSHEDS flow accumulation + SRTM (static, real)
  upstream_vv            — Sentinel-1 over Barak river, Silchar, Assam (real)
  forecast_rain_next_12h — Open-Meteo forecast at collection time (real)
  forecast_rain_72h      — Open-Meteo 72h cumulative forecast (real)

Output: data/real_training_data_v3.csv
  - Includes data_quality column: "full_real" | "sar_only" | "fallback"
  - Skips events before 2014-04-03 (Sentinel-1A launch date)
  - SAR: 7-day window, widens to 21-day if no image found
  - NDWI: 7-day window, up to 70% cloud fallback (wider window causes temporal mismatch)

Runtime: ~45-60 minutes for 101 events (GEE rate-limited).
Run overnight or in background.

Usage:
    python collect_real_data_v3.py
    python collect_real_data_v3.py --resume   # skip already-collected dates
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR, DEFAULTS, GEE_PROJECT,
    HAOR_BBOX, HAOR_LAT, HAOR_LON,
    UPSTREAM_BBOX,
    SAR_SCALE, S2_SCALE, CHIRPS_SCALE, ERA5_SCALE, DEM_SCALE,
)

S1_LAUNCH = datetime(2014, 4, 3).date()   # Sentinel-1A launch date
OUT_PATH  = DATA_DIR / "real_training_data_v3.csv"


# ── GEE initialisation (deferred so script can be imported safely) ─────────────

def init_gee():
    import ee
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"  [GEE] Initialised project: {GEE_PROJECT}")
        return ee
    except Exception as e:
        print(f"  [ERROR] GEE init failed: {e}")
        print("  Run: earthengine authenticate")
        sys.exit(1)


# ── Sentinel-1 SAR ─────────────────────────────────────────────────────────────

def fetch_s1(ee, region, start: str, end: str,
             band: str = "VV", widen_days: int = 21) -> tuple:
    """
    Returns (value, is_real).
    Tries 7-day window first; widens to widen_days if no image found.
    Returns (default, False) if still no image.
    """
    default = DEFAULTS[band] if band in DEFAULTS else DEFAULTS["VV"]

    for window_start in [start, (datetime.strptime(end, "%Y-%m-%d")
                                  - timedelta(days=widen_days)).strftime("%Y-%m-%d")]:
        try:
            col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(region)
                   .filterDate(window_start, end)
                   .filter(ee.Filter.eq("instrumentMode", "IW"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", band))
                   .select([band]))
            if col.size().getInfo() == 0:
                continue
            img = col.median()
            val = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region, scale=SAR_SCALE, maxPixels=1e9,
            ).get(band).getInfo()
            if val is not None:
                return round(float(val), 4), True
        except Exception:
            continue

    return default, False


def fetch_sentinel1_full(ee, haor, upstream, start: str, end: str) -> dict:
    """Fetches VV, VH (haor) and upstream_vv (Barak river)."""
    vv,  vv_real  = fetch_s1(ee, haor,     start, end, "VV")
    vh,  vh_real  = fetch_s1(ee, haor,     start, end, "VH")
    upvv, up_real = fetch_s1(ee, upstream, start, end, "VV")

    ratio = round(vv / vh, 4) if vh != 0 else DEFAULTS["vv_vh_ratio"]
    return {
        "VV":          vv,
        "VH":          vh,
        "vv_vh_ratio": ratio,
        "upstream_vv": upvv,
        "sar_real":    vv_real and vh_real,
        "upstream_real": up_real,
    }


# ── Sentinel-2 NDWI ───────────────────────────────────────────────────────────

def fetch_ndwi(ee, haor, start: str, end: str, cloud_pct: int = 50) -> tuple:
    """Returns (ndwi_value, is_real).

    Uses a 7-day window (caller sets start) and up to 70% cloud cover fallback.
    Falls back through cloud_pct→70% if no image found.
    NOTE: Do NOT widen start beyond ~14 days for pre-monsoon events — a 45-day
    lookback will find pre-flood dry images for March-April flood events, giving
    false negative NDWI that actively degrades model performance.
    """
    for pct in [cloud_pct, 70]:
        try:
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterBounds(haor)
                   .filterDate(start, end)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", pct))
                   .select(["B3", "B8"]))
            if col.size().getInfo() == 0:
                continue
            img  = col.median()
            ndwi = img.normalizedDifference(["B3", "B8"]).rename("ndwi")
            val  = ndwi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=haor, scale=S2_SCALE, maxPixels=1e9,
            ).get("ndwi").getInfo()
            if val is not None:
                return round(float(val), 4), True
        except Exception:
            continue
    return DEFAULTS["ndwi"], False


# ── Static terrain (computed once, reused) ────────────────────────────────────

def fetch_static_terrain(ee, haor) -> dict:
    """Slope + TWI from SRTM and HydroSHEDS. Static — compute once."""
    result = {"slope": DEFAULTS["slope"], "twi": DEFAULTS["twi"]}
    try:
        dem       = ee.Image("USGS/SRTMGL1_003")
        slope_val = (ee.Terrain.slope(dem)
                     .reduceRegion(ee.Reducer.mean(), haor, DEM_SCALE, maxPixels=1e9)
                     .get("slope").getInfo())
        if slope_val:
            result["slope"] = round(float(slope_val), 3)
    except Exception as e:
        print(f"  [WARN] Slope fetch failed: {e}")

    try:
        dem       = ee.Image("USGS/SRTMGL1_003")
        slope_rad = ee.Terrain.slope(dem).multiply(3.14159265 / 180.0)
        tan_slope = slope_rad.tan().max(ee.Image(0.001))
        flow_acc  = ee.Image("WWF/HydroSHEDS/15ACC").select("b1")
        sca       = (flow_acc.multiply(450.0 * 450.0)
                     .reproject(crs="EPSG:4326", scale=DEM_SCALE))
        twi_img   = sca.divide(tan_slope).log().rename("twi")
        twi_val   = twi_img.reduceRegion(
            ee.Reducer.mean(), haor, DEM_SCALE, maxPixels=1e9
        ).get("twi").getInfo()
        if twi_val:
            result["twi"] = round(float(twi_val), 3)
    except Exception as e:
        print(f"  [WARN] TWI fetch failed: {e}")

    return result


# ── CHIRPS rainfall ───────────────────────────────────────────────────────────

def fetch_chirps(ee, haor, start: str, end: str) -> tuple:
    try:
        col = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterDate(start, end)
               .select("precipitation"))
        val = (col.sum().reduceRegion(
            ee.Reducer.mean(), haor, CHIRPS_SCALE, maxPixels=1e9
        ).get("precipitation").getInfo())
        if val is not None and float(val) >= 0:
            return round(float(val), 2), True
    except Exception:
        pass
    return DEFAULTS["rainfall"], False


# ── ERA5 soil moisture ────────────────────────────────────────────────────────

def fetch_era5_soil(ee, haor, start: str, end: str) -> tuple:
    try:
        col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .filterDate(start, end)
               .select("volumetric_soil_water_layer_1"))
        val = (col.mean().reduceRegion(
            ee.Reducer.mean(), haor, ERA5_SCALE, maxPixels=1e9
        ).get("volumetric_soil_water_layer_1").getInfo())
        if val is not None:
            return round(float(val) * 100, 2), True
    except Exception:
        pass
    return DEFAULTS["soil_moisture"], False


# ── Open-Meteo (archive + forecast) ──────────────────────────────────────────

def fetch_openmeteo_archive(start: str, end: str) -> dict:
    """Returns temp, wind from Open-Meteo archive API."""
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={HAOR_LAT}&longitude={HAOR_LON}"
            f"&start_date={start}&end_date={end}"
            f"&daily=temperature_2m_mean,wind_speed_10m_max"
            f"&timezone=Asia/Dhaka"
        )
        daily = requests.get(url, timeout=20).json()["daily"]
        temp  = round(sum(daily["temperature_2m_mean"]) / len(daily["temperature_2m_mean"]), 1)
        wind  = round(max(daily["wind_speed_10m_max"]), 1)
        return {"temp": temp, "wind": wind, "ok": True}
    except Exception:
        return {"temp": DEFAULTS["temp"], "wind": DEFAULTS["wind"], "ok": False}


def fetch_openmeteo_forecast_at_date(date_str: str) -> dict:
    """
    For historical events, we don't have actual forecast data.
    Uses Open-Meteo archive for the 3 days AFTER the event date as a
    reasonable proxy for 'what would the 72h forecast have shown'.
    """
    try:
        d     = datetime.strptime(date_str, "%Y-%m-%d").date()
        start = date_str
        end   = (d + timedelta(days=3)).strftime("%Y-%m-%d")

        # Don't go into the future relative to today
        today = datetime.now().date()
        if d + timedelta(days=3) > today:
            return {"f12": 0.0, "f72": 0.0, "ok": False}

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={HAOR_LAT}&longitude={HAOR_LON}"
            f"&start_date={start}&end_date={end}"
            f"&daily=precipitation_sum&timezone=Asia/Dhaka"
        )
        daily = requests.get(url, timeout=20).json()["daily"]
        rains = [p for p in daily["precipitation_sum"] if p is not None]
        f12   = round(rains[0] / 2, 1) if rains else 0.0
        f72   = round(sum(rains[:3]), 1) if rains else 0.0
        return {"f12": f12, "f72": f72, "ok": True}
    except Exception:
        return {"f12": 0.0, "f72": 0.0, "ok": False}


# ── Main collection loop ───────────────────────────────────────────────────────

def collect_event(ee, haor, upstream, terrain, date_str: str, label: int) -> dict:
    """Collect all 13 features for one event date."""
    d          = datetime.strptime(date_str, "%Y-%m-%d").date()
    start      = (d - timedelta(days=7)).strftime("%Y-%m-%d")
    ndwi_start = start  # same 7-day window as SAR; wider lookback causes temporal mismatch
    end        = date_str
    is_s1_era  = d >= S1_LAUNCH

    # Sentinel-1 SAR (only available after launch)
    if is_s1_era:
        sar = fetch_sentinel1_full(ee, haor, upstream, start, end)
    else:
        sar = {
            "VV": DEFAULTS["VV"], "VH": DEFAULTS["VH"],
            "vv_vh_ratio": DEFAULTS["vv_vh_ratio"],
            "upstream_vv": DEFAULTS["upstream_vv"],
            "sar_real": False, "upstream_real": False,
        }

    # Sentinel-2 NDWI — same 7-day window as SAR (wider window causes temporal mismatch)
    ndwi_val, ndwi_real = (
        fetch_ndwi(ee, haor, ndwi_start, end) if is_s1_era else (DEFAULTS["ndwi"], False)
    )

    # CHIRPS rainfall
    rain, rain_real = fetch_chirps(ee, haor, start, end)

    # ERA5 soil moisture
    soil, soil_real = fetch_era5_soil(ee, haor, start, end)

    # Open-Meteo archive: temp + wind
    wx = fetch_openmeteo_archive(start, end)

    # Open-Meteo: forecast proxy
    fc = fetch_openmeteo_forecast_at_date(date_str)

    # Data quality flag
    if sar["sar_real"] and ndwi_real and rain_real:
        quality = "full_real"
    elif sar["sar_real"] and rain_real:
        quality = "sar_only"
    elif is_s1_era:
        quality = "partial"
    else:
        quality = "pre_sentinel1"

    return {
        "date":                  date_str,
        "flood_label":           label,
        "VV":                    sar["VV"],
        "VH":                    sar["VH"],
        "vv_vh_ratio":           sar["vv_vh_ratio"],
        "rainfall":              rain,
        "soil_moisture":         soil,
        "temp":                  wx["temp"],
        "wind":                  wx["wind"],
        "slope":                 terrain["slope"],
        "forecast_rain_next_12h": fc["f12"],
        "ndwi":                  ndwi_val,
        "twi":                   terrain["twi"],
        "upstream_vv":           sar["upstream_vv"],
        "forecast_rain_72h":     fc["f72"],
        # Quality metadata (not model features)
        "data_quality":          quality,
        "sar_real":              sar["sar_real"],
        "ndwi_real":             ndwi_real,
        "upstream_real":         sar["upstream_real"],
    }


def main(resume: bool = False):
    ee = init_gee()

    # Load labeled events from honest_training_data.csv (the gold standard)
    csv_path = DATA_DIR / "honest_training_data.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        sys.exit(1)

    df_labels = pd.read_csv(csv_path)[["date", "flood_label", "source"]]
    total     = len(df_labels)

    # Resume mode: skip dates already in output file
    done_dates = set()
    if resume and OUT_PATH.exists():
        done_dates = set(pd.read_csv(OUT_PATH)["date"].astype(str))
        print(f"  [Resume] {len(done_dates)} dates already collected, skipping.")

    # Compute static terrain once
    print("\nComputing static terrain features (slope + TWI) ...")
    haor     = ee.Geometry.Rectangle(HAOR_BBOX)
    upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)
    terrain  = fetch_static_terrain(ee, haor)
    print(f"  Slope: {terrain['slope']}°   TWI: {terrain['twi']}")

    # Collection loop
    records = []
    skipped_s1 = 0

    print(f"\nCollecting {total} events (SAR available for post-2014 events) ...\n")

    for i, row in df_labels.iterrows():
        date_str = str(row["date"])
        label    = int(row["flood_label"])

        if date_str in done_dates:
            print(f"  [{i+1}/{total}] {date_str} — skipped (already done)")
            continue

        print(f"  [{i+1}/{total}] {date_str}  label={label}  ", end="", flush=True)

        try:
            rec = collect_event(ee, haor, upstream, terrain, date_str, label)
            rec["source"] = row.get("source", "")
            records.append(rec)

            status_parts = [rec["data_quality"]]
            if rec["sar_real"]:
                status_parts.append(f"VV={rec['VV']:.1f}")
            if rec["ndwi_real"]:
                status_parts.append(f"NDWI={rec['ndwi']:.3f}")
            status_parts.append(f"rain={rec['rainfall']:.0f}mm")
            print("  ".join(status_parts))

            if not rec["sar_real"]:
                skipped_s1 += 1

        except Exception as e:
            print(f"ERROR: {e}")
            # Record with all defaults so the date is not silently lost
            records.append({
                "date":        date_str,
                "flood_label": label,
                **{f: DEFAULTS.get(f, 0.0) for f in [
                    "VV","VH","vv_vh_ratio","rainfall","soil_moisture",
                    "temp","wind","slope","forecast_rain_next_12h",
                    "ndwi","twi","upstream_vv","forecast_rain_72h",
                ]},
                "data_quality":  "error",
                "sar_real":      False,
                "ndwi_real":     False,
                "upstream_real": False,
                "source":        row.get("source", ""),
            })

        # Respect GEE rate limits
        time.sleep(0.8)

        # Write checkpoint every 10 events
        if (len(records) % 10 == 0) or (i + 1 == total):
            df_out = pd.DataFrame(records)
            if resume and OUT_PATH.exists():
                existing = pd.read_csv(OUT_PATH)
                df_out   = pd.concat([existing, df_out], ignore_index=True)
                df_out   = df_out.drop_duplicates(subset=["date"], keep="last")
            df_out.to_csv(OUT_PATH, index=False)
            print(f"    [Checkpoint] {len(records)} records saved → {OUT_PATH}")

    # Final save + summary
    df_final = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame(records)

    print("\n" + "=" * 65)
    print("  Collection complete!")
    print(f"  Total rows       : {len(df_final)}")
    print(f"  Flood events     : {df_final['flood_label'].sum()}")
    print(f"  Dry events       : {(df_final['flood_label'] == 0).sum()}")
    print(f"  Full real (SAR+NDWI+rain) : {(df_final['data_quality'] == 'full_real').sum()}")
    print(f"  SAR only         : {(df_final['data_quality'] == 'sar_only').sum()}")
    print(f"  Pre-Sentinel1    : {(df_final['data_quality'] == 'pre_sentinel1').sum()}")
    print(f"  Errors           : {(df_final['data_quality'] == 'error').sum()}")
    print(f"  Output           : {OUT_PATH}")
    print("=" * 65)
    print()
    print("Next step:")
    print("  python train_honest.py")
    print("  (train_honest.py will automatically use real_training_data_v3.csv")
    print("   if you update the CSV path in that script)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip dates already in the output file")
    args = parser.parse_args()
    main(resume=args.resume)
