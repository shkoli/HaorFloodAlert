"""
fix_csv.py
Patches data/real_training_data_v2.csv:
  1. Renames old column names  ->  correct 13-feature names
  2. Fetches missing features   ->  ndwi, twi, upstream_vv, forecast_rain_72h
  3. Saves updated CSV  (overwrites in place)
Run once — takes ~20 min.
"""
import ee
import pandas as pd
import requests
from datetime import datetime, timedelta
from config import (
    GEE_PROJECT, HAOR_BBOX, UPSTREAM_BBOX,
    SAR_SCALE, S2_SCALE, ERA5_SCALE, DEM_SCALE, CHIRPS_SCALE,
    HAOR_LAT, HAOR_LON, DATA_DIR, DEFAULTS
)

ee.Initialize(project=GEE_PROJECT)
haor     = ee.Geometry.Rectangle(HAOR_BBOX)
upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)

# ── helpers ───────────────────────────────────────────────────────────────

def _count(region, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region).filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .size().getInfo()
    )


def fetch_upstream_vv(start, end, wide_start):
    for s, e in [(start, end), (wide_start, end)]:
        try:
            if _count(upstream, s, e) == 0:
                continue
            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(upstream).filterDate(s, e)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .select(["VV"]).median()
            )
            val = s1.select("VV").reduceRegion(
                ee.Reducer.mean(), upstream, SAR_SCALE
            ).get("VV").getInfo()
            if val is not None:
                return round(float(val), 2)
        except Exception:
            continue
    return DEFAULTS["upstream_vv"]


def fetch_ndwi(start, end):
    try:
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(haor).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .select(["B3", "B8"]).median()
        )
        val = (
            s2.normalizedDifference(["B3", "B8"]).rename("ndwi")
            .reduceRegion(ee.Reducer.mean(), haor, S2_SCALE)
            .get("ndwi").getInfo()
        )
        if val is not None:
            return round(float(val), 4)
    except Exception:
        pass
    return DEFAULTS["ndwi"]


def fetch_twi():
    """Static — computed once for the Haor region."""
    try:
        flow_acc   = ee.Image("WWF/HydroSHEDS/15ACC").select("b1")
        dem        = ee.Image("USGS/SRTMGL1_003")
        slope_rad  = ee.Terrain.slope(dem).multiply(3.14159265 / 180.0)
        tan_slope  = slope_rad.tan()
        tan_safe   = tan_slope.where(tan_slope.lt(0.001), 0.001)
        sca        = flow_acc.multiply(450.0 * 450.0).reproject(crs="EPSG:4326", scale=DEM_SCALE)
        twi_img    = sca.divide(tan_safe).log().rename("twi")
        val = twi_img.reduceRegion(ee.Reducer.mean(), haor, DEM_SCALE).get("twi").getInfo()
        if val is not None:
            return round(float(val), 3)
    except Exception:
        pass
    return DEFAULTS["twi"]


def fetch_forecast_72h(start, end):
    """Proxy: 3-day cumulative from archive rainfall."""
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={HAOR_LAT}&longitude={HAOR_LON}"
            f"&start_date={start}&end_date={end}"
            f"&daily=precipitation_sum&timezone=Asia/Dhaka"
        )
        d = requests.get(url, timeout=20).json()["daily"]
        days = max(len(d["precipitation_sum"]), 1)
        return round(sum(d["precipitation_sum"]) / days * 3.0, 1)
    except Exception:
        return DEFAULTS["forecast_rain_72h"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    csv_path = DATA_DIR / "real_training_data_v2.csv"
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows  |  columns: {list(df.columns)}\n")

    # Step 1 — rename old column names
    rename_map = {
        "vvvh_ratio":             "vv_vh_ratio",
        "forecast_rain_next12h":  "forecast_rain_next_12h",
    }
    df.rename(columns=rename_map, inplace=True)
    print("  [OK] Column names fixed")

    # Step 2 — fetch missing features row by row
    print("\nFetching ndwi, twi, upstream_vv, forecast_rain_72h from GEE ...")
    print("(~20 min — do not close CMD)\n")

    twi_val = fetch_twi()
    print(f"  [OK] TWI (static) = {twi_val}\n")

    ndwi_list, upvv_list, f72_list = [], [], []

    for i, row in df.iterrows():
        date_str   = row["date"]
        date       = datetime.strptime(date_str, "%Y-%m-%d").date()
        end_str    = date_str
        start_str  = str(date - timedelta(days=7))
        wide_start = str(date - timedelta(days=30))

        print(f"  [{i+1}/{len(df)}] {date_str} ...", flush=True)

        ndwi_list.append(fetch_ndwi(start_str, end_str))
        upvv_list.append(fetch_upstream_vv(start_str, end_str, wide_start))
        f72_list.append(fetch_forecast_72h(start_str, end_str))

    df["ndwi"]               = ndwi_list
    df["twi"]                = twi_val
    df["upstream_vv"]        = upvv_list
    df["forecast_rain_72h"]  = f72_list

    # Step 3 — verify all 13 features
    required = [
        "VV","VH","vv_vh_ratio","rainfall","soil_moisture",
        "temp","wind","slope","forecast_rain_next_12h",
        "ndwi","twi","upstream_vv","forecast_rain_72h"
    ]
    missing = [f for f in required if f not in df.columns]
    if missing:
        print(f"\n  [WARN] Still missing: {missing}")
    else:
        print(f"\n  [OK] All 13 features present!")

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved -> {csv_path}")
    print(f"\n  Preview:")
    print(df[["date","flood_label","VV","ndwi","twi","upstream_vv","forecast_rain_72h"]].head(5).to_string(index=False))
    print("\n  Now run: python validate_models.py")


if __name__ == "__main__":
    main()
