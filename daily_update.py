"""
daily_update.py
===============
Pulls today's Sentinel-1 + CHIRPS + ERA5 data from GEE, fetches the
12h/72h precipitation forecast from Open-Meteo, runs the trained ensemble,
and appends a daily risk report to results/daily_report.csv.

Designed to be scheduled (cron / Windows Task Scheduler).

Fixed: uses all 13 features matching config.FEATURES exactly.
"""

import ee
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

from config import (
    DEFAULTS, FEATURES, GEE_PROJECT,
    HAOR_BBOX, HAOR_LAT, HAOR_LON,
    UPSTREAM_BBOX,
    MODELS_DIR, RESULTS_DIR,
    RF_WEIGHT, XGB_WEIGHT,
    RISK_THRESHOLDS, SAR_SCALE, CHIRPS_SCALE, ERA5_SCALE, DEM_SCALE,
    S2_SCALE,
)


def init_gee():
    ee.Initialize(project=GEE_PROJECT)
    return ee.Geometry.Rectangle(HAOR_BBOX)


def fetch_sentinel1(haor, start: str, end: str) -> tuple:
    try:
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(haor)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
            .median()
        )
        vv = s1.select("VV").reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("VV").getInfo()
        vh = s1.select("VH").reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("VH").getInfo()
        vv = float(vv) if vv is not None else DEFAULTS["VV"]
        vh = float(vh) if vh is not None else DEFAULTS["VH"]
        return vv, vh, (vv / vh if vh != 0 else DEFAULTS["vv_vh_ratio"])
    except Exception as e:
        print(f"  [WARN] Sentinel-1 fetch failed: {e}")
        return DEFAULTS["VV"], DEFAULTS["VH"], DEFAULTS["vv_vh_ratio"]


def fetch_upstream_vv(start: str, end: str) -> float:
    """Barak river upstream SAR — correct coordinates from config."""
    try:
        upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)
        s1_up = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(upstream)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select(["VV"])
            .median()
        )
        val = s1_up.select("VV").reduceRegion(
            ee.Reducer.mean(), upstream, SAR_SCALE
        ).get("VV").getInfo()
        return float(val) if val is not None else DEFAULTS["upstream_vv"]
    except Exception as e:
        print(f"  [WARN] Upstream VV fetch failed: {e}")
        return DEFAULTS["upstream_vv"]


def fetch_ndwi(haor, start: str, end: str) -> float:
    try:
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(haor)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
            .select(["B3", "B8"])
            .median()
        )
        ndwi_img = s2.normalizedDifference(["B3", "B8"]).rename("ndwi")
        val = ndwi_img.reduceRegion(
            ee.Reducer.mean(), haor, S2_SCALE
        ).get("ndwi").getInfo()
        return round(float(val), 4) if val is not None else DEFAULTS["ndwi"]
    except Exception:
        return DEFAULTS["ndwi"]


def fetch_twi(haor) -> float:
    try:
        dem       = ee.Image("USGS/SRTMGL1_003")
        slope_rad = ee.Terrain.slope(dem).multiply(3.14159265 / 180.0)
        tan_slope = slope_rad.tan().max(ee.Image(0.001))
        flow_acc  = ee.Image("WWF/HydroSHEDS/15ACC").select("b1")
        sca       = flow_acc.multiply(450.0 * 450.0).reproject(crs="EPSG:4326", scale=DEM_SCALE)
        twi_img   = sca.divide(tan_slope).log().rename("twi")
        val = twi_img.reduceRegion(
            ee.Reducer.mean(), haor, DEM_SCALE
        ).get("twi").getInfo()
        return round(float(val), 3) if val is not None else DEFAULTS["twi"]
    except Exception:
        return DEFAULTS["twi"]


def fetch_rainfall_chirps(haor, start: str, end: str) -> float:
    try:
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start, end)
            .sum()
            .select("precipitation")
        )
        val = chirps.reduceRegion(
            ee.Reducer.mean(), haor, CHIRPS_SCALE
        ).get("precipitation").getInfo()
        if val and float(val) > 0:
            return round(float(val), 1)
    except Exception:
        pass
    return DEFAULTS["rainfall"]


def fetch_soil_moisture(haor, start: str, end: str) -> float:
    try:
        era = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start, end)
            .select("volumetric_soil_water_layer_1")
            .mean()
        )
        val = era.reduceRegion(
            ee.Reducer.mean(), haor, ERA5_SCALE
        ).get("volumetric_soil_water_layer_1").getInfo()
        if val is not None:
            return round(float(val) * 100, 1)
    except Exception:
        pass
    return DEFAULTS["soil_moisture"]


def fetch_slope(haor) -> float:
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        val = ee.Terrain.slope(dem).reduceRegion(
            ee.Reducer.mean(), haor, DEM_SCALE
        ).get("slope").getInfo()
        return round(float(val), 2) if val else DEFAULTS["slope"]
    except Exception:
        return DEFAULTS["slope"]


def fetch_weather(start: str, end: str) -> tuple:
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
        return temp, wind
    except Exception:
        return DEFAULTS["temp"], DEFAULTS["wind"]


def fetch_forecast() -> tuple:
    """Returns (forecast_12h_mm, forecast_72h_mm)."""
    f12, f72 = 0.0, 0.0
    try:
        from datetime import timezone as tz
        now = datetime.now(tz.utc)
        r   = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={HAOR_LAT}&longitude={HAOR_LON}"
            f"&hourly=precipitation&forecast_days=4&timezone=Asia/Dhaka",
            timeout=15,
        ).json()["hourly"]
        for t_str, p in zip(r["time"], r["precipitation"]):
            dt_aware = datetime.fromisoformat(t_str).replace(tzinfo=tz.utc)
            hours    = (dt_aware - now).total_seconds() / 3600
            if 0 < hours <= 12:
                f12 += p
            if 0 < hours <= 72:
                f72 += p
    except Exception:
        pass
    return round(f12, 1), round(f72, 1)


def classify_risk(prob: float) -> str:
    for level, threshold in RISK_THRESHOLDS.items():
        if prob >= threshold:
            return level
    return "LOW"


def estimate_inundation(prob: float) -> float:
    return round(45 + prob * 85, 1)


def main():
    now       = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    week_ago  = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    print(f"=== Daily Haor Flood Update — {today_str} ===")

    haor = init_gee()

    print("  Fetching 13 features from GEE + APIs ...")
    vv, vh, ratio   = fetch_sentinel1(haor, week_ago, today_str)
    rain            = fetch_rainfall_chirps(haor, week_ago, today_str)
    soil            = fetch_soil_moisture(haor, week_ago, today_str)
    temp, wind      = fetch_weather(week_ago, today_str)
    slope           = fetch_slope(haor)
    f12, f72        = fetch_forecast()
    ndwi            = fetch_ndwi(haor, week_ago, today_str)
    twi             = fetch_twi(haor)
    upstream_vv     = fetch_upstream_vv(week_ago, today_str)

    # Load models and active feature list
    rf         = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb        = joblib.load(MODELS_DIR / "xgb_model.pkl")
    act_path   = MODELS_DIR / "active_features.pkl"
    active_feats = joblib.load(act_path) if act_path.exists() else list(FEATURES)

    # Build feature row in canonical FEATURES order
    feat_values = {
        "VV":                    round(vv,    2),
        "VH":                    round(vh,    2),
        "vv_vh_ratio":           round(ratio, 4),
        "rainfall":              rain,
        "soil_moisture":         soil,
        "temp":                  temp,
        "wind":                  wind,
        "slope":                 slope,
        "forecast_rain_next_12h": f12,
        "ndwi":                  ndwi,
        "twi":                   twi,
        "upstream_vv":           round(upstream_vv, 2),
        "forecast_rain_72h":     f72,
    }

    # Use only features the model was trained on
    input_df = pd.DataFrame(
        [[feat_values[f] for f in active_feats]],
        columns=active_feats,
    )

    rf_prob    = float(rf.predict_proba(input_df)[0][1])
    xgb_prob   = float(xgb.predict_proba(input_df)[0][1])
    final_prob = RF_WEIGHT * rf_prob + XGB_WEIGHT * xgb_prob

    risk      = classify_risk(final_prob)
    inundated = estimate_inundation(final_prob)

    print(f"  VV backscatter : {vv:.1f} dB")
    print(f"  Upstream VV    : {upstream_vv:.1f} dB")
    print(f"  7-day rainfall : {rain:.1f} mm")
    print(f"  Soil moisture  : {soil:.1f} %")
    print(f"  NDWI           : {ndwi:.3f}")
    print(f"  Forecast 12h   : {f12:.1f} mm")
    print(f"  Forecast 72h   : {f72:.1f} mm")
    print(f"  Flood prob     : {final_prob:.3f}")
    print(f"  Risk level     : {risk}")
    print(f"  Est. inundated : {inundated} km²")

    report_row = pd.DataFrame({
        "timestamp":           [now.strftime("%Y-%m-%d %H:%M UTC")],
        "VV_dB":               [round(vv, 1)],
        "VH_dB":               [round(vh, 1)],
        "upstream_vv_dB":      [round(upstream_vv, 1)],
        "ndwi":                [ndwi],
        "rainfall_7d_mm":      [round(rain, 1)],
        "soil_moisture_pct":   [round(soil, 1)],
        "forecast_rain_12h":   [round(f12, 1)],
        "forecast_rain_72h":   [round(f72, 1)],
        "flood_prob":          [round(final_prob, 3)],
        "rf_prob":             [round(rf_prob, 3)],
        "xgb_prob":            [round(xgb_prob, 3)],
        "risk_level":          [risk],
        "inundated_km2":       [inundated],
        "active_features":     [len(active_feats)],
    })

    out_path = RESULTS_DIR / "daily_report.csv"
    report_row.to_csv(
        out_path,
        mode="a",
        header=not out_path.exists(),
        index=False,
    )
    print(f"  Report appended → {out_path}")


if __name__ == "__main__":
    main()
