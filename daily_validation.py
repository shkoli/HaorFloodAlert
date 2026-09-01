"""
daily_validation.py — Log today's live prediction vs. observed water level.

Fetches live features, runs the ensemble, fetches SW269 water level from the
FFWC live API (fallback: XLSX), and appends one row to data/daily_validation_log.csv.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import ee
import joblib
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup
urllib3.disable_warnings()
from datetime import datetime, timedelta, timezone

from config import GEE_PROJECT, MODELS_DIR, DATA_DIR, FEATURES
from utils.gee_features import fetch_all_features, get_forecast_rainfall_72h
from utils.predict import predict_flood_72h
from utils.upstream_discharge import get_barak_discharge_current, classify_discharge_risk
from utils.discharge_trend import analyze_discharge_trend, predict_discharge_next_72h

BST         = timezone(timedelta(hours=6))
LOG_CSV     = DATA_DIR / "daily_validation_log.csv"
WL_XLSX     = DATA_DIR / "Sunamganj (SW269) WL.xlsx"

FFWC_CHART_URL = "http://old.ffwc.gov.bd/ffwc_charts/index.php?stid=65"
SW269_STATION  = "SW269"
SW269_DANGER   = 6.05   # metres

CSV_COLUMNS = [
    "date", "time", "flood_probability", "risk_level",
    "sw269_water_level", "forecast_rain_72h", "soil_moisture", "barak_discharge",
    "danger_level", "wl_status",
]

RISK_THRESHOLDS = [
    (0.70, "CRITICAL"),
    (0.50, "HIGH"),
    (0.30, "MODERATE"),
    (0.00, "LOW"),
]


def fetch_ffwc_live_wl() -> dict:
    """
    Fetch latest SW269 (Sunamganj) water level from FFWC live API.

    Returns dict with keys:
        water_level  (float | nan)
        danger_level (float)
        status       (str)   e.g. "12 cm below danger level"
        trend        (str)   "rising" | "falling" | "steady"
        source       (str)   "scrape" | "xlsx"
    """
    danger = SW269_DANGER

    def _parse_scrape() -> dict | None:
        try:
            resp = requests.get(FFWC_CHART_URL, timeout=20, verify=False)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract all visible numeric text that could be water levels
            wl = None
            danger_scraped = None
            trend = "steady"

            # Try to find table rows or labelled cells containing water level data
            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            wl_values = []
            for i, line in enumerate(lines):
                low = line.lower()
                # Danger level
                if danger_scraped is None and "danger" in low:
                    for j in range(i, min(i + 4, len(lines))):
                        try:
                            danger_scraped = float(lines[j].replace("m", "").strip())
                            break
                        except ValueError:
                            continue
                # Water level readings — look for bare float values near "water level" labels
                if "water level" in low or "wl" in low:
                    for j in range(i, min(i + 4, len(lines))):
                        try:
                            v = float(lines[j].replace("m", "").strip())
                            wl_values.append(v)
                            break
                        except ValueError:
                            continue

            # Fallback: collect all plausible water-level floats (0 – 20 m range)
            if not wl_values:
                import re
                for token in re.findall(r"\b\d{1,2}\.\d{2,3}\b", text):
                    v = float(token)
                    if 0.5 <= v <= 15.0:
                        wl_values.append(v)

            if not wl_values:
                return None

            wl = round(wl_values[-1], 3)

            # Trend from last two values
            if len(wl_values) >= 2:
                diff = wl_values[-1] - wl_values[-2]
                if diff > 0.02:
                    trend = "rising"
                elif diff < -0.02:
                    trend = "falling"

            # Danger level override from scrape if found
            nonlocal danger
            if danger_scraped is not None and 4.0 <= danger_scraped <= 10.0:
                danger = round(danger_scraped, 2)

            return {"water_level": wl, "trend": trend, "source": "scrape"}

        except Exception as e:
            print(f"  [WARNING] FFWC scrape error: {e}")
            return None

    def _parse_xlsx() -> dict | None:
        try:
            df = pd.read_excel(WL_XLSX, header=0)
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return None
            series = df[num_cols[-1]].dropna()
            if series.empty:
                return None
            wl = round(float(series.iloc[-1]), 3)
            trend = "steady"
            if len(series) >= 2:
                diff = wl - float(series.iloc[-2])
                if diff > 0.02:
                    trend = "rising"
                elif diff < -0.02:
                    trend = "falling"
            return {"water_level": wl, "trend": trend, "source": "xlsx"}
        except Exception as e:
            print(f"  [WARNING] SW269 XLSX fallback error: {e}")
            return None

    result = _parse_scrape() or _parse_xlsx()

    if result is None:
        return {
            "water_level": float("nan"),
            "danger_level": danger,
            "status": "unknown",
            "trend": "unknown",
            "source": "none",
        }

    wl = result["water_level"]
    diff_cm = round((wl - danger) * 100)
    if diff_cm > 0:
        status = f"{abs(diff_cm)} cm above danger level"
    elif diff_cm < 0:
        status = f"{abs(diff_cm)} cm below danger level"
    else:
        status = "at danger level"

    return {
        "water_level": wl,
        "danger_level": danger,
        "status": status,
        "trend": result["trend"],
        "source": result["source"],
    }


def risk_label(prob: float) -> str:
    for threshold, label in RISK_THRESHOLDS:
        if prob >= threshold:
            return label
    return "LOW"


def load_models():
    rf  = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")
    act_path = MODELS_DIR / "active_features.pkl"
    active_feats = joblib.load(act_path) if act_path.exists() else list(FEATURES)[:rf.n_features_in_]
    return rf, xgb, active_feats




def main():
    now_bst = datetime.now(BST)
    today   = now_bst.strftime("%Y-%m-%d")
    time_str = now_bst.strftime("%H:%M")

    print(f"=== HaorFloodAlert — Daily Validation  {today} {time_str} BST ===")

    # ── GEE init ─────────────────────────────────────────────────────────────
    print("Initialising Google Earth Engine...")
    ee.Initialize(project=GEE_PROJECT)

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading ML models...")
    rf, xgb, active_feats = load_models()

    # ── Fetch live features ───────────────────────────────────────────────────
    end_date   = today
    start_date = (now_bst - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"Fetching live GEE features ({start_date} → {end_date})...")
    features = fetch_all_features(start_date, end_date, include_upstream=True)
    print(f"  rainfall={features['rainfall']} mm  soil_moisture={features['soil_moisture']}%"
          f"  VV={features['VV']} dB  upstream_vv={features['upstream_vv']} dB")

    # ── Forecast rainfall 72h ─────────────────────────────────────────────────
    print("Fetching 72h rainfall forecast...")
    forecast_breakdown = get_forecast_rainfall_72h()
    forecast_rain_72h  = forecast_breakdown.get("total", features.get("forecast_rain_72h", 0.0))
    features["forecast_rain_72h"] = forecast_rain_72h
    print(f"  forecast_rain_72h={forecast_rain_72h} mm")

    # ── Barak discharge ───────────────────────────────────────────────────────
    print("Fetching Barak discharge...")
    discharge_info = get_barak_discharge_current()
    barak_discharge   = discharge_info.get("discharge", 0.0)
    discharge_trend   = discharge_info.get("trend", 0.0)
    is_trend_reliable = discharge_info.get("is_trend_reliable", True)
    discharge_proj    = predict_discharge_next_72h(barak_discharge, discharge_trend)
    print(f"  barak_discharge={barak_discharge:,.0f} m³/s  trend={discharge_trend:+.0f} m³/s/day")

    # ── Run ensemble prediction ───────────────────────────────────────────────
    print("Running 3-layer ensemble prediction...")
    result = predict_flood_72h(
        current_features       = features,
        forecast_breakdown     = forecast_breakdown,
        rf                     = rf,
        xgb                    = xgb,
        active_feats           = active_feats,
        current_discharge      = barak_discharge,
        discharge_trend        = discharge_trend,
        discharge_projections  = discharge_proj,
        is_trend_reliable      = is_trend_reliable,
    )
    flood_prob  = result["peak"]
    risk        = risk_label(flood_prob)
    print(f"  peak_prob={flood_prob:.3f}  peak_window={result['peak_window']}  risk={risk}")

    # ── Fetch SW269 water level (FFWC API → XLSX fallback) ───────────────────
    print("Fetching SW269 water level from FFWC chart page...")
    wl_info  = fetch_ffwc_live_wl()
    sw269_wl = wl_info["water_level"]
    print(f"  sw269_water_level={sw269_wl} m  danger={wl_info['danger_level']} m"
          f"  status='{wl_info['status']}'  trend={wl_info['trend']}  source={wl_info['source']}")

    # ── Build log row ─────────────────────────────────────────────────────────
    row = {
        "date":              today,
        "time":              time_str,
        "flood_probability": round(flood_prob, 4),
        "risk_level":        risk,
        "sw269_water_level": sw269_wl,
        "forecast_rain_72h": forecast_rain_72h,
        "soil_moisture":     features.get("soil_moisture", float("nan")),
        "barak_discharge":   round(barak_discharge, 1),
        "danger_level":      wl_info["danger_level"],
        "wl_status":         wl_info["status"],
    }

    # ── Append to CSV ─────────────────────────────────────────────────────────
    row_df = pd.DataFrame([row], columns=CSV_COLUMNS)
    if LOG_CSV.exists():
        row_df.to_csv(LOG_CSV, mode="a", header=False, index=False)
        print(f"Appended to {LOG_CSV}")
    else:
        row_df.to_csv(LOG_CSV, mode="w", header=True, index=False)
        print(f"Created {LOG_CSV}")

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print(f"│  Date              : {today}  {time_str} BST")
    print(f"│  Flood Probability : {flood_prob*100:.1f}%  ({result['peak_window']})")
    print(f"│  Risk Level        : {risk}")
    print(f"│  SW269 Water Level : {sw269_wl} m  [{wl_info['source']}]")
    print(f"│  Danger Level      : {wl_info['danger_level']} m")
    print(f"│  Status            : {wl_info['status']}")
    print(f"│  WL Trend          : {wl_info['trend']}")
    print(f"│  Forecast Rain 72h : {forecast_rain_72h} mm")
    print(f"│  Soil Moisture     : {features.get('soil_moisture')}%")
    print(f"│  Barak Discharge   : {barak_discharge:,.0f} m³/s")
    print("└─────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
