"""
train_models.py
Trains the RF + XGBoost ensemble on real GEE + Open-Meteo haor data.
Run collect_real_haor_data.py first to generate haor_real_data.csv,
or pass --synthetic to train on synthetic data for a quick sanity check.
"""

import argparse
import sys

import ee
import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    DATA_DIR, DEFAULTS, FEATURES, GEE_PROJECT,
    HAOR_BBOX, HAOR_LAT, HAOR_LON, MODELS_DIR,
    SAR_SCALE, CHIRPS_SCALE, ERA5_SCALE,
)

# ---------------------------------------------------------------------------
# Historical training periods (2017-2024) — balanced flood / dry
# ---------------------------------------------------------------------------
TRAINING_PERIODS = [
    # --- flood periods ---
    ("2017-06-01", "2017-06-30", 1),
    ("2017-07-01", "2017-07-15", 1),
    ("2017-08-01", "2017-08-15", 1),
    ("2018-05-01", "2018-05-15", 1),
    ("2018-06-15", "2018-07-10", 1),
    ("2018-07-11", "2018-07-25", 1),
    ("2018-08-01", "2018-08-15", 1),
    ("2019-05-01", "2019-05-15", 1),
    ("2019-06-20", "2019-07-05", 1),
    ("2019-07-06", "2019-07-20", 1),
    ("2019-08-01", "2019-08-15", 1),
    ("2020-05-01", "2020-05-15", 1),
    ("2020-06-10", "2020-06-25", 1),
    ("2020-07-01", "2020-07-15", 1),
    ("2021-05-01", "2021-05-15", 1),
    ("2021-07-01", "2021-07-15", 1),
    ("2022-05-01", "2022-05-15", 1),
    ("2022-06-17", "2022-06-25", 1),
    ("2022-06-26", "2022-07-05", 1),
    ("2022-07-01", "2022-07-15", 1),
    ("2023-05-01", "2023-05-15", 1),
    ("2023-07-01", "2023-07-15", 1),
    ("2024-05-01", "2024-05-15", 1),
    ("2024-06-01", "2024-06-20", 1),
    ("2024-06-21", "2024-07-05", 1),
    ("2024-07-01", "2024-07-15", 1),
    # --- dry periods ---
    ("2017-01-01", "2017-01-31", 0),
    ("2017-02-01", "2017-02-28", 0),
    ("2017-09-01", "2017-09-15", 0),
    ("2018-02-01", "2018-02-28", 0),
    ("2018-03-01", "2018-03-31", 0),
    ("2018-09-01", "2018-09-15", 0),
    ("2019-01-01", "2019-01-15", 0),
    ("2019-02-01", "2019-02-28", 0),
    ("2019-03-01", "2019-03-31", 0),
    ("2019-09-01", "2019-09-15", 0),
    ("2020-01-01", "2020-01-31", 0),
    ("2020-02-01", "2020-02-29", 0),
    ("2020-09-01", "2020-09-15", 0),
    ("2021-01-01", "2021-01-15", 0),
    ("2022-01-01", "2022-01-15", 0),
    ("2022-02-01", "2022-02-28", 0),
    ("2022-03-01", "2022-03-31", 0),
    ("2023-01-01", "2023-01-15", 0),
    ("2023-03-01", "2023-03-15", 0),
    ("2023-08-10", "2023-08-25", 0),
    ("2024-01-01", "2024-01-31", 0),
    ("2024-02-01", "2024-02-29", 0),
]

ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_mean,wind_speed_10m_max"
    "&timezone=Asia/Dhaka"
)


# ---------------------------------------------------------------------------
# GEE helpers
# ---------------------------------------------------------------------------

def get_sentinel1_features(haor, start: str, end: str) -> tuple[float, float, float]:
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
    vv = vv if vv is not None else DEFAULTS["VV"]
    vh = vh if vh is not None else DEFAULTS["VH"]
    ratio = vv / vh if vh != 0 else 0.0
    return float(vv), float(vh), round(ratio, 4)


def get_rainfall_chirps(haor, start: str, end: str) -> float:
    try:
        chirps = (
            ee.ImageCollection("UCSB/CHIRPS/DAILY")
            .filterDate(start, end)
            .sum()
            .select("precipitation")
        )
        val = chirps.reduceRegion(ee.Reducer.mean(), haor, CHIRPS_SCALE).get("precipitation").getInfo()
        if val and val > 0:
            return round(float(val), 1)
    except Exception:
        pass
    return DEFAULTS["rainfall"]


def get_soil_moisture_era5(haor, start: str, end: str) -> float:
    try:
        era = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start, end)
            .select("volumetric_soil_water_layer_1")
            .mean()
        )
        val = era.reduceRegion(ee.Reducer.mean(), haor, ERA5_SCALE).get("volumetric_soil_water_layer_1").getInfo()
        if val is not None:
            return round(float(val) * 100, 1)
    except Exception:
        pass
    return DEFAULTS["soil_moisture"]


def get_weather_openmeteo(start: str, end: str) -> tuple[float, float, float]:
    try:
        url = ARCHIVE_URL.format(lat=HAOR_LAT, lon=HAOR_LON, start=start, end=end)
        daily = requests.get(url, timeout=20).json()["daily"]
        rain = round(sum(daily["precipitation_sum"]), 1)
        temp = round(sum(daily["temperature_2m_mean"]) / len(daily["temperature_2m_mean"]), 1)
        wind = round(max(daily["wind_speed_10m_max"]), 1)
        return rain, temp, wind
    except Exception:
        return DEFAULTS["rainfall"], DEFAULTS["temp"], DEFAULTS["wind"]


def get_slope(haor) -> float:
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        val = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("slope").getInfo()
        return float(val) if val else DEFAULTS["slope"]
    except Exception:
        return DEFAULTS["slope"]


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_real_dataset() -> pd.DataFrame:
    ee.Initialize(project=GEE_PROJECT)
    haor = ee.Geometry.Rectangle(HAOR_BBOX)

    slope = get_slope(haor)
    rows = []

    for idx, (start_str, end_str, label) in enumerate(TRAINING_PERIODS):
        print(f"  [{idx + 1}/{len(TRAINING_PERIODS)}] {start_str} → {end_str} ...")

        vv, vh, ratio = get_sentinel1_features(haor, start_str, end_str)
        soil = get_soil_moisture_era5(haor, start_str, end_str)
        rain_gee = get_rainfall_chirps(haor, start_str, end_str)
        rain_om, temp, wind = get_weather_openmeteo(start_str, end_str)

        # Prefer CHIRPS if it returned a non-default value
        rain = rain_gee if rain_gee != DEFAULTS["rainfall"] else rain_om
        forecast_proxy = rain * 0.5  # no real historical forecast; proxy used at train time

        rows.append([vv, vh, ratio, rain, soil, temp, wind, slope, forecast_proxy, label])

    return pd.DataFrame(rows, columns=FEATURES + ["flood"])


def build_synthetic_dataset(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vv    = rng.normal(-19, 6, n)
    vh    = rng.normal(-23, 5, n)
    ratio = vv / np.where(vh != 0, vh, -0.001)
    rain  = rng.normal(200, 100, n)
    soil  = rng.normal(48, 12, n)
    slope = rng.normal(2.0, 1.0, n)
    temp  = rng.normal(29, 3, n)
    wind  = rng.normal(12, 5, n)
    fcast = rain * 0.5 + rng.normal(0, 20, n)

    # Flood logic: low SAR backscatter OR high rainfall OR saturated soil
    flood = ((vv < -16) | (rain > 160) | (soil > 42)).astype(int)

    return pd.DataFrame({
        "VV": vv, "VH": vh, "vv_vh_ratio": ratio,
        "rainfall": rain, "soil_moisture": soil,
        "temp": temp, "wind": wind, "slope": slope,
        "forecast_rain_next_12h": fcast,
        "flood": flood,
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(df: pd.DataFrame) -> None:
    X = df[FEATURES]
    y = df["flood"]

    print(f"\nDataset: {len(X)} samples | flood={y.sum()} | dry={(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        scale_pos_weight=y_train.value_counts().get(0, 1) / max(y_train.value_counts().get(1, 1), 1),
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    print("Training Random Forest ...")
    rf.fit(X_train, y_train)

    print("Training XGBoost ...")
    xgb.fit(X_train, y_train)

    rf_path  = MODELS_DIR / "rf_model.pkl"
    xgb_path = MODELS_DIR / "xgb_model.pkl"
    joblib.dump(rf, rf_path)
    joblib.dump(xgb, xgb_path)
    print(f"Models saved → {MODELS_DIR}")

    # Evaluation
    rf_pred  = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    rf_prob  = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    ens_prob = 0.55 * rf_prob + 0.45 * xgb_prob
    ens_pred = (ens_prob > 0.5).astype(int)

    for name, pred, prob in [("RF", rf_pred, rf_prob), ("XGBoost", xgb_pred, xgb_prob), ("Ensemble", ens_pred, ens_prob)]:
        acc  = accuracy_score(y_test, pred) * 100
        prec = precision_score(y_test, pred, zero_division=0) * 100
        rec  = recall_score(y_test, pred, zero_division=0) * 100
        f1   = f1_score(y_test, pred, zero_division=0) * 100
        auc  = roc_auc_score(y_test, prob)
        print(f"  {name:<10} Acc={acc:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%  F1={f1:.1f}%  AUC={auc:.4f}")

    print("\nFeature importance (RF):")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:<28} {imp:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Haor flood prediction models")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no GEE auth needed)")
    parser.add_argument("--csv", type=str, default=None, help="Path to pre-collected CSV (skips GEE)")
    args = parser.parse_args()

    if args.csv:
        print(f"Loading dataset from {args.csv} ...")
        df = pd.read_csv(args.csv)
        df = df.rename(columns={"flood_label": "flood"})
    elif args.synthetic:
        print("Building synthetic dataset ...")
        df = build_synthetic_dataset()
    else:
        print("Fetching real GEE data ...")
        df = build_real_dataset()

    train_and_save(df)


if __name__ == "__main__":
    main()
