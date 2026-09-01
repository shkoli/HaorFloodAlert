"""
validate_models.py  -  HaorFloodAlert Validation v3
====================================================
Validates RF + XGBoost ensemble on held-out historical flood events.
Works in TWO modes:
  - FAST mode : uses data/real_training_data_v2.csv (no GEE needed)
  - GEE  mode : fetches live Sentinel-1 / CHIRPS / ERA5 (requires auth)

Fixed: UTF-8 encoding on all file writes + console output.
"""

import sys
import warnings
import os

warnings.filterwarnings("ignore")

# Force UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import (
    DEFAULTS, FEATURES, GEE_PROJECT,
    HAOR_BBOX, HAOR_LAT, HAOR_LON,
    MODELS_DIR, RESULTS_DIR,
    RF_WEIGHT, XGB_WEIGHT,
    SAR_SCALE, CHIRPS_SCALE, ERA5_SCALE,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_V2    = os.path.join("data", "real_training_data_v2.csv")
DATA_V1    = os.path.join("data", "real_training_data.csv")
REPORT_CSV = RESULTS_DIR / "validation_13features.csv"
REPORT_TXT = RESULTS_DIR / "training_report.txt"

DIVIDER = "=" * 70
SUBDIV  = "-" * 70

# ── Held-out test periods ──────────────────────────────────────────────────────
TEST_PERIODS = [
    ("2024-06-01", "2024-06-20", 1, "Jun 2024 Flood"),
    ("2024-07-01", "2024-07-20", 1, "Jul 2024 Flood"),
    ("2024-08-01", "2024-08-20", 1, "Aug 2024 Flood"),
    ("2025-06-01", "2025-06-20", 1, "Jun 2025 Flood"),
    ("2025-07-01", "2025-07-20", 1, "Jul 2025 Flood"),
    ("2024-01-01", "2024-01-31", 0, "Jan 2024 Dry"),
    ("2024-02-01", "2024-02-29", 0, "Feb 2024 Dry"),
    ("2025-01-01", "2025-01-31", 0, "Jan 2025 Dry"),
    ("2025-02-01", "2025-02-28", 0, "Feb 2025 Dry"),
    ("2025-03-01", "2025-03-31", 0, "Mar 2025 Dry"),
]

ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_mean,wind_speed_10m_max"
    "&timezone=Asia/Dhaka"
)


def banner(title):
    print("\n" + DIVIDER)
    print(f"  {title}")
    print(DIVIDER)


def risk_level(prob):
    if prob > 0.85: return "EXTREME"
    if prob > 0.65: return "HIGH"
    if prob > 0.40: return "MEDIUM"
    return "LOW"


def _gee_available():
    try:
        import ee
        ee.Initialize(project=GEE_PROJECT)
        return True
    except Exception:
        return False


# ── GEE helpers ────────────────────────────────────────────────────────────────
def get_sentinel1(haor, start, end):
    import ee
    try:
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(haor).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]).median()
        )
        vv = s1.select("VV").reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("VV").getInfo()
        vh = s1.select("VH").reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("VH").getInfo()
        vv = float(vv) if vv is not None else DEFAULTS["VV"]
        vh = float(vh) if vh is not None else DEFAULTS["VH"]
        return vv, vh, (vv / vh if vh != 0 else 0.0)
    except Exception:
        return DEFAULTS["VV"], DEFAULTS["VH"], 0.0


def get_ndwi(haor, start, end):
    import ee
    try:
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(haor).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
            .select(["B3", "B8"])
            .median()
        )
        ndwi_img = s2.normalizedDifference(["B3", "B8"]).rename("ndwi")
        val = ndwi_img.reduceRegion(
            ee.Reducer.mean(), haor, SAR_SCALE
        ).get("ndwi").getInfo()
        return float(val) if val is not None else DEFAULTS["ndwi"]
    except Exception:
        return DEFAULTS["ndwi"]


def get_twi(haor):
    import ee
    try:
        dem      = ee.Image("USGS/SRTMGL1_003")
        slope    = ee.Terrain.slope(dem)
        slopeRad = slope.multiply(3.14159265358979 / 180)
        tanSlope = slopeRad.tan().max(ee.Image(0.001))
        flow     = ee.Image("WWF/HydroSHEDS/15ACC")
        twi      = flow.divide(tanSlope).log()
        val = twi.reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("b1").getInfo()
        return float(val) if val else DEFAULTS.get("TWI", 12.0)
    except Exception:
        return DEFAULTS.get("TWI", 12.0)


def get_upstream_vv(haor, start, end):
    """
    Barak river upstream proxy — Silchar, Assam (India).
    Correct coordinates from config.UPSTREAM_BBOX: [92.70, 24.60, 93.20, 25.00]
    Previous bug: used [91.0, 25.1, 91.8, 25.6] which is over the haor itself.
    """
    import ee
    from config import UPSTREAM_BBOX
    try:
        upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(upstream).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select(["VV"]).median()
        )
        val = s1.select("VV").reduceRegion(
            ee.Reducer.mean(), upstream, SAR_SCALE
        ).get("VV").getInfo()
        return float(val) if val is not None else DEFAULTS["upstream_vv"]
    except Exception:
        return DEFAULTS["upstream_vv"]


def get_rainfall(haor, start, end):
    import ee
    try:
        chirps = (
            ee.ImageCollection("UCSB/CHIRPS/DAILY")
            .filterDate(start, end).sum().select("precipitation")
        )
        val = chirps.reduceRegion(ee.Reducer.mean(), haor, CHIRPS_SCALE).get("precipitation").getInfo()
        if val and val > 0:
            return round(float(val), 1)
    except Exception:
        pass
    return DEFAULTS["rainfall"]


def get_soil_moisture(haor, start, end):
    import ee
    try:
        era = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start, end)
            .select("volumetric_soil_water_layer_1").mean()
        )
        val = era.reduceRegion(
            ee.Reducer.mean(), haor, ERA5_SCALE
        ).get("volumetric_soil_water_layer_1").getInfo()
        if val is not None:
            return round(float(val) * 100, 1)
    except Exception:
        pass
    return DEFAULTS["soil_moisture"]


def get_weather(start, end):
    import requests
    try:
        url   = ARCHIVE_URL.format(lat=HAOR_LAT, lon=HAOR_LON, start=start, end=end)
        daily = requests.get(url, timeout=20).json()["daily"]
        rain  = round(sum(daily["precipitation_sum"]), 1)
        temp  = round(sum(daily["temperature_2m_mean"]) / len(daily["temperature_2m_mean"]), 1)
        wind  = round(max(daily["wind_speed_10m_max"]), 1)
        return rain, temp, wind
    except Exception:
        return DEFAULTS["rainfall"], DEFAULTS["temp"], DEFAULTS["wind"]


def get_slope(haor):
    import ee
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        val = ee.Terrain.slope(dem).reduceRegion(
            ee.Reducer.mean(), haor, SAR_SCALE
        ).get("slope").getInfo()
        return float(val) if val else DEFAULTS["slope"]
    except Exception:
        return DEFAULTS["slope"]


# ── Mode A: Fast CSV-based validation ─────────────────────────────────────────
def validate_from_csv(rf, xgb, data_file):
    banner(f"Mode: FAST  (using {data_file})")

    df = pd.read_csv(data_file)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  [ERROR] Missing columns: {missing}")
        return

    X = df[FEATURES].values
    # Auto-detect label column name
    label_col = next(
        (c for c in ["flood", "flood_label", "label", "target", "is_flood"]
         if c in df.columns), None
    )
    if label_col is None:
        print(f"  [ERROR] No label column found.")
        print(f"  Columns in CSV: {list(df.columns)}")
        return
    print(f"  [OK] Label column: '{label_col}'")
    y = df[label_col].values

    # ── 5-Fold CV ──────────────────────────────────────────────────────────────
    banner("5-Fold Cross-Validation  (RF + XGB Ensemble)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        prob = RF_WEIGHT * rf.predict_proba(X_val)[:, 1] + \
               XGB_WEIGHT * xgb.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y_val, pred) * 100
        f1  = f1_score(y_val, pred, zero_division=0) * 100
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold}: Accuracy={acc:.1f}%   F1={f1:.1f}%")

    print(f"\n  Mean Accuracy : {np.mean(fold_accs):.1f}%  (+/- {np.std(fold_accs):.1f}%)")
    print(f"  Mean F1 Score : {np.mean(fold_f1s):.1f}%  (+/- {np.std(fold_f1s):.1f}%)")

    # Retrain on full data
    rf.fit(X, y)
    xgb.fit(X, y)

    # ── Held-Out Test (last 20%) ───────────────────────────────────────────────
    banner("Held-Out Test  (last 20% of dataset)")

    split  = int(len(X) * 0.80)
    X_test = X[split:]
    y_test = y[split:]

    prob = RF_WEIGHT * rf.predict_proba(X_test)[:, 1] + \
           XGB_WEIGHT * xgb.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, pred) * 100
    prec = precision_score(y_test, pred, zero_division=0) * 100
    rec  = recall_score(y_test, pred, zero_division=0) * 100
    f1   = f1_score(y_test, pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_test, prob) * 100
    except Exception:
        auc = 0.0
    cm = confusion_matrix(y_test, pred)

    print(f"\n  Accuracy  : {acc:.1f}%")
    print(f"  Precision : {prec:.1f}%")
    print(f"  Recall    : {rec:.1f}%")
    print(f"  F1 Score  : {f1:.1f}%")
    print(f"  AUC-ROC   : {auc:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    # Save CSV report
    rows = []
    for i in range(len(X_test)):
        rows.append({
            "index":        split + i,
            "true_class":   "FLOOD" if y_test[i] == 1 else "DRY",
            "flood_prob":   round(float(prob[i]), 3),
            "predicted":    "FLOOD" if pred[i] == 1 else "DRY",
            "risk":         risk_level(float(prob[i])),
            "correct":      "YES" if pred[i] == y_test[i] else "NO",
        })
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False, encoding="utf-8")
    print(f"\n  CSV saved -> {REPORT_CSV}")

    # Save TXT report
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("HaorFloodAlert - Validation Report\n")
        f.write(DIVIDER + "\n")
        f.write(f"5-Fold CV Mean Accuracy : {np.mean(fold_accs):.1f}%\n")
        f.write(f"5-Fold CV Mean F1       : {np.mean(fold_f1s):.1f}%\n")
        f.write(SUBDIV + "\n")
        f.write(f"Held-Out Accuracy  : {acc:.1f}%\n")
        f.write(f"Held-Out Precision : {prec:.1f}%\n")
        f.write(f"Held-Out Recall    : {rec:.1f}%\n")
        f.write(f"Held-Out F1 Score  : {f1:.1f}%\n")
        f.write(f"Held-Out AUC-ROC   : {auc:.1f}%\n")
        f.write(SUBDIV + "\n")
        f.write(
            f"Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} "
            f"FN={cm[1][0]} TP={cm[1][1]}\n"
        )
    print(f"  TXT saved -> {REPORT_TXT}")


# ── Mode B: GEE live validation ────────────────────────────────────────────────
def validate_from_gee(rf, xgb):
    import ee
    banner("Mode: GEE  (live Sentinel-1 / CHIRPS / ERA5)")

    haor  = ee.Geometry.Rectangle(HAOR_BBOX)
    slope = get_slope(haor)
    twi   = get_twi(haor)

    rows, true_labels, pred_labels, probs = [], [], [], []

    print(f"\n  {'Period':<22} {'True':<6} {'VV':>6} {'Rain':>8} {'Soil':>6} {'Prob':>6} {'Pred':<8} Risk")
    print("  " + SUBDIV)

    for start_str, end_str, true_flood, label in TEST_PERIODS:
        vv, vh, ratio = get_sentinel1(haor, start_str, end_str)
        ndwi          = get_ndwi(haor, start_str, end_str)
        rain          = get_rainfall(haor, start_str, end_str)
        soil          = get_soil_moisture(haor, start_str, end_str)
        upstream      = get_upstream_vv(haor, start_str, end_str)
        _, temp, wind = get_weather(start_str, end_str)
        forecast      = round(rain * 0.5, 1)

        row = pd.DataFrame(
            [[vv, vh, ratio, ndwi, rain, soil, temp, wind,
              slope, twi, upstream, forecast, forecast * 0.5]],
            columns=FEATURES,
        )

        prob_val  = (RF_WEIGHT  * rf.predict_proba(row)[0][1] +
                     XGB_WEIGHT * xgb.predict_proba(row)[0][1])
        predicted = int(prob_val >= 0.5)

        true_str = "FLOOD" if true_flood else "DRY"
        pred_str = "FLOOD" if predicted else "DRY"
        print(
            f"  {label:<22} {true_str:<6} {vv:>6.1f} {rain:>8.1f} "
            f"{soil:>6.1f} {prob_val:>6.3f} {pred_str:<8} {risk_level(prob_val)}"
        )

        rows.append({
            "period":        label,
            "true_class":    true_str,
            "VV":            round(vv, 1),
            "rainfall_mm":   round(rain, 1),
            "soil_moisture": round(soil, 1),
            "flood_prob":    round(prob_val, 3),
            "predicted":     pred_str,
            "risk":          risk_level(prob_val),
            "correct":       "YES" if predicted == true_flood else "NO",
        })
        true_labels.append(true_flood)
        pred_labels.append(predicted)
        probs.append(prob_val)

    acc  = accuracy_score(true_labels, pred_labels) * 100
    prec = precision_score(true_labels, pred_labels, zero_division=0) * 100
    rec  = recall_score(true_labels, pred_labels, zero_division=0) * 100
    f1   = f1_score(true_labels, pred_labels, zero_division=0) * 100
    try:
        auc = roc_auc_score(true_labels, probs) * 100
    except Exception:
        auc = 0.0
    cm = confusion_matrix(true_labels, pred_labels)

    print("\n  " + SUBDIV)
    print(f"  Accuracy:{acc:.1f}%  Precision:{prec:.1f}%  Recall:{rec:.1f}%  F1:{f1:.1f}%  AUC:{auc:.1f}%")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False, encoding="utf-8")
    print(f"\n  CSV saved -> {REPORT_CSV}")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("HaorFloodAlert - GEE Validation Report\n")
        f.write(DIVIDER + "\n")
        f.write(f"Accuracy  : {acc:.1f}%\n")
        f.write(f"Precision : {prec:.1f}%\n")
        f.write(f"Recall    : {rec:.1f}%\n")
        f.write(f"F1 Score  : {f1:.1f}%\n")
        f.write(f"AUC-ROC   : {auc:.1f}%\n")
        f.write(SUBDIV + "\n")
        f.write(
            f"Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} "
            f"FN={cm[1][0]} TP={cm[1][1]}\n"
        )
    print(f"  TXT saved -> {REPORT_TXT}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    banner("HaorFloodAlert  -  Validation v3  (13 Features)")

    # Check models
    rf_path  = MODELS_DIR / "rf_model.pkl"
    xgb_path = MODELS_DIR / "xgb_model.pkl"

    if not rf_path.exists() or not xgb_path.exists():
        print(f"\n  [ERROR] Models not found in {MODELS_DIR}")
        print("  Run train_real_haor.py first.")
        sys.exit(1)

    print("\n  Loading models...")
    rf  = joblib.load(rf_path)
    xgb = joblib.load(xgb_path)
    print("  [OK] rf_model.pkl  +  xgb_model.pkl loaded")

    # Choose data source
    print("\n  Checking data sources...")
    if os.path.exists(DATA_V2):
        print(f"  [OK] Found {DATA_V2}")
        validate_from_csv(rf, xgb, DATA_V2)
    elif os.path.exists(DATA_V1):
        print(f"  [OK] Found {DATA_V1} (using v1)")
        validate_from_csv(rf, xgb, DATA_V1)
    else:
        print("  CSV not found — trying GEE mode...")
        if _gee_available():
            validate_from_gee(rf, xgb)
        else:
            print("\n  [ERROR] No CSV AND GEE not authenticated.")
            print("  Run generate_synthetic_v2.py to create data/real_training_data_v2.csv")
            sys.exit(1)

    banner("Validation Complete!")


if __name__ == "__main__":
    main()
