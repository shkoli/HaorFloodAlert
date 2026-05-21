"""
train_ensemble.py  —  HaorFloodAlert RF + XGBoost ensemble trainer.

Legacy script retained for reference. Uses synthetic feature distributions
calibrated to Sunamganj Haor physical ranges. For primary training on real
Sentinel-1 SAR events use train_honest.py instead.

Note: models saved by this script are trained on synthetic data and are
NOT suitable for production deployment without re-validation.

Usage
-----
    python train_ensemble.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from config import MODELS_DIR

np.random.seed(42)
N_SAMPLES = 15000

print("=" * 60)
print("  HaorFloodAlert — Synthetic Ensemble Trainer (legacy)")
print("  For real-SAR training use: python train_honest.py")
print("=" * 60)


def generate_synthetic_data(n):
    """
    Generate synthetic training data using haor-calibrated feature distributions.

    Flood condition: low VV backscatter OR heavy rainfall OR high soil moisture.
    Feature ranges are calibrated to Sunamganj Haor physical observations.
    """
    vv   = np.random.normal(-19, 6, n)
    vh   = np.random.normal(-23, 5, n)
    rain = np.random.normal(200, 100, n)
    soil = np.random.normal(48, 12, n)
    slope = np.random.normal(2.0, 1.0, n)
    temp = np.random.normal(29, 3, n)
    wind = np.random.normal(12, 5, n)

    flood = ((vv < -16) | (rain > 160) | (soil > 42)).astype(int)

    return pd.DataFrame({
        "VV": vv, "VH": vh, "rainfall": rain,
        "soil_moisture": soil, "temp": temp,
        "wind": wind, "slope": slope, "flood": flood,
    })


data = generate_synthetic_data(N_SAMPLES)
X = data.drop("flood", axis=1)
y = data["flood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=400, max_depth=15,
    random_state=42, class_weight="balanced",
)
xgb = XGBClassifier(
    n_estimators=400, max_depth=10, learning_rate=0.05,
    random_state=42, scale_pos_weight=1.5,
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

rf_acc  = accuracy_score(y_test, rf.predict(X_test)) * 100
xgb_acc = accuracy_score(y_test, xgb.predict(X_test)) * 100
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

print(f"\nRF  Accuracy: {rf_acc:.2f}%  AUC: {rf_auc:.4f}")
print(f"XGB Accuracy: {xgb_acc:.2f}%  AUC: {xgb_auc:.4f}")

joblib.dump(rf,  MODELS_DIR / "rf_model.pkl")
joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
print(f"\nModels saved to {MODELS_DIR}")
print("Run train_honest.py for validated real-SAR models.")
