"""
train_real_v2.py  —  HaorFloodAlert
Proper training on 40 real GEE rows:
  - Drops zero-variance features automatically
  - Uses Leave-One-Out CV (correct for n=40)
  - Simpler model to avoid overfitting
  - Honest paper-ready metrics
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES

CSV_PATH = DATA_DIR / "real_training_data_v2.csv"

print("=" * 65)
print("  HaorFloodAlert  -  Real GEE Training v2 (LOOCV)")
print("=" * 65)

# 1. Load
df = pd.read_csv(CSV_PATH)
label_col = next((c for c in ["flood_label","flood","label","target"] if c in df.columns), None)
X_full = df[FEATURES].values
y = df[label_col].values
print(f"\nLoaded: {len(df)} rows | Flood={int(y.sum())} | Dry={int((y==0).sum())}")

# 2. Drop zero-variance features
variances = X_full.var(axis=0)
keep_mask = variances > 1e-6
kept_features = [f for f, k in zip(FEATURES, keep_mask) if k]
dropped = [f for f, k in zip(FEATURES, keep_mask) if not k]
X = X_full[:, keep_mask]

print(f"\nFeature check:")
if dropped:
    print(f"  [WARN] Zero-variance features DROPPED: {dropped}")
    print(f"  (These had same value for all 40 rows — no signal)")
print(f"  [OK] Using {len(kept_features)} features: {kept_features}")

# 3. Leave-One-Out CV (best for n=40)
print("\nRunning Leave-One-Out CV (40 folds) ...")
loo = LeaveOneOut()
loo_preds, loo_probs, loo_true = [], [], []

for tr_idx, val_idx in loo.split(X):
    Xtr, Xval = X[tr_idx], X[val_idx]
    ytr, yval = y[tr_idx], y[val_idx]
    sp = float((ytr==0).sum()) / max(float(ytr.sum()), 1)

    rf_l = RandomForestClassifier(
        n_estimators=200, max_depth=4,
        class_weight="balanced", random_state=42
    )
    xgb_l = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        scale_pos_weight=sp, eval_metric="logloss",
        random_state=42, verbosity=0
    )
    rf_l.fit(Xtr, ytr)
    xgb_l.fit(Xtr, ytr)

    p = 0.5*rf_l.predict_proba(Xval)[:,1] + 0.5*xgb_l.predict_proba(Xval)[:,1]
    loo_probs.append(float(p[0]))
    loo_preds.append(int(p[0] >= 0.5))
    loo_true.append(int(yval[0]))

loo_true  = np.array(loo_true)
loo_preds = np.array(loo_preds)
loo_probs = np.array(loo_probs)

loo_acc  = accuracy_score(loo_true, loo_preds)  * 100
loo_prec = precision_score(loo_true, loo_preds, zero_division=0) * 100
loo_rec  = recall_score(loo_true, loo_preds, zero_division=0)    * 100
loo_f1   = f1_score(loo_true, loo_preds, zero_division=0)        * 100
loo_auc  = roc_auc_score(loo_true, loo_probs) * 100
loo_cm   = confusion_matrix(loo_true, loo_preds)

print(f"\n  ── Leave-One-Out CV Results (n=40) ──")
print(f"  Accuracy  : {loo_acc:.1f}%")
print(f"  Precision : {loo_prec:.1f}%")
print(f"  Recall    : {loo_rec:.1f}%")
print(f"  F1 Score  : {loo_f1:.1f}%")
print(f"  AUC-ROC   : {loo_auc:.1f}%")
print(f"  Confusion Matrix:")
print(f"    TN={loo_cm[0,0]}  FP={loo_cm[0,1]}")
print(f"    FN={loo_cm[1,0]}  TP={loo_cm[1,1]}")

# 4. Train final model on ALL 40 rows
print("\nTraining final model on all 40 real rows ...")
sp_final = float((y==0).sum()) / max(float(y.sum()), 1)
rf_final = RandomForestClassifier(
    n_estimators=500, max_depth=4,
    class_weight="balanced", random_state=42, n_jobs=-1
)
xgb_final = XGBClassifier(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    scale_pos_weight=sp_final, eval_metric="logloss",
    random_state=42, verbosity=0
)
rf_final.fit(X, y)
xgb_final.fit(X, y)
print("  [OK] Final models trained on all 40 real rows")

# Feature importance
print("\n  ── Feature Importance (RF) ──")
for feat, imp in sorted(zip(kept_features, rf_final.feature_importances_), key=lambda x: -x[1]):
    bar = chr(9608) * int(imp * 40)
    print(f"  {feat:<30} {bar} {imp:.4f}")

# 5. Save — wrap models with feature selector metadata
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
joblib.dump(rf_final,  MODELS_DIR / "rf_model.pkl")
joblib.dump(xgb_final, MODELS_DIR / "xgb_model.pkl")
joblib.dump(kept_features, MODELS_DIR / "active_features.pkl")
print(f"\n  [OK] rf_model.pkl  -> {MODELS_DIR}")
print(f"  [OK] xgb_model.pkl -> {MODELS_DIR}")
print(f"  [OK] active_features.pkl -> {MODELS_DIR}")

# Save report
with open(RESULTS_DIR / "training_report.txt", "w", encoding="utf-8") as f:
    f.write("HaorFloodAlert - Real GEE Training Report\n")
    f.write("=" * 50 + "\n")
    f.write(f"Dataset   : real_training_data_v2.csv (GEE satellite data)\n")
    f.write(f"Rows      : {len(df)} (2017-2024 Sunamganj Haor)\n")
    f.write(f"Flood     : {int(y.sum())}   Dry: {int((y==0).sum())}\n")
    f.write(f"Features  : {len(kept_features)} active (dropped: {dropped})\n\n")
    f.write(f"Evaluation: Leave-One-Out CV (LOOCV, n=40)\n")
    f.write(f"  Accuracy  : {loo_acc:.1f}%\n")
    f.write(f"  Precision : {loo_prec:.1f}%\n")
    f.write(f"  Recall    : {loo_rec:.1f}%\n")
    f.write(f"  F1 Score  : {loo_f1:.1f}%\n")
    f.write(f"  AUC-ROC   : {loo_auc:.1f}%\n")

print(f"  Report -> {RESULTS_DIR / 'training_report.txt'}")
print("\n" + "=" * 65)
print(f"  REAL RESULT (LOOCV):  Acc={loo_acc:.1f}%  F1={loo_f1:.1f}%  AUC={loo_auc:.1f}%")
print(f"  Trained on: {len(df)} real GEE satellite observations")
print(f"  Evaluation: Leave-One-Out (most honest method for n<50)")
print("  This result is valid for conference paper.")
print("=" * 65)
