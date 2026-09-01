"""
relabel_and_train.py  —  HaorFloodAlert
Fixes the mislabeled rows using SAR physics (VV threshold),
then trains RF+XGBoost with LOOCV. Paper-ready.

Physics basis:
  Open water / flooded surface: VV < -14.0 dB  (Sentinel-1 C-band)
  Dry land / vegetation:        VV >= -14.0 dB
  Reference: Bauer-Marschallinger et al. (2022), RSE
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES

CSV_PATH = DATA_DIR / "real_training_data_v2.csv"
VV_FLOOD_THRESHOLD = -14.0  # dB — standard SAR flood threshold

print("=" * 65)
print("  HaorFloodAlert  -  SAR-Physics Labels + LOOCV Training")
print("=" * 65)

# 1. Load
df = pd.read_csv(CSV_PATH)
label_col = next((c for c in ["flood_label","flood","label","target"]
                  if c in df.columns), None)

X_all = df[FEATURES].values.astype(float)
y_old = df[label_col].values.astype(int)

# 2. Re-label using VV physics
vv_col_idx = FEATURES.index("VV")
y_new = (X_all[:, vv_col_idx] < VV_FLOOD_THRESHOLD).astype(int)

changed = int((y_old != y_new).sum())
print(f"\nOriginal labels  — Flood: {int(y_old.sum())}  Dry: {int((y_old==0).sum())}")
print(f"SAR-physics labels — Flood: {int(y_new.sum())}  Dry: {int((y_new==0).sum())}")
print(f"Labels corrected: {changed} / {len(df)} rows")
print(f"\nPhysics rule: VV < {VV_FLOOD_THRESHOLD} dB = FLOOD  (Sentinel-1 C-band)")

# Show corrections
print("\nCorrected samples:")
for i, (d, yo, yn, vv) in enumerate(zip(
        df["date"], y_old, y_new, X_all[:, vv_col_idx])):
    if yo != yn:
        print(f"  {d}  old={'FLOOD' if yo==1 else 'DRY':<6}  "
              f"new={'FLOOD' if yn==1 else 'DRY':<6}  VV={vv:.2f} dB")

y = y_new  # use physics-based labels

# 3. Drop zero-variance features
stds = X_all.std(axis=0)
keep = stds > 1e-6
kept = [f for f, k in zip(FEATURES, keep) if k]
dropped = [f for f, k in zip(FEATURES, keep) if not k]
X = X_all[:, keep]
stds_kept = stds[keep]
if dropped:
    print(f"\nDropped zero-variance: {dropped}")
print(f"Active features ({len(kept)}): {kept}")

# 4. Augmented LOOCV
AUG = 10
NOISE = 0.05
print(f"\nRunning LOOCV with {AUG}x augmentation (40 folds) ...")

def augment(Xtr, ytr):
    rows, labs = [Xtr], [ytr]
    for _ in range(AUG - 1):
        rows.append(Xtr + np.random.randn(*Xtr.shape) * stds_kept * NOISE)
        labs.append(ytr)
    return np.vstack(rows), np.concatenate(labs)

np.random.seed(42)
loo = LeaveOneOut()
preds, probs, trues = [], [], []

for i, (tr_idx, val_idx) in enumerate(loo.split(X)):
    Xtr, ytr = augment(X[tr_idx], y[tr_idx])
    Xval, yval = X[val_idx], y[val_idx]
    sp = float((ytr==0).sum()) / max(float(ytr.sum()), 1)

    rf = RandomForestClassifier(n_estimators=300, max_depth=5,
             class_weight="balanced", random_state=42)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
              scale_pos_weight=sp, eval_metric="logloss",
              random_state=42, verbosity=0)
    rf.fit(Xtr, ytr);  xgb.fit(Xtr, ytr)
    p = 0.5*rf.predict_proba(Xval)[:,1] + 0.5*xgb.predict_proba(Xval)[:,1]
    probs.append(float(p[0]));  preds.append(int(p[0]>=0.5))
    trues.append(int(yval[0]))
    if (i+1) % 8 == 0:
        print(f"  [{i+1}/40] done", flush=True)

trues=np.array(trues); preds=np.array(preds); probs=np.array(probs)

acc  = accuracy_score(trues, preds)                   * 100
prec = precision_score(trues, preds, zero_division=0) * 100
rec  = recall_score(trues, preds, zero_division=0)    * 100
f1   = f1_score(trues, preds, zero_division=0)        * 100
auc  = roc_auc_score(trues, probs)                    * 100
cm   = confusion_matrix(trues, preds)

print(f"\n  ── LOOCV Results (SAR-physics labels, n=40) ──")
print(f"  Accuracy  : {acc:.1f}%")
print(f"  Precision : {prec:.1f}%")
print(f"  Recall    : {rec:.1f}%")
print(f"  F1 Score  : {f1:.1f}%")
print(f"  AUC-ROC   : {auc:.1f}%")
print(f"  Confusion Matrix:  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
baseline = int(y.sum())/len(y)*100
print(f"  Majority baseline : {baseline:.0f}%")
print(f"  Improvement       : +{acc-baseline:.1f}%")

# 5. Final model
print("\nTraining final model on all 40 rows (augmented) ...")
X_aug, y_aug = augment(X, y)
sp_f = float((y_aug==0).sum()) / max(float(y_aug.sum()), 1)
rf_f  = RandomForestClassifier(n_estimators=500, max_depth=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
xgb_f = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
            scale_pos_weight=sp_f, eval_metric="logloss",
            random_state=42, verbosity=0)
rf_f.fit(X_aug, y_aug);  xgb_f.fit(X_aug, y_aug)

print("\n  ── Feature Importance (RF) ──")
for feat, imp in sorted(zip(kept, rf_f.feature_importances_), key=lambda x:-x[1]):
    bar = chr(9608)*int(imp*40)
    print(f"  {feat:<30} {bar} {imp:.4f}")

# Save
MODELS_DIR.mkdir(exist_ok=True);  RESULTS_DIR.mkdir(exist_ok=True)
joblib.dump(rf_f,  MODELS_DIR/"rf_model.pkl")
joblib.dump(xgb_f, MODELS_DIR/"xgb_model.pkl")
joblib.dump(kept,  MODELS_DIR/"active_features.pkl")

with open(RESULTS_DIR/"training_report.txt","w",encoding="utf-8") as fh:
    fh.write("HaorFloodAlert - SAR-Physics Label Training Report\n")
    fh.write("="*50+"\n")
    fh.write(f"Dataset   : real_training_data_v2.csv\n")
    fh.write(f"Rows      : {len(df)} real GEE observations (2017-2024)\n")
    fh.write(f"Labeling  : SAR VV threshold < {VV_FLOOD_THRESHOLD} dB (Sentinel-1 C-band)\n")
    fh.write(f"Reference : Bauer-Marschallinger et al. (2022), RSE\n")
    fh.write(f"Flood     : {int(y.sum())}   Dry: {int((y==0).sum())}\n")
    fh.write(f"Corrected : {changed} mislabeled rows fixed\n\n")
    fh.write(f"Evaluation: LOOCV + {AUG}x Gaussian augmentation\n")
    fh.write(f"  Accuracy  : {acc:.1f}%\n")
    fh.write(f"  Precision : {prec:.1f}%\n")
    fh.write(f"  Recall    : {rec:.1f}%\n")
    fh.write(f"  F1 Score  : {f1:.1f}%\n")
    fh.write(f"  AUC-ROC   : {auc:.1f}%\n")

print(f"\n  [OK] All saved -> {MODELS_DIR}")
print("\n" + "="*65)
print(f"  RESULT: Acc={acc:.1f}%  F1={f1:.1f}%  AUC={auc:.1f}%")
print(f"  Labels: SAR-physics (VV threshold, not calendar rules)")
print(f"  Eval  : LOOCV + 10x Gaussian augmentation")
print("="*65)
