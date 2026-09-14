"""
train_augmented.py  —  HaorFloodAlert
Data augmentation for small GEE dataset (40 real rows).
Uses Gaussian noise augmentation inside each LOOCV fold.
No data leakage — test always uses original real samples only.
Cited technique: Shorten & Khoshgoftaar (2019) J.Big Data
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
AUG_FACTOR = 10   # each real row → 10 augmented copies
NOISE_FRAC  = 0.05 # noise = 5% of each feature std

np.random.seed(42)

print("=" * 65)
print("  HaorFloodAlert  -  Augmented Real GEE Training (LOOCV)")
print("=" * 65)

# 1. Load
df = pd.read_csv(CSV_PATH)
label_col = next((c for c in ["flood_label","flood","label","target"]
                  if c in df.columns), None)
X_all = df[FEATURES].values.astype(float)
y_all = df[label_col].values.astype(int)

# 2. Drop zero/near-zero variance features
stds = X_all.std(axis=0)
keep = stds > 1e-6
kept = [f for f, k in zip(FEATURES, keep) if k]
dropped = [f for f, k in zip(FEATURES, keep) if not k]
X = X_all[:, keep]
stds_kept = stds[keep]

print(f"\nDataset: {len(df)} real rows | Flood={int(y_all.sum())} | Dry={int((y_all==0).sum())}")
if dropped:
    print(f"Dropped zero-variance: {dropped}")
print(f"Active features ({len(kept)}): {kept}")
print(f"Augmentation: {AUG_FACTOR}x  (noise={NOISE_FRAC*100:.0f}% of std per feature)")

def augment(X_train, y_train):
    rows, labels = [X_train], [y_train]
    noise_scale = stds_kept * NOISE_FRAC
    for _ in range(AUG_FACTOR - 1):
        noise = np.random.randn(*X_train.shape) * noise_scale
        rows.append(X_train + noise)
        labels.append(y_train)
    return np.vstack(rows), np.concatenate(labels)

# 3. LOOCV — augment inside each fold (no leakage)
print("\nRunning LOOCV with augmentation (40 folds) ...")
loo = LeaveOneOut()
preds, probs, trues = [], [], []

for i, (tr_idx, val_idx) in enumerate(loo.split(X)):
    Xtr, ytr = augment(X[tr_idx], y_all[tr_idx])
    Xval, yval = X[val_idx], y_all[val_idx]

    sp = float((ytr==0).sum()) / max(float(ytr.sum()), 1)

    rf = RandomForestClassifier(n_estimators=300, max_depth=5,
             class_weight="balanced", random_state=42)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
              scale_pos_weight=sp, eval_metric="logloss",
              random_state=42, verbosity=0)
    rf.fit(Xtr, ytr);  xgb.fit(Xtr, ytr)

    p = 0.5*rf.predict_proba(Xval)[:,1] + 0.5*xgb.predict_proba(Xval)[:,1]
    probs.append(float(p[0]))
    preds.append(int(p[0] >= 0.5))
    trues.append(int(yval[0]))
    if (i+1) % 8 == 0:
        print(f"  [{i+1}/40] done", flush=True)

trues = np.array(trues); preds = np.array(preds); probs = np.array(probs)

acc  = accuracy_score(trues, preds)                   * 100
prec = precision_score(trues, preds, zero_division=0) * 100
rec  = recall_score(trues, preds, zero_division=0)    * 100
f1   = f1_score(trues, preds, zero_division=0)        * 100
auc  = roc_auc_score(trues, probs)                    * 100
cm   = confusion_matrix(trues, preds)

print(f"\n  ── LOOCV Results on Real Data (n=40) ──")
print(f"  Accuracy  : {acc:.1f}%")
print(f"  Precision : {prec:.1f}%")
print(f"  Recall    : {rec:.1f}%")
print(f"  F1 Score  : {f1:.1f}%")
print(f"  AUC-ROC   : {auc:.1f}%")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
baseline = int(y_all.sum()) / len(y_all) * 100
print(f"\n  Majority-class baseline: {baseline:.0f}%")
print(f"  Model improvement over baseline: +{acc-baseline:.1f}%")

# 4. Final model trained on ALL 40 rows (augmented)
print("\nTraining final model on all 40 rows (augmented) ...")
X_aug, y_aug = augment(X, y_all)
sp_f = float((y_aug==0).sum()) / max(float(y_aug.sum()), 1)
rf_f = RandomForestClassifier(n_estimators=500, max_depth=5,
           class_weight="balanced", random_state=42, n_jobs=-1)
xgb_f = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
            scale_pos_weight=sp_f, eval_metric="logloss",
            random_state=42, verbosity=0)
rf_f.fit(X_aug, y_aug);  xgb_f.fit(X_aug, y_aug)
print("  [OK] Final models ready")

print("\n  ── Feature Importance (RF) ──")
for feat, imp in sorted(zip(kept, rf_f.feature_importances_), key=lambda x: -x[1]):
    bar = chr(9608) * int(imp * 40)
    print(f"  {feat:<30} {bar} {imp:.4f}")

# 5. Save
MODELS_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
joblib.dump(rf_f,  MODELS_DIR / "rf_model.pkl")
joblib.dump(xgb_f, MODELS_DIR / "xgb_model.pkl")
joblib.dump(kept,  MODELS_DIR / "active_features.pkl")
print(f"\n  [OK] Models saved -> {MODELS_DIR}")

with open(RESULTS_DIR / "training_report.txt", "w", encoding="utf-8") as fh:
    fh.write("HaorFloodAlert - Real GEE Training Report\n")
    fh.write("=" * 50 + "\n")
    fh.write(f"Dataset   : real_training_data_v2.csv\n")
    fh.write(f"Rows      : {len(df)} real GEE satellite observations (2017-2024)\n")
    fh.write(f"Augmented : {len(X_aug)} rows ({AUG_FACTOR}x, Gaussian noise {NOISE_FRAC*100:.0f}% std)\n")
    fh.write(f"Flood     : {int(y_all.sum())}   Dry: {int((y_all==0).sum())}\n")
    fh.write(f"Features  : {len(kept)} active\n\n")
    fh.write(f"Evaluation: LOOCV on original 40 real rows (no augmented samples in test)\n")
    fh.write(f"  Accuracy  : {acc:.1f}%\n")
    fh.write(f"  Precision : {prec:.1f}%\n")
    fh.write(f"  Recall    : {rec:.1f}%\n")
    fh.write(f"  F1 Score  : {f1:.1f}%\n")
    fh.write(f"  AUC-ROC   : {auc:.1f}%\n")
    fh.write(f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}\n")

print(f"  Report -> {RESULTS_DIR / 'training_report.txt'}")
print("\n" + "=" * 65)
print(f"  FINAL RESULT:  Acc={acc:.1f}%  F1={f1:.1f}%  AUC={auc:.1f}%")
print(f"  Method: Gaussian augmentation (10x) + LOOCV")
print(f"  Reference: Shorten & Khoshgoftaar (2019), J. Big Data")
print("=" * 65)
