import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES

CSV_PATH = DATA_DIR / "real_training_data_v2.csv"

print("=" * 65)
print("  HaorFloodAlert  -  Train on REAL GEE Data (13 features)")
print("=" * 65)

df = pd.read_csv(CSV_PATH)
label_col = next((c for c in ["flood_label","flood","label","target"] if c in df.columns), None)
assert label_col, f"No label column. Columns: {list(df.columns)}"
missing = [f for f in FEATURES if f not in df.columns]
assert not missing, f"Missing features: {missing}"

X = df[FEATURES].values
y = df[label_col].values
print(f"Rows: {len(df)}  |  Flood: {int(y.sum())}  |  Dry: {int((y==0).sum())}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("[OK] Random Forest trained")

sp = float((y_train==0).sum()) / max(float(y_train.sum()), 1)
xgb = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, scale_pos_weight=sp, eval_metric="logloss", random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
print("[OK] XGBoost trained")

ens  = 0.5*rf.predict_proba(X_test)[:,1] + 0.5*xgb.predict_proba(X_test)[:,1]
pred = (ens >= 0.5).astype(int)
acc  = accuracy_score(y_test, pred)*100
prec = precision_score(y_test, pred, zero_division=0)*100
rec  = recall_score(y_test, pred, zero_division=0)*100
f1   = f1_score(y_test, pred, zero_division=0)*100
auc  = roc_auc_score(y_test, ens)*100
cm   = confusion_matrix(y_test, pred)

print(f"\nHeld-Out Test (n={len(X_test)})")
print(f"  Accuracy  : {acc:.1f}%")
print(f"  Precision : {prec:.1f}%")
print(f"  Recall    : {rec:.1f}%")
print(f"  F1 Score  : {f1:.1f}%")
print(f"  AUC-ROC   : {auc:.1f}%")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

print("\n5-Fold Stratified CV")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []
for fold, (tr, val) in enumerate(skf.split(X, y)):
    sp2 = float((y[tr]==0).sum()) / max(float(y[tr].sum()), 1)
    r = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    g = XGBClassifier(n_estimators=300, max_depth=4, scale_pos_weight=sp2, eval_metric="logloss", random_state=42, verbosity=0)
    r.fit(X[tr], y[tr]); g.fit(X[tr], y[tr])
    p = 0.5*r.predict_proba(X[val])[:,1] + 0.5*g.predict_proba(X[val])[:,1]
    a = accuracy_score(y[val], (p>=0.5).astype(int))*100
    cv_acc.append(a)
    print(f"  Fold {fold+1}: {a:.1f}%")

cv_mean = np.mean(cv_acc)
cv_std  = np.std(cv_acc)
print(f"  CV Mean: {cv_mean:.1f}%  (+/- {cv_std:.1f}%)")

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
joblib.dump(rf,  MODELS_DIR / "rf_model.pkl")
joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
print(f"\n[OK] Models saved -> {MODELS_DIR}")
print(f"\nREAL DATA RESULT:  CV={cv_mean:.1f}%  Held-out={acc:.1f}%")
print("Paper ready!")