import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

print("=== Haor Flash Flood Model Retraining (Strong Signal) ===")

root_dir = r"C:\Users\Lenovo\HaorFloodAlert"
os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)

np.random.seed(42)
n_samples = 15000

# Stronger haor logic: low VV + high rain + high soil = flood
vv = np.random.normal(-19, 6, n_samples)
vh = np.random.normal(-23, 5, n_samples)
rain = np.random.normal(200, 100, n_samples)
soil = np.random.normal(48, 12, n_samples)
slope = np.random.normal(2.0, 1.0, n_samples)
temp = np.random.normal(29, 3, n_samples)
wind = np.random.normal(12, 5, n_samples)

flood = ((vv < -16) | (rain > 160) | (soil > 42)).astype(int)  # stronger OR condition

data = pd.DataFrame({
    'VV': vv, 'VH': vh, 'rainfall': rain, 'soil_moisture': soil,
    'temp': temp, 'wind': wind, 'slope': slope, 'flood': flood
})

X = data.drop('flood', axis=1)
y = data['flood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42, class_weight='balanced')
xgb = XGBClassifier(n_estimators=400, max_depth=10, learning_rate=0.05, random_state=42, scale_pos_weight=1.5)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

joblib.dump(rf, os.path.join(root_dir, "models", "rf_model.pkl"))
joblib.dump(xgb, os.path.join(root_dir, "models", "xgb_model.pkl"))

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
xgb_prob = xgb.predict_proba(X_test)[:, 1]

print(f"RF Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}% | AUC: {roc_auc_score(y_test, rf_prob):.4f}")
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred)*100:.2f}% | AUC: {roc_auc_score(y_test, xgb_prob):.4f}")
print("Feature Importance (RF):", dict(zip(X.columns, rf.feature_importances_.round(3))))
print("✅ Model retrained with STRONG flood signal (low VV OR high rain = flood)")
print("lstm_model.h5 already deleted → no crashes")