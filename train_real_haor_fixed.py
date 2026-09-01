import ee
import joblib
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=== Training with Forecast Proxy (Real Haor Data) ===")

ee.Initialize(project='flood-haor-project')

haor = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00])

# Historical periods (add more if needed - minimum 40-50 for real model)
periods = [
    ('2017-06-01', '2017-06-30', 1),
    ('2017-07-01', '2017-07-15', 1),
    ('2018-06-15', '2018-07-10', 1),
    ('2018-07-11', '2018-07-25', 1),
    ('2019-06-20', '2019-07-05', 1),
    ('2019-07-06', '2019-07-20', 1),
    ('2020-06-10', '2020-06-25', 1),
    ('2022-06-17', '2022-06-25', 1),
    ('2024-06-01', '2024-06-20', 1),
    ('2017-01-01', '2017-01-15', 0),
    ('2018-02-01', '2018-02-15', 0),
    ('2023-03-01', '2023-03-15', 0),
    ('2017-02-01', '2017-02-15', 0),
    ('2018-03-01', '2018-03-15', 0),
    ('2019-01-01', '2019-01-15', 0),
    ('2020-01-01', '2020-01-15', 0),
    ('2021-01-01', '2021-01-15', 0),
    ('2022-01-01', '2022-01-15', 0),
    ('2023-01-01', '2023-01-15', 0),
    ('2024-01-01', '2024-01-15', 0),
    ('2017-08-01', '2017-08-15', 1),
    ('2018-08-01', '2018-08-15', 1),
    ('2019-08-01', '2019-08-15', 1),
    ('2020-07-01', '2020-07-15', 1),
    ('2021-07-01', '2021-07-15', 1),
    ('2022-07-01', '2022-07-15', 1),
    ('2023-07-01', '2023-07-15', 1),
    ('2024-07-01', '2024-07-15', 1),
    ('2017-05-01', '2017-05-15', 1),
    ('2018-05-01', '2018-05-15', 1),
    ('2019-05-01', '2019-05-15', 1),
    ('2020-05-01', '2020-05-15', 1),
    ('2021-05-01', '2021-05-15', 1),
    ('2022-05-01', '2022-05-15', 1),
    ('2023-05-01', '2023-05-15', 1),
    ('2024-05-01', '2024-05-15', 1),
    ('2017-09-01', '2017-09-15', 0),
    ('2018-09-01', '2018-09-15', 0),
    ('2019-09-01', '2019-09-15', 0),
    ('2020-09-01', '2020-09-15', 0),
    # আরও যোগ কর — minimum 40-50 periods চাই
]

X = []
y = []

for start_str, end_str, label in periods:
    start = ee.Date(start_str)
    end = ee.Date(end_str)
    
    # Sentinel-1 VV/VH - VH mandatory filter to fix band mismatch error
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(haor)
          .filterDate(start, end)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))  # THIS LINE FIXES YOUR GEE ERROR
          .median())
    
    vv = s1.select('VV').reduceRegion(ee.Reducer.mean(), haor, 30).get('VV').getInfo() or -15.0
    vh = s1.select('VH').reduceRegion(ee.Reducer.mean(), haor, 30).get('VH').getInfo() or -20.0
    vv_vh_ratio = vv / vh if vh != 0 else 0.0
    
    # Slope from SRTM
    slope = 1.9
    try:
        dem = ee.Image('USGS/SRTMGL1_003')
        slope = float(ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), haor, 30).get('slope').getInfo() or 1.9)
    except:
        pass
    
    # Open-Meteo historical data
    rain = 0.0
    temp = 25.0
    wind = 5.0
    soil_proxy = -14.6  # current proxy
    
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude=24.87&longitude=91.45&start_date={start_str}&end_date={end_str}&daily=precipitation_sum,temperature_2m_mean,wind_speed_10m_max&timezone=Asia/Dhaka"
        r = requests.get(url, timeout=20).json()['daily']
        rain = round(sum(r['precipitation_sum']), 1)
        temp = round(sum(r['temperature_2m_mean']) / len(r['temperature_2m_mean']), 1)
        wind = round(max(r['wind_speed_10m_max']), 1)
    except:
        pass
    
    # Forecast proxy for training (no real historical forecast, so 50% of rain + noise)
    forecast_rain_proxy = rain * 0.5 + np.random.normal(0, 20)
    
    # Append row
    X.append([vv, vh, vv_vh_ratio, rain, soil_proxy, temp, wind, slope, forecast_rain_proxy])
    y.append(label)

X = pd.DataFrame(X, columns=['VV', 'VH', 'vv_vh_ratio', 'rainfall', 'soil_moisture', 'temp', 'wind', 'slope', 'forecast_rain_next_12h'])
y = pd.Series(y)

print(f"Dataset size: {len(X)} samples")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=12,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=700,
    max_depth=9,
    learning_rate=0.05,
    scale_pos_weight=3.5,
    random_state=42
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

joblib.dump(rf, r"C:\Users\Lenovo\HaorFloodAlert\models\rf_model.pkl")
joblib.dump(xgb, r"C:\Users\Lenovo\HaorFloodAlert\models\xgb_model.pkl")

# Quick test on hold-out set
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
ensemble_prob = 0.55 * rf.predict_proba(X_test)[:,1] + 0.45 * xgb.predict_proba(X_test)[:,1]
ensemble_pred = (ensemble_prob > 0.5).astype(int)

print(f"RF Accuracy: {accuracy_score(y_test, rf_pred)*100:.1f}%")
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred)*100:.1f}%")
print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred)*100:.1f}%")
print(f"Recall (flood class): {recall_score(y_test, ensemble_pred)*100:.1f}%")
print("Model retrained with forecast proxy. Now run app to test live forecast.")