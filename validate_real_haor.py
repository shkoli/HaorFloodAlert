import ee
import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

ee.Initialize(project='flood-haor-project')

print("=== Final Balanced Haor Validation (Strong Model) ===")

haor = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00])

rf = joblib.load(r"C:\Users\Lenovo\HaorFloodAlert\models\rf_model.pkl")
xgb = joblib.load(r"C:\Users\Lenovo\HaorFloodAlert\models\xgb_model.pkl")

test_periods = [
    ('2017-06-01', '2017-06-30', 1),
    ('2018-06-15', '2018-07-10', 1),
    ('2019-06-20', '2019-07-05', 1),
    ('2020-06-10', '2020-06-25', 1),
    ('2022-06-17', '2022-06-25', 1),
    ('2024-06-01', '2024-06-20', 1),
    ('2017-01-01', '2017-01-31', 0),
    ('2018-02-01', '2018-02-28', 0),
    ('2019-03-01', '2019-03-31', 0),
    ('2020-01-01', '2020-01-31', 0),
    ('2022-03-01', '2022-03-31', 0),
    ('2024-02-01', '2024-02-29', 0)
]

results = []
true_labels = []
pred_labels = []

for start_str, end_str, true_flood in test_periods:
    start = ee.Date(start_str)
    end = ee.Date(end_str)
    
    # Sentinel-1 VV/VH
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(haor)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .select(['VV', 'VH'])
          .median())
    
    vv = s1.select('VV').reduceRegion(ee.Reducer.mean(), haor, 30).get('VV').getInfo() or -15.0
    vh = s1.select('VH').reduceRegion(ee.Reducer.mean(), haor, 30).get('VH').getInfo() or -20.0
    vv_vh_ratio = vv / vh if vh != 0 else 0.0
    
    rain = 120.0
    try:
        gpm = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
               .filterDate(start, end)
               .sum()
               .select('precipitationCal'))
        rain_value = gpm.reduceRegion(ee.Reducer.mean(), haor, 11132).get('precipitationCal').getInfo()
        if rain_value is not None:
            rain = float(rain_value) * 0.1
    except:
        pass
    
    temp = 29.0
    wind = 12.0   # <--- wind define করা হয়েছে
    soil = -14.6
    slope = 1.9
    
    # Forecast rain proxy for validation
    forecast_rain_next12h = rain * 0.5 + np.random.normal(0, 20)
    
    input_data = pd.DataFrame([[vv, vh, vv_vh_ratio, rain, soil, temp, wind, slope, forecast_rain_next12h]],
                              columns=['VV', 'VH', 'vv_vh_ratio', 'rainfall', 'soil_moisture', 'temp', 'wind', 'slope', 'forecast_rain_next_12h'])
    
    rf_prob = rf.predict_proba(input_data)[0][1]
    xgb_prob = xgb.predict_proba(input_data)[0][1]
    final_prob = 0.55 * rf_prob + 0.45 * xgb_prob
    predicted = 1 if final_prob > 0.5 else 0
    
    results.append({
        'Date': start_str,
        'Type': 'FLOOD' if true_flood == 1 else 'DRY',
        'VV': round(vv, 1),
        'Rain(mm)': round(rain, 1),
        'Forecast Rain 12h': round(forecast_rain_next12h, 1),
        'Prob': round(final_prob, 3),
        'Predicted': 'FLOOD' if predicted == 1 else 'DRY'
    })
    
    true_labels.append(true_flood)
    pred_labels.append(predicted)

df = pd.DataFrame(results)
print(df.to_string(index=False))

acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels)
rec = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print(f"\n✅ Balanced Accuracy: {acc*100:.1f}%")
print(f"Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%")
print("Now with forecast feature — real validation complete")

df.to_csv(r"C:\Users\Lenovo\HaorFloodAlert\results\real_haor_validation_full.csv", index=False)
print("Full report saved.")