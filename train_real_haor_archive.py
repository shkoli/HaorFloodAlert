import ee
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

ee.Initialize(project='flood-haor-project')
haor = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00])

dates = [
    ('2017-06-01', 1), ('2018-06-15', 1), ('2019-06-20', 1),
    ('2020-06-10', 1), ('2022-06-17', 1), ('2024-06-01', 1),
    ('2017-01-01', 0), ('2018-02-01', 0), ('2023-03-01', 0),
    ('2021-12-15', 0)
]

data = []
for date_str, flood in dates:
    date = ee.Date(date_str)
    start = date.advance(-3, 'day')
    end = date.advance(3, 'day')
    
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(haor)
          .filterDate(start, end)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .median())
    vv = s1.select('VV').reduceRegion(ee.Reducer.mean(), haor, 30).get('VV').getInfo() or -15.0
    vh = s1.select('VH').reduceRegion(ee.Reducer.mean(), haor, 30).get('VH').getInfo() or -20.0
    
    rain = 85.0
    try:
        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start, end)
                  .sum()
                  .select('precipitation'))
        rain_value = chirps.reduceRegion(ee.Reducer.mean(), haor, 5566).get('precipitation').getInfo()
        if rain_value is not None and rain_value > 10:
            rain = float(rain_value)
    except:
        pass
    
    data.append([vv, vh, rain, 38.0, 29.0, 12.0, 1.9, flood])

df = pd.DataFrame(data, columns=['VV', 'VH', 'rainfall', 'soil_moisture', 'temp', 'wind', 'slope', 'flood'])

X = df.drop('flood', axis=1).values
y = df['flood'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RF + XGB (stable)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced')
xgb = XGBClassifier(n_estimators=500, scale_pos_weight=3)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

lstm = LSTMModel()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
criterion = nn.BCELoss()

X_torch = torch.FloatTensor(X_train)
y_torch = torch.FloatTensor(y_train).unsqueeze(1)
for epoch in range(100):
    optimizer.zero_grad()
    out = lstm(X_torch)
    loss = criterion(out, y_torch)
    loss.backward()
    optimizer.step()

torch.save(lstm.state_dict(), r"C:\Users\Lenovo\HaorFloodAlert\models\lstm_model.pth")

joblib.dump(rf, r"C:\Users\Lenovo\HaorFloodAlert\models\rf_model.pkl")
joblib.dump(xgb, r"C:\Users\Lenovo\HaorFloodAlert\models\xgb_model.pkl")

print("=== LSTM + RF + XGB TRAINED ===")
print("Models saved. Now run Streamlit.")