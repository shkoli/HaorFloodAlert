# HaorFloodAlert

**72-hour flood early warning system for the Sunamganj Haor, Bangladesh**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)

---

## Overview

HaorFloodAlert is a machine-learning flood prediction system designed specifically for the haor wetlands of Sunamganj, Bangladesh. It combines Sentinel-1 SAR satellite imagery, meteorological forecasts, and upstream Barak river discharge monitoring to issue 72-hour flood probability forecasts for ~8,000 km² of haor area.

**Developed by:** Salma Hoque Talukdar Koli
**Institution:** RTM Al-Kabir Technical University, Sylhet, Bangladesh — CSE Thesis 2026
**Supervisor:** Md. Samiul Alim

---

## Key Features

| Feature | Description |
|---------|-------------|
| 72-hour forecast | Three 24-hour windows (0–24h, 24–48h, 48–72h) with per-window flood probability |
| 3-layer model | RF+XGB ensemble + Barak discharge level + 14-day rising trend |
| SAR integration | Sentinel-1 Otsu change detection for real flood extent mapping |
| Upstream monitoring | GloFAS discharge at Silchar (Barak river) provides ~36h lead time |
| Boro rice damage | Flood depth × duration × BRRI growth stage → yield loss and BDT economic loss |
| Bengali alerts | Gmail SMTP community alerts in Bengali and English |

---

## Validated Performance

| Split | Accuracy | Recall | AUC-ROC | F1 |
|-------|----------|--------|---------|-----|
| LOOCV — 77 real-SAR events (primary) | **89.6%** | 87.5% | 94.3%| 86.2%|
| Hold-out — 45 events, 5-seed mean | 86.7% | — | — | — |

> LSTM component (weight 0.20) trains on synthetic sequences and is excluded
> from the primary accuracy metric. The reported numbers use RF (0.45) + XGBoost (0.35)
> with weights renormalized to sum to 1.

---

## Architecture

```
Input (11 active features)
    Sentinel-1 VV/VH/ratio  |  NDWI  |  rainfall  |  soil moisture
    temp_anomaly  |  wind  |  slope  |  TWI  |  upstream_vv
    forecast_rain_next_12h  |  forecast_rain_72h

         |
         v
Layer 1  RF (w=0.5625) + XGBoost (w=0.4375)  ->  p_base
         |
         v
Layer 2  Barak discharge (GloFAS / Open-Meteo)
         Q > 7500 m3/s  -> +15 pp
         Q > 6000 m3/s  -> +10 pp
         |
         v
Layer 3  14-day rising trend (OLS, R2-gated)
         slope > 500 m3/s/day  -> +15 pp
         slope > 300            -> +10 pp
         slope > 100            ->  +5 pp
         |
         v
p_final = min(p_base + L2 + L3, 0.95)
```

---

## Project Structure

```
HaorFloodAlert/
├── config.py                   # All constants, feature list, thresholds
├── train_honest.py             # Primary trainer — LOOCV on real SAR events
├── train_lstm.py               # LSTM trainer (synthetic sequences)
├── train_ensemble.py           # Legacy RF+XGB trainer (synthetic)
├── collect_honest_data.py      # GEE data collection pipeline
├── paper_figures.py            # Reproducible paper figures (9 IEEE figs)
├── requirements.txt
├── .env.example                # Environment variable template
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lstm_model.pth
│   ├── lstm_scaler.pkl
│   ├── lstm_window.pkl
│   └── active_features.pkl
├── data/
│   └── honest_training_data_v2.csv   # 101 events (77 real-SAR + 24 proxy)
├── utils/
│   ├── predict.py              # 3-layer prediction engine
│   ├── gee_features.py         # Live GEE feature extraction
│   ├── upstream_discharge.py   # Barak/GloFAS discharge fetch
│   └── discharge_trend.py      # OLS trend analysis + R2 gate
├── streamlit_app/
│   ├── app.py                  # Home page
│   └── pages/
│       ├── 1_Prediction.py     # Live 72h forecast
│       ├── 2_Map.py            # SAR flood extent map
│       ├── 3_Alerts.py         # Email alert system
│       ├── 4_About.py          # Methodology
│       ├── 5_Validation.py     # Accuracy details
│       └── 6_CropDamage.py     # Boro rice damage estimator
├── gee_scripts/
│   ├── 01_sentinel1_flood_detection.js
│   ├── 02_sentinel2_ndwi_mapping.js
│   └── 03_chirps_rainfall_analysis.js
└── docs/
    └── figures/
        ├── fig1_confusion_matrix.png
        ├── fig5_temp_confound.png
        ├── fig9_study_area.png
        └── fig_dashboard_mockup.png
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/shkoli/HaorFloodAlert.git
cd HaorFloodAlert
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in GEE_PROJECT, ALERT_EMAIL_SENDER, ALERT_APP_PASSWORD
```

### 3. Authenticate Google Earth Engine

```bash
earthengine authenticate
```

### 4. Run the dashboard

```bash
cd streamlit_app
streamlit run app.py
```

### 5. Retrain models (optional)

```bash
python train_honest.py              # RF + XGBoost, LOOCV validation
python train_lstm.py                # LSTM (synthetic sequences)
```

---

## Data Sources

| Source | Variable | Access |
|--------|----------|--------|
| Copernicus Sentinel-1 GRD | VV, VH backscatter | Google Earth Engine |
| Copernicus Sentinel-2 SR | NDWI | Google Earth Engine |
| UCSB CHIRPS Daily | 7-day cumulative rainfall | Google Earth Engine |
| ECMWF ERA5-Land | Soil moisture, temperature, wind | Open-Meteo archive API |
| USGS SRTM 30m | Terrain slope | Google Earth Engine |
| WWF HydroSHEDS | Topographic Wetness Index | Google Earth Engine |
| GloFAS (Open-Meteo) | Barak/Surma river discharge | Open-Meteo Flood API |
| Open-Meteo forecast | 72-hour precipitation | Open-Meteo forecast API |

Flood event labels sourced from **FFWC Annual Flood Reports** and peer-reviewed literature.
Features and labels are collected from independent sources (zero circularity).

---

## Flood Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| EXTREME | >= 85% | Immediate evacuation; move crops and livestock |
| HIGH | 65–84% | Prepare; move livestock to higher ground |
| MEDIUM | 40–64% | Monitor closely; move valuables to higher ground |
| LOW | < 40% | Normal monitoring |

Upstream Barak discharge at Silchar, Assam provides approximately **36 hours lead time**
before haor inundation. Always cross-check with FFWC (ffwc.gov.bd).

---

## Citation

```bibtex
@thesis{koli2026haorfloodalert,
  author      = {Salma Hoque Talukdar Koli},
  title       = {Flood Prediction in the Haor Regions of Bangladesh Using
                 Machine Learning and Satellite-Based Data},
  school      = {RTM Al-Kabir Technical University},
  year        = {2026},
  address     = {Sylhet, Bangladesh},
  type        = {B.Sc. Thesis, Department of Computer Science and Engineering},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

© 2026 Salma Hoque Talukdar Koli · RTM Al-Kabir Technical University · Sunamganj Haor, Bangladesh
