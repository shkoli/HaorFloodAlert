"""
config.py  —  HaorFloodAlert central configuration
All paths, constants, and feature definitions live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.resolve()
load_dotenv(ROOT / ".env")
MODELS_DIR  = ROOT / "models"
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"

for _d in (MODELS_DIR, DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Google Earth Engine ───────────────────────────────────────────────────────
GEE_PROJECT = "flood-haor-project"

# ── Study area: Sunamganj Haor ────────────────────────────────────────────────
HAOR_BBOX = [91.35, 24.75, 91.55, 25.00]   # [west, south, east, north]
HAOR_LAT  = 24.87
HAOR_LON  = 91.45

# ── Upstream proxy: Barak river near Silchar, Assam (India) ──────────────────
# Water released from upstream barrages reaches Sunamganj in ~24-48 hours.
UPSTREAM_BBOX = [92.70, 24.60, 93.20, 25.00]
UPSTREAM_LAT  = 24.80
UPSTREAM_LON  = 92.95

# Estimated river travel time (Barak → Surma → Haor), hours
UPSTREAM_TRAVEL_HOURS = 36   # ~120 km at 0.9 m/s average flow

# ── GEE pixel scales ─────────────────────────────────────────────────────────
SAR_SCALE    = 30
S2_SCALE     = 20
CHIRPS_SCALE = 5560
ERA5_SCALE   = 11132
DEM_SCALE    = 30

# ── Feature set ───────────────────────────────────────────────────────────────
#
#  Total features collected:  15
#  ─────────────────────────────────────────────────────────────────
#  ML model inputs (FEATURES): 13  — listed below
#  Dashboard only (excluded):   2  — multicollinearity with model features:
#    surma_discharge  r=0.792 with soil_moisture  (tested, excluded; see add_discharge.py)
#    barak_discharge  likely correlated with upstream_vv  (see add_barak_discharge.py)
#
#  Feature sources (all real, live or reanalysis):
#  ─────────────────────────────────────────────────────────────────
#  #  Feature                   Source          Notes
#  1  VV                        Sentinel-1 GRD  SAR backscatter, 30m, 6-day
#  2  VH                        Sentinel-1 GRD  cross-polarisation
#  3  vv_vh_ratio               Sentinel-1 GRD  flood discriminant
#  4  rainfall                  CHIRPS daily    7-day cumulative
#  5  soil_moisture             ERA5-Land       11km, volumetric
#  6  temp                      ERA5-Land       2m mean temperature
#  7  wind                      ERA5-Land       10m max wind speed
#  8  slope                     USGS SRTM 30m   terrain slope (degrees)
#  9  forecast_rain_next_12h    Open-Meteo      hourly precipitation forecast
#  10 ndwi                      Sentinel-2 SR   (B3-B8)/(B3+B8), 20m
#  11 twi                       HydroSHEDS+SRTM ln(SCA/tan(slope))
#  12 upstream_vv               Sentinel-1 GRD  Barak at Silchar (92.79°E)
#  13 forecast_rain_72h         Open-Meteo      3-day cumulative forecast
# [14] surma_discharge           GloFAS/Open-Meteo  DASHBOARD ONLY — multicollinear
# [15] barak_discharge_cumecs    GloFAS/Open-Meteo  DASHBOARD ONLY — see note above

# Monthly temperature climatology for Sunamganj Haor (24.87°N, 91.45°E).
# Empirical means from ERA5-Land via Open-Meteo archive, 2009–2024.
# Used to compute temp_anomaly = observed_temp - monthly_mean, eliminating
# the seasonal confound (raw temp r=0.57 with flood label; anomaly r=−0.03).
TEMP_CLIMATOLOGY = {
    1: 17.03,   # January
    2: 19.35,   # February
    3: 22.04,   # March
    4: 24.83,   # April
    5: 25.35,   # May
    6: 27.14,   # June
    7: 27.26,   # July
    8: 27.48,   # August
    9: 28.26,   # September
   10: 26.95,   # October
   11: 25.40,   # November
   12: 20.09,   # December
}

FEATURES = [
    # SAR backscatter (Sentinel-1) — Features 1–3
    "VV",
    "VH",
    "vv_vh_ratio",
    # Meteorological (CHIRPS + ERA5 + Open-Meteo) — Features 4–7
    "rainfall",
    "soil_moisture",
    # temp replaced by temp_anomaly to remove seasonal confound:
    #   raw temp r=0.57 with flood_label (spurious — proxies monsoon season)
    #   temp_anomaly r=−0.03 (confound eliminated; legitimate signal preserved)
    "temp_anomaly",
    "wind",
    # Terrain (USGS SRTM 30m) — Feature 8
    "slope",
    # Short-term rainfall forecast (Open-Meteo hourly) — Feature 9
    "forecast_rain_next_12h",
    # Optical water index (Sentinel-2 SR 20m) — Feature 10
    "ndwi",
    # Topographic Wetness Index (HydroSHEDS + SRTM) — Feature 11
    "twi",
    # Upstream Barak river SAR backscatter — Feature 12 (India barrage release proxy)
    "upstream_vv",
    # 72-hour cumulative precipitation forecast (Open-Meteo) — Feature 13
    "forecast_rain_72h",
]

# ── Ensemble weights ──────────────────────────────────────────────────────────
RF_WEIGHT  = 0.45
XGB_WEIGHT = 0.35
LSTM_WEIGHT= 0.20

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS = {"EXTREME": 0.85, "HIGH": 0.65, "MEDIUM": 0.40}

# ── Default fallback values ───────────────────────────────────────────────────
DEFAULTS = {
    "VV":                  -15.0,
    "VH":                  -20.0,
    "vv_vh_ratio":           0.75,
    "rainfall":            120.0,
    "soil_moisture":        30.0,
    "temp":                 29.0,   # raw temperature (display only — not a model input)
    "temp_anomaly":          0.0,   # default = no anomaly (unknown deviation from seasonal mean)
    "wind":                 12.0,
    "slope":                 1.9,
    "forecast_rain_next_12h": 0.0,
    "ndwi":                 -0.1,
    "twi":                   8.0,
    "upstream_vv":         -13.0,
    "forecast_rain_72h":     0.0,
    # Surma at Sunamganj: pre-monsoon dry-season baseline ~20 m³/s
    "surma_discharge":      20.0,
}

# ── NDWI physical bounds for Sunamganj Haor ───────────────────────────────────
# Flood: positive NDWI (open water ~0.1 to 0.6)
# Dry:   negative NDWI (vegetation/bare soil ~-0.4 to 0.0)
NDWI_FLOOD_MEAN  =  0.25
NDWI_DRY_MEAN    = -0.18

# ── TWI calibration for Haor bowl-shaped terrain ─────────────────────────────
# Haor is a closed depression — TWI is very high (12-20) in the bowl centre
# Upland areas have lower TWI (6-10)
TWI_FLOOD_MEAN   = 15.0
TWI_DRY_MEAN     =  8.5

# ── Upstream Barak discharge thresholds (m³/s = cumecs) ──────────────────────
# Barak river at Silchar, Assam. Source: CWPRS flood records + BWDB correlation.
# Haor inundation begins ~36h after discharge exceeds DANGER threshold.
UPSTREAM_DISCHARGE_THRESHOLD_DANGER  = 7500   # red  — flood expected in 24-36h
UPSTREAM_DISCHARGE_THRESHOLD_HIGH    = 6000   # orange — high flood risk
UPSTREAM_DISCHARGE_THRESHOLD_WARNING = 4000   # yellow — rising, monitor closely
# Default fallback (dry-season mean at Silchar, pre-monsoon baseline)
UPSTREAM_DISCHARGE_DEFAULT = 450.0            # m³/s

# ── Email alert configuration ─────────────────────────────────────────────────
# Set these in a local .env file (see .env.example) — never hardcode credentials.
# Generate an App Password at: myaccount.google.com/apppasswords
# (Requires 2-Step Verification to be enabled on your Google account.)
ALERT_EMAIL_SENDER    = os.getenv("ALERT_EMAIL_SENDER", "")
ALERT_EMAIL_RECIPIENT = os.getenv("ALERT_EMAIL_RECIPIENT", "")
ALERT_APP_PASSWORD    = os.getenv("ALERT_APP_PASSWORD", "")   # 16-char Gmail App Password

# BulkSMSBD SMS gateway (optional, used by streamlit_app/pages/3_Alerts.py UI)
BULKSMS_API_KEY  = os.getenv("BULKSMS_API_KEY", "")
BULKSMS_SENDER_ID = os.getenv("BULKSMS_SENDER_ID", "")
