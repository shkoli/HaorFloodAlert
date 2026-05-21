"""
config.py  —  HaorFloodAlert central configuration.

All paths, constants, and feature definitions live here.
API credentials are read from environment variables; copy .env.example
to .env and fill in your values before running.
"""

import os
from pathlib import Path

ROOT        = Path(__file__).parent.resolve()
MODELS_DIR  = ROOT / "models"
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"

for _d in (MODELS_DIR, DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Google Earth Engine ───────────────────────────────────────────────────────
GEE_PROJECT = os.environ.get("GEE_PROJECT", "your-gee-project-id")

# ── Study area: Sunamganj Haor ────────────────────────────────────────────────
HAOR_BBOX = [91.35, 24.75, 91.55, 25.00]   # [west, south, east, north]
HAOR_LAT  = 24.87
HAOR_LON  = 91.45

# ── Upstream proxy: Barak river near Silchar, Assam ──────────────────────────
# Water released upstream reaches Sunamganj in approximately 24-48 hours.
UPSTREAM_BBOX = [92.70, 24.60, 93.20, 25.00]
UPSTREAM_LAT  = 24.80
UPSTREAM_LON  = 92.95

# Estimated river travel time (Barak -> Surma -> Haor), hours
UPSTREAM_TRAVEL_HOURS = 36

# ── GEE pixel scales ─────────────────────────────────────────────────────────
SAR_SCALE    = 30
S2_SCALE     = 20
CHIRPS_SCALE = 5560
ERA5_SCALE   = 11132
DEM_SCALE    = 30

# ── Feature set ───────────────────────────────────────────────────────────────
# Total features collected: 15
# ML model inputs (FEATURES): 13
# Dashboard-only (excluded from model due to multicollinearity): 2
#   surma_discharge  r=0.792 with soil_moisture
#   barak_discharge  correlated with upstream_vv
#
# Feature sources:
#   VV, VH, vv_vh_ratio    — Sentinel-1 GRD (SAR), 30 m, 6-day repeat
#   rainfall               — CHIRPS daily, 7-day cumulative
#   soil_moisture          — ERA5-Land, 11 km, volumetric
#   temp_anomaly           — ERA5-Land 2m temp minus monthly climatology
#   wind                   — ERA5-Land, 10 m max wind speed
#   slope                  — USGS SRTM 30 m terrain slope (degrees)
#   forecast_rain_next_12h — Open-Meteo hourly precipitation forecast
#   ndwi                   — Sentinel-2 SR, (B3-B8)/(B3+B8), 20 m
#   twi                    — HydroSHEDS + SRTM, ln(SCA / tan(slope))
#   upstream_vv            — Sentinel-1 GRD at Barak/Silchar (92.79 E)
#   forecast_rain_72h      — Open-Meteo 3-day cumulative forecast

# Monthly temperature climatology for Sunamganj Haor (24.87 N, 91.45 E).
# Empirical means from ERA5-Land via Open-Meteo archive, 2009-2024.
# Used to compute temp_anomaly = observed_temp - monthly_mean, eliminating
# the seasonal confound (raw temp Pearson r=0.57 with flood label;
# anomaly r=-0.03 after deconfounding).
TEMP_CLIMATOLOGY = {
    1:  17.03,
    2:  19.35,
    3:  22.04,
    4:  24.83,
    5:  25.35,
    6:  27.14,
    7:  27.26,
    8:  27.48,
    9:  28.26,
    10: 26.95,
    11: 25.40,
    12: 20.09,
}

FEATURES = [
    "VV",
    "VH",
    "vv_vh_ratio",
    "rainfall",
    "soil_moisture",
    # raw temp replaced by temp_anomaly to remove seasonal confound:
    # raw temp r=0.57 with flood_label (proxies monsoon season);
    # temp_anomaly r=-0.03 (confound eliminated).
    "temp_anomaly",
    "wind",
    "slope",
    "forecast_rain_next_12h",
    "ndwi",
    "twi",
    "upstream_vv",
    "forecast_rain_72h",
]

# ── Ensemble weights ──────────────────────────────────────────────────────────
RF_WEIGHT   = 0.45
XGB_WEIGHT  = 0.35
LSTM_WEIGHT = 0.20

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS = {"EXTREME": 0.85, "HIGH": 0.65, "MEDIUM": 0.40}

# ── Default fallback values ───────────────────────────────────────────────────
DEFAULTS = {
    "VV":                     -15.0,
    "VH":                     -20.0,
    "vv_vh_ratio":              0.75,
    "rainfall":               120.0,
    "soil_moisture":           30.0,
    "temp":                    29.0,
    "temp_anomaly":             0.0,
    "wind":                    12.0,
    "slope":                    1.9,
    "forecast_rain_next_12h":   0.0,
    "ndwi":                    -0.1,
    "twi":                      8.0,
    "upstream_vv":            -13.0,
    "forecast_rain_72h":        0.0,
    "surma_discharge":         20.0,
}

# ── NDWI bounds for Sunamganj Haor ───────────────────────────────────────────
NDWI_FLOOD_MEAN =  0.25
NDWI_DRY_MEAN   = -0.18

# ── TWI calibration ───────────────────────────────────────────────────────────
TWI_FLOOD_MEAN  = 15.0
TWI_DRY_MEAN    =  8.5

# ── Upstream Barak discharge thresholds (m3/s) ───────────────────────────────
# Source: CWPRS flood records + BWDB correlation.
# Haor inundation begins approximately 36h after discharge exceeds DANGER level.
UPSTREAM_DISCHARGE_THRESHOLD_DANGER  = 7500
UPSTREAM_DISCHARGE_THRESHOLD_HIGH    = 6000
UPSTREAM_DISCHARGE_THRESHOLD_WARNING = 4000
UPSTREAM_DISCHARGE_DEFAULT           = 450.0

# ── Email alert configuration (read from environment) ────────────────────────
# Set these in your .env file or shell environment.
ALERT_EMAIL_SENDER    = os.environ.get("ALERT_EMAIL_SENDER",    "")
ALERT_EMAIL_RECIPIENT = os.environ.get("ALERT_EMAIL_RECIPIENT", "")
ALERT_APP_PASSWORD    = os.environ.get("ALERT_APP_PASSWORD",    "")
