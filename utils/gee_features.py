"""
utils/gee_features.py
=====================
Central GEE feature extraction for all 13 model inputs.
Used by: 1_Prediction.py, collect_real_haor_data.py, validate_models.py

New features added:
  ndwi         — Sentinel-2 Normalized Difference Water Index
  twi          — Topographic Wetness Index (HydroSHEDS)
  upstream_vv  — Barak river upstream SAR backscatter (India gate proxy)
  forecast_72h — 72-hour cumulative precipitation forecast
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
import requests
import ee

from config import (
    DEFAULTS, GEE_PROJECT,
    HAOR_BBOX, HAOR_LAT, HAOR_LON,
    UPSTREAM_BBOX, UPSTREAM_LAT, UPSTREAM_LON, UPSTREAM_TRAVEL_HOURS,
    SAR_SCALE, S2_SCALE, CHIRPS_SCALE, ERA5_SCALE, DEM_SCALE,
    TEMP_CLIMATOLOGY,
)


def get_haor_region():
    return ee.Geometry.Rectangle(HAOR_BBOX)


def get_upstream_region():
    return ee.Geometry.Rectangle(UPSTREAM_BBOX)


# ── Sentinel-1 SAR (VV, VH) ───────────────────────────────────────────────────

def fetch_sentinel1(haor, start: str, end: str) -> tuple[float, float, float]:
    """Returns (VV, VH, vv_vh_ratio)."""
    try:
        s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
              .filterBounds(haor).filterDate(start, end)
              .filter(ee.Filter.eq("instrumentMode", "IW"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
              .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
              .select(["VV", "VH"]).median())
        vv = s1.select("VV").reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("VV").getInfo()
        vh = s1.select("VH").reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE).get("VH").getInfo()
        vv = float(vv) if vv is not None else DEFAULTS["VV"]
        vh = float(vh) if vh is not None else DEFAULTS["VH"]
        return vv, vh, float(vv / vh) if vh != 0 else DEFAULTS["vv_vh_ratio"]
    except Exception:
        return DEFAULTS["VV"], DEFAULTS["VH"], DEFAULTS["vv_vh_ratio"]


# ── Upstream Barak river SAR (India gate proxy) ────────────────────────────────

def fetch_upstream_vv(start: str, end: str) -> float:
    """
    Sentinel-1 VV backscatter upstream of Sunamganj (Barak river, Assam).
    When India opens dam/barrage gates, water level rises → VV drops.
    Low upstream_vv = high water upstream = flood risk in Haor within ~36h.
    """
    try:
        upstream = get_upstream_region()
        s1_up = (ee.ImageCollection("COPERNICUS/S1_GRD")
                 .filterBounds(upstream).filterDate(start, end)
                 .filter(ee.Filter.eq("instrumentMode", "IW"))
                 .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                 .select(["VV"]).median())
        val = s1_up.select("VV").reduceRegion(ee.Reducer.mean(), upstream, SAR_SCALE)\
                   .get("VV").getInfo()
        return float(val) if val is not None else DEFAULTS["upstream_vv"]
    except Exception:
        return DEFAULTS["upstream_vv"]


def estimate_flood_lead_time(upstream_vv: float, haor_vv: float,
                              upstream_rain_72h: float) -> dict:
    """
    Estimates flood lead time based on upstream conditions.

    Returns dict with:
      lead_hours       — estimated hours before haor flooding
      upstream_alert   — True if upstream shows flood signal
      upstream_status  — human-readable status string
    """
    # Upstream flood signal: VV < -16 dB (open water) or heavy upstream rain
    upstream_flood = upstream_vv < -16.0 or upstream_rain_72h > 150.0

    if not upstream_flood:
        return {
            "lead_hours":     None,
            "upstream_alert": False,
            "upstream_status": "Normal — no upstream flood signal",
        }

    # Estimate lead time: travel time minus how long signal has persisted
    # (We approximate using VV depression depth as proxy for event onset)
    vv_depression = max(0, -16.0 - upstream_vv)   # dB below threshold
    # Deeper depression → event already ongoing → less lead time remaining
    hours_elapsed  = min(vv_depression * 4, UPSTREAM_TRAVEL_HOURS * 0.8)
    lead_remaining = max(0, UPSTREAM_TRAVEL_HOURS - hours_elapsed)

    status = (
        f"⚠️ UPSTREAM FLOOD SIGNAL — Barak river elevated\n"
        f"Upstream VV: {upstream_vv:.1f} dB (threshold: -16 dB)\n"
        f"Estimated travel time to Haor: ~{UPSTREAM_TRAVEL_HOURS}h\n"
        f"Estimated lead time remaining: ~{lead_remaining:.0f}h"
    )

    return {
        "lead_hours":     round(lead_remaining, 0),
        "upstream_alert": True,
        "upstream_status": status,
    }


# ── Sentinel-2 NDWI ──────────────────────────────────────────────────────────

def fetch_ndwi(haor, start: str, end: str) -> float:
    """
    Sentinel-2 NDWI = (Green - NIR) / (Green + NIR) = (B3 - B8) / (B3 + B8).
    Positive NDWI → water presence.  Negative → dry land / vegetation.
    """
    try:
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(haor).filterDate(start, end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
              .select(["B3", "B8"]).median())
        ndwi_img = s2.normalizedDifference(["B3", "B8"]).rename("ndwi")
        val = ndwi_img.reduceRegion(ee.Reducer.mean(), haor, S2_SCALE)\
                      .get("ndwi").getInfo()
        return round(float(val), 4) if val is not None else DEFAULTS["ndwi"]
    except Exception:
        return DEFAULTS["ndwi"]


# ── Topographic Wetness Index (TWI) ───────────────────────────────────────────

def fetch_twi(haor) -> float:
    """
    TWI = ln(α / tan(β))
    where α = specific catchment area (from HydroSHEDS flow accumulation)
          β = slope angle (from SRTM)

    Haor bowl-shaped terrain has very high TWI (14-20) in the depression centre.
    High TWI → terrain accumulates water → high flood susceptibility.
    """
    try:
        # Flow accumulation from HydroSHEDS (15 arc-second, ~450m)
        flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select("b1")

        # Slope from SRTM in radians
        dem   = ee.Image("USGS/SRTMGL1_003")
        slope_deg = ee.Terrain.slope(dem)
        slope_rad = slope_deg.multiply(3.14159265 / 180.0)
        tan_slope  = slope_rad.tan()

        # Avoid division by zero: set minimum tan(slope) = 0.001 (0.057°)
        tan_slope_safe = tan_slope.where(tan_slope.lt(0.001), 0.001)

        # Specific catchment area: flow_acc × cell_area (15 arc-sec ≈ 450m)
        cell_area = 450.0 * 450.0   # m²
        sca = flow_acc.multiply(cell_area)

        # TWI = ln(SCA / tan(slope))
        # Resample flow_acc to DEM resolution first
        sca_resampled = sca.reproject(crs="EPSG:4326", scale=DEM_SCALE)
        twi_img = sca_resampled.divide(tan_slope_safe).log().rename("twi")

        val = twi_img.reduceRegion(ee.Reducer.mean(), haor, DEM_SCALE)\
                     .get("twi").getInfo()
        return round(float(val), 3) if val is not None else DEFAULTS["twi"]
    except Exception:
        return DEFAULTS["twi"]


# ── CHIRPS rainfall ───────────────────────────────────────────────────────────

def fetch_rainfall_chirps(haor, start: str, end: str) -> float:
    try:
        chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                  .filterDate(start, end).sum().select("precipitation"))
        val = chirps.reduceRegion(ee.Reducer.mean(), haor, CHIRPS_SCALE)\
                    .get("precipitation").getInfo()
        if val is not None and float(val) > 0:
            return round(float(val), 1)
    except Exception:
        pass
    return DEFAULTS["rainfall"]


# ── ERA5 soil moisture ────────────────────────────────────────────────────────

def fetch_soil_moisture(haor, start: str, end: str) -> float:
    try:
        era = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .filterDate(start, end)
               .select("volumetric_soil_water_layer_1").mean())
        val = era.reduceRegion(ee.Reducer.mean(), haor, ERA5_SCALE)\
                 .get("volumetric_soil_water_layer_1").getInfo()
        if val is not None:
            return round(float(val) * 100, 1)
    except Exception:
        pass
    return DEFAULTS["soil_moisture"]


# ── SRTM slope ────────────────────────────────────────────────────────────────

def fetch_slope(haor) -> float:
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        val = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), haor, DEM_SCALE)\
                        .get("slope").getInfo()
        return round(float(val), 2) if val else DEFAULTS["slope"]
    except Exception:
        return DEFAULTS["slope"]


# ── Open-Meteo weather ────────────────────────────────────────────────────────

def fetch_weather_historical(start: str, end: str,
                             lat: float = HAOR_LAT,
                             lon: float = HAOR_LON) -> tuple[float, float]:
    """Returns (temp_mean, wind_max)."""
    try:
        url = (f"https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}"
               f"&start_date={start}&end_date={end}"
               f"&daily=temperature_2m_mean,wind_speed_10m_max&timezone=Asia/Dhaka")
        d = requests.get(url, timeout=20).json()["daily"]
        temp = round(float(sum(d["temperature_2m_mean"]) / len(d["temperature_2m_mean"])), 1)
        wind = round(float(max(d["wind_speed_10m_max"])), 1)
        return temp, wind
    except Exception:
        return DEFAULTS["temp"], DEFAULTS["wind"]


def fetch_forecast(lat: float = HAOR_LAT,
                   lon: float = HAOR_LON) -> tuple[float, float]:
    """
    Returns (forecast_12h, forecast_72h) precipitation in mm.
    forecast_12h — next 12 hours
    forecast_72h — next 72 hours (3 days, key for flash flood early warning)
    """
    forecast_12h = 0.0
    forecast_72h = 0.0
    try:
        now = datetime.now(timezone.utc)
        r   = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=precipitation&forecast_days=4&timezone=Asia/Dhaka",
            timeout=15
        ).json()["hourly"]

        for t_str, p in zip(r["time"], r["precipitation"]):
            dt_aware = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
            hours_ahead = (dt_aware - now).total_seconds() / 3600
            if 0 < hours_ahead <= 12:
                forecast_12h += p
            if 0 < hours_ahead <= 72:
                forecast_72h += p

        forecast_12h = round(float(forecast_12h), 1)
        forecast_72h = round(float(forecast_72h), 1)
    except Exception:
        pass
    return forecast_12h, forecast_72h


def fetch_upstream_forecast_72h() -> float:
    """72h forecast rainfall for upstream Barak watershed."""
    _, f72 = fetch_forecast(lat=UPSTREAM_LAT, lon=UPSTREAM_LON)
    return f72


def get_forecast_rainfall_72h(lat: float = HAOR_LAT,
                               lon: float = HAOR_LON) -> dict:
    """
    Fetch hourly precipitation forecast for the next 72 hours from Open-Meteo
    and bucket into five time windows used by predict_flood_72h().

    Returns
    -------
    {
        "total":        float,        # mm total over 72 h
        "breakdown": {
            "0_6h":   float,          # mm, hours 0–6
            "6_12h":  float,          # mm, hours 6–12
            "12_24h": float,          # mm, hours 12–24
            "24_48h": float,          # mm, hours 24–48
            "48_72h": float,          # mm, hours 48–72
        },
        "hourly_times": list[str],    # ISO timestamps (Asia/Dhaka)
        "hourly_rain":  list[float],  # mm per hour
        "source":       str,          # "open-meteo" | "default"
        "fetched_at":   str,          # UTC timestamp of this fetch
    }
    """
    _empty = {
        "total": 0.0,
        "breakdown": {
            "0_6h": 0.0, "6_12h": 0.0, "12_24h": 0.0,
            "24_48h": 0.0, "48_72h": 0.0,
        },
        "hourly_times": [],
        "hourly_rain":  [],
        "source":       "default",
        "fetched_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    try:
        now = datetime.now(timezone.utc)
        r   = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=precipitation&forecast_days=4&timezone=Asia/Dhaka",
            timeout=15,
        ).json()["hourly"]

        breakdown  = {k: 0.0 for k in ("0_6h", "6_12h", "12_24h", "24_48h", "48_72h")}
        hourly_t   = []
        hourly_r   = []
        total      = 0.0

        for t_str, p in zip(r["time"], r["precipitation"]):
            p = float(p or 0.0)
            dt_aware = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
            h = (dt_aware - now).total_seconds() / 3600.0
            if not (0 < h <= 72):
                continue
            total     += p
            hourly_t.append(t_str)
            hourly_r.append(round(p, 2))
            if   h <=  6: breakdown["0_6h"]   += p
            elif h <= 12: breakdown["6_12h"]  += p
            elif h <= 24: breakdown["12_24h"] += p
            elif h <= 48: breakdown["24_48h"] += p
            else:         breakdown["48_72h"] += p

        return {
            "total":        round(total, 1),
            "breakdown":    {k: round(v, 1) for k, v in breakdown.items()},
            "hourly_times": hourly_t,
            "hourly_rain":  hourly_r,
            "source":       "open-meteo",
            "fetched_at":   now.strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception:
        return _empty


# ── Surma river discharge (Open-Meteo Flood API) ──────────────────────────────

def fetch_surma_discharge(date_str: str = None, days: int = 7) -> float:
    """
    Mean daily discharge of the Surma river at Sunamganj (m³/s) over the
    preceding `days` window, from the Open-Meteo Flood API (GloFAS reanalysis).

    Rising discharge directly precedes haor backwater inundation — this is the
    primary hydraulic driver the model was previously missing.

    Flood thresholds (approximate, Surma at Sunamganj):
      < 50  m³/s  — dry season / normal
      50–150       — elevated, early-season
      150–400      — high flood risk
      > 400        — major flood (rare)

    Parameters
    ----------
    date_str : ISO date string 'YYYY-MM-DD' (end of window). None = today.
    days     : lookback window length in days (default 7).
    """
    from datetime import date, timedelta
    try:
        if date_str:
            end_d   = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            end_d   = datetime.now(timezone.utc).date()
        start_d = end_d - timedelta(days=days)

        url = (
            "https://flood-api.open-meteo.com/v1/flood"
            f"?latitude=24.87&longitude=91.40"
            f"&daily=river_discharge"
            f"&start_date={start_d}&end_date={end_d}"
        )
        r    = requests.get(url, timeout=15)
        data = r.json()
        vals = [v for v in data["daily"]["river_discharge"] if v is not None]
        if vals:
            return round(float(sum(vals) / len(vals)), 2)
    except Exception:
        pass
    return DEFAULTS["surma_discharge"]


# ── Full feature fetch (all 14) ───────────────────────────────────────────────

def fetch_all_features(start: str, end: str,
                       include_upstream: bool = True) -> dict:
    """
    Fetches all 13 model features from GEE + APIs.
    Call ee.Initialize(project=GEE_PROJECT) before using this.

    Parameters
    ----------
    start, end   : ISO date strings e.g. '2024-04-01'
    include_upstream : set False to skip upstream fetch (faster)

    Returns
    -------
    dict with keys matching config.FEATURES
    """
    haor = get_haor_region()

    vv, vh, ratio = fetch_sentinel1(haor, start, end)
    rain          = fetch_rainfall_chirps(haor, start, end)
    soil          = fetch_soil_moisture(haor, start, end)
    temp, wind    = fetch_weather_historical(start, end)
    slope         = fetch_slope(haor)
    f12, f72      = fetch_forecast()
    ndwi          = fetch_ndwi(haor, start, end)
    twi           = fetch_twi(haor)

    if include_upstream:
        upstream_vv = fetch_upstream_vv(start, end)
    else:
        upstream_vv = DEFAULTS["upstream_vv"]

    surma_q = fetch_surma_discharge(date_str=end, days=7)

    # temp_anomaly removes the seasonal confound from raw temperature.
    # Raw temp r=0.57 with flood_label (proxies monsoon season, not flood physics).
    # Anomaly r=−0.03 — seasonal bias eliminated; legitimate signal preserved.
    end_month    = int(end[5:7])
    temp_anomaly = round(temp - TEMP_CLIMATOLOGY.get(end_month, temp), 2)

    return {
        "VV":                     round(vv,   2),
        "VH":                     round(vh,   2),
        "vv_vh_ratio":            round(ratio, 4),
        "rainfall":               rain,
        "soil_moisture":          soil,
        "temp":                   temp,          # raw — display only, not a model input
        "temp_anomaly":           temp_anomaly,  # model input (replaces raw temp)
        "wind":                   wind,
        "slope":                  slope,
        "forecast_rain_next_12h": f12,
        "ndwi":                   ndwi,
        "twi":                    twi,
        "upstream_vv":            round(upstream_vv, 2),
        "forecast_rain_72h":      f72,
        "surma_discharge":        surma_q,
    }
