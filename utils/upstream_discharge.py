"""Barak river discharge monitoring via GloFAS reanalysis (Silchar 24.82°N 92.79°E, ~36h lead time)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta, timezone
import requests

from config import (
    UPSTREAM_LAT, UPSTREAM_LON,
    UPSTREAM_DISCHARGE_DEFAULT,
    UPSTREAM_DISCHARGE_THRESHOLD_DANGER,
    UPSTREAM_DISCHARGE_THRESHOLD_HIGH,
    UPSTREAM_DISCHARGE_THRESHOLD_WARNING,
)

_FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
_TIMEOUT   = 15   # seconds


# Internal helpers

def _fetch_discharge_range(start_date: str, end_date: str) -> list[float]:
    """
    Pull daily GloFAS river discharge for [start_date, end_date] from
    the Open-Meteo Flood API.  Returns a list of floats (m³/s), or []
    on failure.
    """
    try:
        r = requests.get(
            _FLOOD_API,
            params={
                "latitude":         UPSTREAM_LAT,
                "longitude":        UPSTREAM_LON,
                "daily":            "river_discharge",
                "start_date":       start_date,
                "end_date":         end_date,
            },
            timeout=_TIMEOUT,
        )
        data = r.json()
        vals = data.get("daily", {}).get("river_discharge", [])
        return [float(v) for v in vals if v is not None]
    except Exception:
        return []


# Public API

def get_barak_discharge_current(days_back: int = 14) -> dict:
    """Fetch recent Barak discharge mean and trend from GloFAS reanalysis."""
    now     = datetime.now(timezone.utc)
    end_d   = now.date()
    start_d = end_d - timedelta(days=days_back)
    vals    = _fetch_discharge_range(str(start_d), str(end_d))

    if len(vals) < 3:
        return {
            "discharge":   UPSTREAM_DISCHARGE_DEFAULT,
            "trend":       0.0,
            "trend_label": "Unknown",
            "daily":       [],
            "status":      "default",
            "fetched_at":  now.strftime("%Y-%m-%d %H:%M UTC"),
        }

    dates = [(start_d + timedelta(days=i)).isoformat()
             for i in range(len(vals))]
    daily = [{"date": d, "value": round(v, 1)}
             for d, v in zip(dates, vals)]

    recent    = vals[-3:]
    discharge = round(sum(recent) / len(recent), 1)

    prev = vals[-7:-3] if len(vals) >= 7 else vals[:max(len(vals)-3, 1)]
    if prev:
        trend = round((sum(recent)/len(recent)) - (sum(prev)/len(prev)), 1)
    else:
        trend = 0.0

    if   trend >  500: trend_label = "Rising fast"
    elif trend >  150: trend_label = "Rising"
    elif trend < -150: trend_label = "Falling"
    else:              trend_label = "Stable"

    return {
        "discharge":   discharge,
        "trend":       trend,
        "trend_label": trend_label,
        "daily":       daily,
        "status":      "ok",
        "fetched_at":  now.strftime("%Y-%m-%d %H:%M UTC"),
    }


def get_barak_discharge_forecast_72h() -> dict:
    """Project 72h Barak discharge as three 24h windows extrapolated from the 10-day GloFAS trend."""
    info = get_barak_discharge_current(days_back=10)

    if info["status"] == "default":
        base  = UPSTREAM_DISCHARGE_DEFAULT
        trend = 0.0
    else:
        base  = info["discharge"]
        trend = info["trend"]

    # Clamp trend: don't let projection go negative or exceed physical maximum
    _max = 12000.0  # m³/s — extreme upper bound for Barak
    windows = []
    for i, label in enumerate(["0–24h", "24–48h", "48–72h"], start=1):
        proj = max(0.0, min(_max, base + trend * i))
        windows.append({
            "label":     label,
            "discharge": round(proj, 1),
            "rain_mm":   0.0,   # rainfall breakdown not available at this level
        })

    method = "trend_projection" if info["status"] == "ok" else "default"
    return {
        "windows": windows,
        "method":  method,
        "note":    (
            "72-hour discharge projected from 10-day GloFAS trend. "
            "Not a direct GloFAS ensemble forecast — treat as indicative only."
        ),
    }


def classify_discharge_risk(discharge: float) -> dict:
    """
    Classify Barak river discharge into flood risk categories.

    Thresholds from config — based on CWPRS flood records and
    BWDB correlation with haor inundation onset at Sunamganj.

    Returns
    -------
    {
        "level":   str,    # "Normal" | "Rising" | "High" | "Danger"
        "color":   str,    # hex color for UI
        "icon":    str,    # emoji
        "message": str,    # one-line English description
        "bangla":  str,    # Bengali description
    }
    """
    if discharge >= UPSTREAM_DISCHARGE_THRESHOLD_DANGER:
        return {
            "level":   "Danger",
            "color":   "#CC0000",
            "icon":    "🔴",
            "message": f"Extreme discharge ({discharge:,.0f} m³/s) — haor flood expected within 24–36 h.",
            "bangla":  "অত্যন্ত বিপজ্জনক — হাওরে ২৪-৩৬ ঘণ্টার মধ্যে বন্যার আশঙ্কা।",
        }
    if discharge >= UPSTREAM_DISCHARGE_THRESHOLD_HIGH:
        return {
            "level":   "High",
            "color":   "#E65C00",
            "icon":    "🟠",
            "message": f"High discharge ({discharge:,.0f} m³/s) — elevated haor flood risk. Monitor closely.",
            "bangla":  "উচ্চ প্রবাহ — হাওরে বন্যার ঝুঁকি বেশি। নজর রাখুন।",
        }
    if discharge >= UPSTREAM_DISCHARGE_THRESHOLD_WARNING:
        return {
            "level":   "Rising",
            "color":   "#B8860B",
            "icon":    "🟡",
            "message": f"Rising discharge ({discharge:,.0f} m³/s) — conditions developing. Stay alert.",
            "bangla":  "প্রবাহ বাড়ছে — পরিস্থিতি তৈরি হচ্ছে। সতর্ক থাকুন।",
        }
    return {
        "level":   "Normal",
        "color":   "#1A7A4A",
        "icon":    "🟢",
        "message": f"Normal discharge ({discharge:,.0f} m³/s) — no upstream flood signal.",
        "bangla":  "স্বাভাবিক প্রবাহ — উজানে কোনো বন্যার সংকেত নেই।",
    }
