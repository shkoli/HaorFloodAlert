"""GloFAS discharge trend analysis — 14-day fetch, OLS regression with R² gating, 72h projection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta, timezone
import numpy as np
import requests

from config import (
    UPSTREAM_LAT, UPSTREAM_LON,
    UPSTREAM_DISCHARGE_DEFAULT,
)

_FLOOD_API = "https://flood-api.open-meteo.com/v1/flood"
_TIMEOUT   = 15

# Module-level fallback cache
# Persists within the Python process lifetime.
# Protects the app from transient API failures between Streamlit cache misses.
_MODULE_CACHE: dict = {
    "values":     None,   # list[float] or None
    "fetched_at": None,   # datetime (UTC) or None
}


# Private helpers

def _smooth_3day(values: list) -> list:
    """
    3-day trailing moving average.
    For the first 1-2 days uses a shorter window (no zero-padding bias).
    """
    smoothed = []
    for i in range(len(values)):
        window = values[max(0, i - 2): i + 1]
        smoothed.append(round(sum(window) / len(window), 1))
    return smoothed


def _r_squared(y_raw: list, slope: float, intercept: float) -> float:
    """
    OLS coefficient of determination (R²) for a linear fit.
    Returns 0.0 when total variance is zero (flat series).
    """
    y     = np.array(y_raw, dtype=float)
    x     = np.arange(len(y), dtype=float)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# Public API

def get_discharge_history_14days() -> dict:
    """Fetch 14-day GloFAS discharge. Falls back to module cache then a flat default if the API fails."""
    global _MODULE_CACHE
    now     = datetime.now(timezone.utc)
    end_d   = now.date()
    start_d = end_d - timedelta(days=14)

    try:
        r = requests.get(
            _FLOOD_API,
            params={
                "latitude":   UPSTREAM_LAT,
                "longitude":  UPSTREAM_LON,
                "daily":      "river_discharge",
                "start_date": str(start_d),
                "end_date":   str(end_d),
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw  = r.json().get("daily", {}).get("river_discharge", [])
        vals = [float(v) for v in raw if v is not None]

        if len(vals) >= 3:
            _MODULE_CACHE["values"]     = vals
            _MODULE_CACHE["fetched_at"] = now
            return {
                "values":         vals,
                "status":         "ok",
                "fetched_at":     now.strftime("%Y-%m-%dT%H:%M UTC"),
                "data_age_hours": 0.0,
                "source":         "GloFAS reanalysis (Open-Meteo Flood API)",
                "n_days":         len(vals),
            }

    except Exception:
        pass

    if _MODULE_CACHE["values"] is not None:
        age_h = (now - _MODULE_CACHE["fetched_at"]).total_seconds() / 3600
        return {
            "values":         _MODULE_CACHE["values"],
            "status":         "cached",
            "fetched_at":     _MODULE_CACHE["fetched_at"].strftime("%Y-%m-%dT%H:%M UTC"),
            "data_age_hours": round(age_h, 1),
            "source":         f"Module cache ({age_h:.1f} h old — API unavailable)",
            "n_days":         len(_MODULE_CACHE["values"]),
        }

    defaults = [UPSTREAM_DISCHARGE_DEFAULT] * 14
    return {
        "values":         defaults,
        "status":         "default",
        "fetched_at":     now.strftime("%Y-%m-%dT%H:%M UTC"),
        "data_age_hours": 0.0,
        "source":         "Default baseline (API unavailable, no cache)",
        "n_days":         14,
    }


def calculate_trend_per_day(history: list) -> float:
    """OLS slope (m³/s/day). Kept for backward compat — prefer analyze_discharge_trend()."""
    if len(history) < 2:
        return 0.0
    x = np.arange(len(history), dtype=float)
    y = np.array(history, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return round(float(slope), 2)


def analyze_discharge_trend(history: list) -> dict:
    """
    3-day smoothing → OLS regression → R² reliability gate.
    Trend flagged unreliable if n < 7, R² < 0.60, or |slope| < 100 m³/s/day.
    """
    n = len(history)

    if n < 7:
        return {
            "slope":       0.0,
            "r_squared":   0.0,
            "is_reliable": False,
            "smoothed":    list(history),
            "reason":      f"Insufficient data: {n} days < 7 required for reliable trend.",
            "n_days":      n,
            "checks":      {"enough_data": False, "good_r2": False, "large_trend": False},
        }

    smoothed  = _smooth_3day(history)
    x         = np.arange(n, dtype=float)
    y         = np.array(smoothed, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    slope     = round(float(slope), 2)
    r2        = round(_r_squared(smoothed, slope, intercept), 3)

    check_data  = True
    check_r2    = r2 >= 0.60
    check_trend = abs(slope) >= 100.0

    is_reliable = check_r2 and check_trend

    if is_reliable:
        direction = "rising" if slope > 0 else "falling"
        reason = (
            f"Trend reliable: slope = {slope:+.0f} m³/s/day, "
            f"R² = {r2:.2f} ≥ 0.60, {n} days of data."
        )
    else:
        parts = []
        if not check_r2:
            parts.append(f"R² = {r2:.2f} < 0.60 (poor linear fit — discharge oscillating)")
        if not check_trend:
            parts.append(f"|slope| = {abs(slope):.0f} < 100 m³/s/day (within noise level)")
        reason = "Trend unreliable: " + "; ".join(parts) + ". Layer 3 adjustment suppressed."

    return {
        "slope":       slope,
        "r_squared":   r2,
        "is_reliable": is_reliable,
        "smoothed":    smoothed,
        "reason":      reason,
        "n_days":      n,
        "checks":      {
            "enough_data": check_data,
            "good_r2":     check_r2,
            "large_trend": check_trend,
        },
    }


def predict_discharge_next_72h(current_discharge: float, trend: float) -> list:
    """Return [+24h, +48h, +72h] discharge projections clamped to [0, 12000] m³/s."""
    _MAX = 12_000.0
    return [
        round(max(0.0, min(_MAX, current_discharge + trend * days)), 1)
        for days in [1.0, 2.0, 3.0]
    ]
