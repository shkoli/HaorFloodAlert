"""
test_forecast.py
Fetches the next 12-hour precipitation forecast from Open-Meteo for a given
location. Used by daily_update.py and the Streamlit dashboard.
"""

from datetime import datetime, timezone

import requests


FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&hourly=precipitation"
    "&forecast_days=2&timezone=Asia/Dhaka"
)


def get_forecast_precip_next_12h(lat: float = 24.87, lon: float = 91.45) -> float:
    """
    Returns the total precipitation (mm) forecast for the next 12 hours.
    Falls back to 0.0 if the API is unreachable.
    """
    now = datetime.now(timezone.utc)
    url = FORECAST_URL.format(lat=lat, lon=lon)

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "hourly" not in data or "precipitation" not in data["hourly"]:
            return 0.0

        total = sum(
            precip
            for t_str, precip in zip(data["hourly"]["time"], data["hourly"]["precipitation"])
            if 0 < (datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc) - now).total_seconds() / 3600 <= 12
        )
        return round(total, 1)

    except Exception as exc:
        print(f"Forecast fetch failed: {exc}")
        return 0.0


if __name__ == "__main__":
    precip = get_forecast_precip_next_12h()
    print(f"Next 12-hour precipitation forecast: {precip} mm")
