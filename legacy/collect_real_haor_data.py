import ee
import pandas as pd
import requests
from datetime import datetime, timedelta
from config import GEE_PROJECT, HAOR_BBOX, HAOR_LAT, HAOR_LON, SAR_SCALE, ERA5_SCALE, DATA_DIR, DEFAULTS

ee.Initialize(project=GEE_PROJECT)
haor = ee.Geometry.Rectangle(HAOR_BBOX)

ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_mean,windspeed_10m_max"
    "&timezone=Asia/Dhaka"
)

LABELED_DATES = [
    ("2017-06-01", 1), ("2017-07-15", 1), ("2017-08-10", 1), ("2017-09-05", 0),
    ("2018-04-15", 1), ("2018-05-20", 1), ("2018-06-01", 1), ("2018-07-20", 0),
    ("2019-05-20", 1), ("2019-06-10", 1), ("2019-07-05", 0), ("2019-08-15", 0),
    ("2020-07-10", 0), ("2020-08-05", 0), ("2020-09-15", 0), ("2020-10-01", 0),
    ("2021-06-05", 1), ("2021-07-01", 1), ("2021-08-10", 1), ("2021-09-10", 0),
    ("2022-03-25", 1), ("2022-04-20", 1), ("2022-05-15", 1), ("2022-06-05", 0),
    ("2023-04-18", 0), ("2023-05-10", 0), ("2023-06-01", 0), ("2023-07-01", 1),
    ("2024-05-12", 1), ("2024-06-01", 1), ("2024-07-10", 1), ("2024-08-01", 0),
    ("2017-05-10", 1), ("2018-03-20", 1), ("2019-04-05", 1), ("2020-06-15", 0),
    ("2021-05-01", 1), ("2022-07-20", 1), ("2023-08-10", 0), ("2024-04-15", 1),
]


def fetch_sentinel1(start: str, end: str, wide_start: str) -> tuple:
    """Try 7-day window first, then widen to 30-day if no Sentinel-1 images found."""
    for s, e in [(start, end), (wide_start, end)]:
        try:
            count = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(haor)
                .filterDate(s, e)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .size()
                .getInfo()
            )
            if count == 0:
                continue

            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(haor)
                .filterDate(s, e)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .select(["VV", "VH"])
                .median()
            )

            vv = s1.select("VV").reduceRegion(
                ee.Reducer.mean(), haor, SAR_SCALE
            ).get("VV").getInfo()

            vh = s1.select("VH").reduceRegion(
                ee.Reducer.mean(), haor, SAR_SCALE
            ).get("VH").getInfo()

            if vv is not None and vh is not None:
                return float(vv), float(vh)

        except Exception:
            continue

    return DEFAULTS["VV"], DEFAULTS["VH"]


def fetch_soil_moisture(start: str, end: str) -> float:
    try:
        era = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(start, end)
            .select("volumetric_soil_water_layer_1")
            .mean()
        )
        val = (
            era.reduceRegion(ee.Reducer.mean(), haor, ERA5_SCALE)
            .get("volumetric_soil_water_layer_1")
            .getInfo()
        )
        if val is not None:
            return round(float(val) * 100, 1)
    except Exception:
        pass
    return DEFAULTS["soil_moisture"]


def fetch_weather(start: str, end: str) -> tuple:
    try:
        url = ARCHIVE_URL.format(lat=HAOR_LAT, lon=HAOR_LON, start=start, end=end)
        daily = requests.get(url, timeout=20).json()["daily"]
        rain = round(sum(daily["precipitation_sum"]), 1)
        temp = round(sum(daily["temperature_2m_mean"]) / len(daily["temperature_2m_mean"]), 1)
        wind = round(max(daily["windspeed_10m_max"]), 1)
        return rain, temp, wind
    except Exception:
        return DEFAULTS["rainfall"], DEFAULTS["temp"], DEFAULTS["wind"]


def fetch_slope() -> float:
    try:
        dem = ee.Image("USGS/SRTMGL1_003")
        val = (
            ee.Terrain.slope(dem)
            .reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE)
            .get("slope")
            .getInfo()
        )
        return round(float(val), 2) if val else DEFAULTS["slope"]
    except Exception:
        return DEFAULTS["slope"]


def main():
    print(f"Collecting {len(LABELED_DATES)} real GEE + Open-Meteo samples...")
    print("(This takes 20-40 minutes — do not close CMD)\n")

    slope = fetch_slope()
    print(f"  [OK] Slope (static): {slope} deg\n")

    rows = []
    skipped = 0

    for idx, (date_str, label) in enumerate(LABELED_DATES):
        print(f"  [{idx+1}/{len(LABELED_DATES)}] {date_str} (label={label}) ...", flush=True)

        date       = datetime.strptime(date_str, "%Y-%m-%d").date()
        end_str    = date_str
        start_str  = str(date - timedelta(days=7))
        wide_start = str(date - timedelta(days=30))

        vv, vh = fetch_sentinel1(start_str, end_str, wide_start)

        if vv == DEFAULTS["VV"]:
            skipped += 1
            print(f"    [WARN] No Sentinel-1 found — using default VV/VH for {date_str}")

        vvvh_ratio     = vv / vh if vh != 0 else 0.0
        soil           = fetch_soil_moisture(start_str, end_str)
        rain, temp, wind = fetch_weather(start_str, end_str)
        forecast_proxy = round(rain * 0.5, 1)

        rows.append({
            "date":                  date_str,
            "flood_label":           label,
            "VV":                    round(vv,         2),
            "VH":                    round(vh,         2),
            "vvvh_ratio":            round(vvvh_ratio, 4),
            "rainfall":              rain,
            "soil_moisture":         soil,
            "temp":                  temp,
            "wind":                  wind,
            "slope":                 slope,
            "forecast_rain_next12h": forecast_proxy,
        })

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "real_training_data_v2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n==============================")
    print(f"  Done! {len(df)} rows saved -> {out_path}")
    print(f"  Skipped (default used): {skipped}/{len(LABELED_DATES)}")
    print(f"  Flood events : {df['flood_label'].sum()}")
    print(f"  Dry events   : {(df['flood_label']==0).sum()}")
    print(f"==============================\n")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
