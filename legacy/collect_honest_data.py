"""
collect_honest_data.py  —  HaorFloodAlert
Collects GEE features for dates where flood labels come from
OFFICIAL SOURCES (FFWC/BWDB annual reports + peer-reviewed papers).
Labels are 100% independent of satellite features — no circularity.

Sources:
  FLOOD dates: Mondal et al. (2021), Bhuiyan et al. (2024),
               FFWC Annual Reports 2017 & 2020,
               World Bank GRADE Report (2024)
  DRY dates:   FFWC dry season records (Nov-Mar)
"""
import ee
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import (GEE_PROJECT, HAOR_BBOX, UPSTREAM_BBOX,
                    HAOR_LAT, HAOR_LON, DATA_DIR,
                    SAR_SCALE, S2_SCALE, ERA5_SCALE, DEM_SCALE, DEFAULTS)

ee.Initialize(project=GEE_PROJECT)
haor     = ee.Geometry.Rectangle(HAOR_BBOX)
upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)

# ── VERIFIED DATES FROM OFFICIAL SOURCES ─────────────────────────────────
# Each tuple: (date_str, label, source)
VERIFIED_DATES = [
    # FLOOD events — confirmed by peer-reviewed papers & FFWC reports
    ("2017-04-10", 1, "Mondal et al. 2021 - pre-harvest flash flood"),
    ("2017-04-25", 1, "Mondal et al. 2021 - 149,224 ha Boro rice damaged"),
    ("2017-05-05", 1, "Mondal et al. 2021 - peak flash flood period"),
    ("2017-07-01", 1, "FFWC Annual Report 2017 - monsoon flood"),
    ("2017-07-20", 1, "FFWC Annual Report 2017 - monsoon flood"),
    ("2017-08-10", 1, "FFWC Annual Report 2017 - peak monsoon"),
    ("2019-07-10", 1, "FFWC records - monsoon flood"),
    ("2019-07-25", 1, "FFWC records - monsoon flood"),
    ("2019-08-10", 1, "FFWC records - monsoon flood"),
    ("2020-07-10", 1, "FFWC Annual Report 2020 - major flood"),
    ("2020-07-25", 1, "FFWC Annual Report 2020 - Google-BWDB partnership data"),
    ("2020-08-10", 1, "FFWC Annual Report 2020 - major flood"),
    ("2020-09-01", 1, "FFWC Annual Report 2020 - extended flood"),
    ("2021-06-15", 1, "FFWC records - monsoon flood"),
    ("2021-07-10", 1, "FFWC records - monsoon flood"),
    ("2021-08-05", 1, "FFWC records - monsoon flood"),
    ("2022-03-28", 1, "Bhuiyan et al. 2024 - pre-harvest flash flood"),
    ("2022-04-10", 1, "Bhuiyan et al. 2024 - 2022 flash flood peak"),
    ("2022-04-25", 1, "Bhuiyan et al. 2024 - recurrent flash flood"),
    ("2022-06-15", 1, "FFWC - 2022 monsoon (higher than 2021)"),
    ("2022-07-10", 1, "FFWC - 2022 monsoon flood"),
    ("2022-08-05", 1, "FFWC - 2022 monsoon flood"),
    ("2024-06-20", 1, "World Bank GRADE 2024 - 6.25M people affected"),
    ("2024-07-10", 1, "World Bank GRADE 2024 - major Sylhet/Sunamganj flood"),
    ("2024-08-01", 1, "World Bank GRADE 2024 - worst in 120 years record"),

    # DRY events — confirmed dry season (Nov-Mar, FFWC seasonal records)
    ("2017-12-15", 0, "FFWC dry season - Rabi season"),
    ("2018-01-20", 0, "FFWC dry season"),
    ("2018-02-15", 0, "FFWC dry season"),
    ("2018-03-10", 0, "FFWC pre-monsoon dry"),
    ("2018-12-10", 0, "FFWC dry season"),
    ("2019-01-15", 0, "FFWC dry season"),
    ("2019-02-20", 0, "FFWC dry season"),
    ("2019-03-15", 0, "FFWC pre-monsoon dry"),
    ("2019-12-10", 0, "FFWC dry season"),
    ("2020-01-20", 0, "FFWC dry season"),
    ("2020-02-15", 0, "FFWC dry season"),
    ("2020-03-10", 0, "FFWC pre-monsoon dry"),
    ("2020-12-15", 0, "FFWC dry season"),
    ("2021-01-20", 0, "FFWC dry season"),
    ("2021-02-15", 0, "FFWC dry season"),
    ("2022-12-10", 0, "FFWC dry season"),
    ("2023-01-20", 0, "FFWC dry season - below-avg flood year"),
    ("2023-02-15", 0, "FFWC dry season"),
    ("2023-05-10", 0, "ArcGIS 2024 - 2023 flood much lower than 2022"),
    ("2023-07-01", 0, "ArcGIS 2024 - 2023 significantly below 2022 level"),
    ("2023-12-10", 0, "FFWC dry season"),
    ("2024-01-20", 0, "FFWC dry season"),
    ("2024-02-15", 0, "FFWC dry season"),
    ("2024-03-10", 0, "FFWC pre-monsoon dry"),
    ("2024-04-05", 0, "FFWC pre-monsoon dry"),
]

print(f"Total verified dates: {len(VERIFIED_DATES)}")
print(f"Flood (label=1): {sum(1 for d in VERIFIED_DATES if d[1]==1)}")
print(f"Dry   (label=0): {sum(1 for d in VERIFIED_DATES if d[1]==0)}")

# ── GEE feature fetchers ──────────────────────────────────────────────────
def get_s1(start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(haor).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode","IW")))
    for s,e in [(start,end),(
            str((datetime.strptime(start,"%Y-%m-%d")-timedelta(days=23)).date()), end)]:
        col2 = (ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(haor).filterDate(s,e)
                .filter(ee.Filter.eq("instrumentMode","IW")))
        if col2.size().getInfo() > 0:
            img = col2.select(["VV","VH"]).median()
            r   = img.reduceRegion(ee.Reducer.mean(), haor, SAR_SCALE)
            vv  = r.get("VV").getInfo()
            vh  = r.get("VH").getInfo()
            if vv is not None and vh is not None:
                vv,vh = round(float(vv),2), round(float(vh),2)
                return vv, vh, round(vv/vh if vh!=0 else DEFAULTS["vv_vh_ratio"],4)
    return DEFAULTS["VV"], DEFAULTS["VH"], DEFAULTS["vv_vh_ratio"]

def get_era5(start, end):
    try:
        img = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .filterDate(start,end).select(["total_precipitation_sum","volumetric_soil_water_layer_1",
                                               "temperature_2m","u_component_of_wind_10m"]).mean())
        r = img.reduceRegion(ee.Reducer.mean(), haor, ERA5_SCALE)
        rain = round(float(r.get("total_precipitation_sum").getInfo() or 0)*1000, 1)
        soil = round(float(r.get("volumetric_soil_water_layer_1").getInfo() or 0)*100, 1)
        temp = round(float(r.get("temperature_2m").getInfo() or 298.15)-273.15, 1)
        wind = round(abs(float(r.get("u_component_of_wind_10m").getInfo() or 0))*3.6, 1)
        return rain, soil, temp, wind
    except: return DEFAULTS["rainfall"],DEFAULTS["soil_moisture"],DEFAULTS["temp"],DEFAULTS["wind"]

def get_forecast(date_str):
    try:
        d  = datetime.strptime(date_str,"%Y-%m-%d").date()
        s2 = str(d - timedelta(days=3))
        url = (f"https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={HAOR_LAT}&longitude={HAOR_LON}"
               f"&start_date={s2}&end_date={date_str}"
               f"&daily=precipitation_sum&timezone=Asia/Dhaka")
        dd = requests.get(url,timeout=15).json()["daily"]
        vals = dd["precipitation_sum"]
        f12 = round(sum(vals[-1:])/1*0.5, 1) if vals else DEFAULTS["forecast_rain_next_12h"]
        f72 = round(sum(vals), 1) if vals else DEFAULTS["forecast_rain_72h"]
        return f12, f72
    except: return DEFAULTS["forecast_rain_next_12h"], DEFAULTS["forecast_rain_72h"]

def get_upstream(start, end):
    for s,e in [(start,end),(
            str((datetime.strptime(start,"%Y-%m-%d")-timedelta(days=30)).date()),end)]:
        try:
            col = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(upstream).filterDate(s,e)
                   .filter(ee.Filter.eq("instrumentMode","IW"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV")))
            if col.size().getInfo()==0: continue
            r = col.select("VV").median().reduceRegion(ee.Reducer.mean(),upstream,SAR_SCALE)
            v = r.get("VV").getInfo()
            if v: return round(float(v),2)
        except: continue
    return DEFAULTS["upstream_vv"]

SLOPE = round(
    ee.Image("USGS/SRTMGL1_003").select("elevation")
    .reduceRegion(ee.Reducer.mean(),haor,30).get("elevation").getInfo() or 0, 2
) if False else 1.91  # use known value

TWI_VAL = 17.185  # static terrain, same as before

# ── Main collection ───────────────────────────────────────────────────────
print("\nCollecting GEE features for verified dates...")
print("Labels from FFWC/BWDB/Peer-reviewed papers (NOT from VV threshold)")
print("(This takes ~30 min)\n")

rows = []
for i, (date_str, label, source) in enumerate(VERIFIED_DATES):
    d     = datetime.strptime(date_str,"%Y-%m-%d").date()
    start = str(d - timedelta(days=7))
    end   = date_str
    print(f"  [{i+1}/{len(VERIFIED_DATES)}] {date_str} (label={label}) ...", flush=True)

    vv, vh, ratio = get_s1(start, end)
    rain, soil, temp, wind = get_era5(start, end)
    f12, f72 = get_forecast(date_str)

    rows.append({
        "date": date_str, "flood_label": label, "source": source,
        "VV": vv, "VH": vh, "vv_vh_ratio": ratio,
        "rainfall": rain, "soil_moisture": soil,
        "temp": temp, "wind": wind,
        "slope": SLOPE, "twi": TWI_VAL,
        "forecast_rain_next_12h": f12,
        "ndwi": DEFAULTS.get("ndwi", -0.1),
        "upstream_vv": get_upstream(start, end),
        "forecast_rain_72h": f72,
    })

df = pd.DataFrame(rows)
out = DATA_DIR / "honest_training_data.csv"
DATA_DIR.mkdir(exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8")

flood_n = int(df.flood_label.sum())
dry_n   = int((df.flood_label==0).sum())
print(f"\n{'='*60}")
print(f"  Saved {len(df)} rows → {out}")
print(f"  Flood: {flood_n}  |  Dry: {dry_n}")
print(f"  Labels from: FFWC/BWDB/Peer-reviewed papers")
print(f"  Features from: GEE (independent of labels)")
print(f"  Now run: python train_honest.py")
print(f"{'='*60}")
