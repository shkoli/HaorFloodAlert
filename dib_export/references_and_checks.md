# References and Data-Source Checks

Compiled by reading source code only. No files were modified.

## 1. Label sources (77-event real_sar subset)

Unique values of `source` in `dib_export/dataset.csv`, with event counts (40 unique values, sum = 77):

| Source | Count |
|---|---|
| FFWC dry season | 22 |
| FFWC pre-monsoon dry | 8 |
| FFWC records - monsoon flood | 6 |
| FFWC Annual Report 2017 - monsoon flood | 2 |
| FFWC - 2022 monsoon flood | 2 |
| FFWC Annual Report 2020 - major flood | 2 |
| FFWC 2018 - below average flood | 2 |
| FFWC 2015 - early monsoon below threshold | 1 |
| FFWC Annual Report 2017 - peak monsoon | 1 |
| Mondal et al.2021 - peak flash flood | 1 |
| Mondal et al.2021 - 149,224 ha Boro damaged | 1 |
| Mondal et al.2021 - pre-harvest flash flood | 1 |
| FFWC 2018 - below-average but real flood | 1 |
| FFWC 2018 - low flood year | 1 |
| FFWC 2019 - pre-monsoon upstream barrage | 1 |
| FFWC 2018 - late April flash flood warning | 1 |
| FFWC 2019 - post-monsoon recovery | 1 |
| FFWC Annual Report 2020 - Google-BWDB data | 1 |
| FFWC Annual Report 2020 - extended flood | 1 |
| FFWC 2021 - pre-monsoon flash flood onset | 1 |
| FFWC 2021 - post-monsoon recession | 1 |
| Bhuiyan et al.2024 - pre-harvest flash flood | 1 |
| Bhuiyan et al.2024 - 2022 flash flood peak | 1 |
| FFWC 2018 - below average monsoon onset | 1 |
| Bhuiyan et al.2024 - recurrent flash flood | 1 |
| FFWC - 2022 monsoon higher than 2021 | 1 |
| FFWC 2022 - post-monsoon recession | 1 |
| FFWC 2022 dry season | 1 |
| ArcGIS 2024 - 2023 below average flood year | 1 |
| FFWC 2023 - localized monsoon flood event | 1 |
| ArcGIS 2024 - 2023 significantly below 2022 | 1 |
| FFWC 2023 - late monsoon flood | 1 |
| FFWC 2023 - post-monsoon | 1 |
| FFWC 2023 dry season | 1 |
| FFWC/World Bank 2024 - flash flood onset | 1 |
| World Bank GRADE 2024 - 6.25M affected | 1 |
| World Bank GRADE 2024 - major flood | 1 |
| World Bank GRADE 2024 - worst in 120 years | 1 |
| World Bank GRADE 2024 - post-peak recession | 1 |
| FFWC 2024 dry season | 1 |

### Bibliographic detail found in the codebase

- **FFWC**: No report titles, years-with-URLs, or bulletin numbers beyond the inline strings above (e.g. "FFWC Annual Report 2017", "FFWC Annual Report 2020"). The only URL found is the live water-level chart page `http://old.ffwc.gov.bd/ffwc_charts/index.php?stid=65` (`daily_validation.py:31`, station SW269/Sunamganj — used for live validation, not for these 77 labels) and a homepage link `http://www.ffwc.gov.bd` (`alerts/email_alert.py:215`). No downloadable "Annual Report" PDF/URL is referenced anywhere.
- **BWDB**: Mentioned only as a co-source in prose (e.g. `expand_training_data.py:34` "Labels from: FFWC Annual Reports, BWDB, Islam et al. 2017, Mondal et al. 2021"; `collect_more_data.py:8`). No BWDB bulletin numbers, dates, or URLs found anywhere.
- **"Mondal et al. 2021"**: Appears only as a short in-line citation (`collect_honest_data.py:8,29-31`, `collect_more_data.py`, `app.py:24`, `train_honest.py:151`). **No full title, journal name, volume, or DOI for "Mondal et al. 2021" exists anywhere in the codebase.** It is not one of the 20 numbered references in `docs/conference_paper.md`'s bibliography (refs [1]–[20], checked line-by-line). Treat "Mondal et al. 2021" as an unresolved/incomplete citation until the author supplies the actual paper.
- **"Bhuiyan et al. 2024"**: Same situation — inline citation only (`collect_honest_data.py:8,45-47`), no full reference found. Note: `docs/conference_paper.md:273,407` cites a *different* paper, "M. R. Bhuiyan, K. Uddin, and M. A. Matin (2021), 'SAR-based flood threshold detection in coastal Bangladesh...', Remote Sensing Letters, 12(9), 881–891" — this is 2021, not 2024, and about coastal Bangladesh, not haor events, so it does not appear to be the same "Bhuiyan et al. 2024" cited as a label source. This is a naming clash worth resolving.
- **"World Bank GRADE 2024"**: No report title/URL/ISBN found beyond the inline phrase; likely refers to the World Bank's Global RApid post-disaster Damage Estimation (GRADE) methodology, but no document link is present in the repo.
- **"ArcGIS 2024"** (2 sources in the table): No further detail anywhere — no map ID, layer name, or URL.
- Formal, fully-cited references that *do* exist in the repo (in `docs/conference_paper.md`'s numbered bibliography) are for the SAR/methodology background, not for these event labels: Uddin et al. 2019 (*Remote Sensing* 11(13):1581), Singha et al. 2020 (*ISPRS J. Photogramm. Remote Sens.* 166:208–220), and ECMWF ERA5-Land (doi: 10.24381/cds.e2161bac).

**Conclusion**: the flood/dry labels are backed by short internal citation strings only; no full bibliographic record (title/journal/DOI) for "Mondal et al. 2021," "Bhuiyan et al. 2024," "World Bank GRADE 2024," or any specific FFWC/BWDB report exists anywhere in this codebase, README, or docs.

## 2. Forecast data verification (forecast_rain_next_12h, forecast_rain_72h)

For the 77-row export, these two columns are generated by `collect_honest_data.py::get_forecast()` (lines 120-133):

```python
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
```

**Finding: these are NOT real forecasts.** The endpoint is `archive-api.open-meteo.com/v1/archive` — the historical reanalysis/archive API, requesting `precipitation_sum` for the window `[event_date − 3 days, event_date]` (i.e. days *ending on* the event date, all in the past relative to the event). `f12` is half the last day's total precipitation; `f72` is the sum of the 4-day window. Both are backward-looking archived values relabeled as "forecast," not a forecast issued ahead of the event.

For the 15 rows added by `expand_training_data.py` (new post-2017 events, still flagged `data_quality="real_sar"` — see caveat below), the same idea is used but derived differently, purely as an arithmetic fraction of the archived 7-day rainfall total (`expand_training_data.py:153-154`):
```python
f12  = round(wx["rain"] * 0.15, 1)
f72  = round(wx["rain"] * 0.50, 1)
```
This is not a query to any forecast endpoint at all — it is a fixed proportion (15% / 50%) of the past 7-day archived rainfall.

By contrast, the *live* production app (`streamlit_app/pages/1_Prediction.py:188`, `utils/gee_features.py:256,320`, `daily_update.py:185`, `test_forecast.py:13`) does call the genuine forecast endpoint `https://api.open-meteo.com/v1/forecast` with hourly `precipitation` parameters — but that live-forecast code path is not what produced the values in this training dataset.

## 3. Rainfall / soil_moisture / temp / wind source verification

For the 77-row export (rows originally from `collect_honest_data.py`), these four columns come from **ERA5-Land via Google Earth Engine**, not Open-Meteo (`collect_honest_data.py::get_era5()`, lines 107-118):

```python
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
```
Dataset: `ECMWF/ERA5_LAND/DAILY_AGGR` (GEE image collection). Variables requested: `total_precipitation_sum` (m → ×1000 = mm), `volumetric_soil_water_layer_1` (fraction → ×100 = %), `temperature_2m` (K → °C), `u_component_of_wind_10m` (m/s → ×3.6 = km/h). Scale: `ERA5_SCALE` = 11,132 m (`config.py`).

For the 15 rows added by `expand_training_data.py` (post-2017 new events within this 77-row subset), `rainfall`/`temp`/`wind` instead come from the **Open-Meteo archive API** directly (not ERA5/GEE) — `expand_training_data.py::fetch_weather()`, lines 109-125:
```python
ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=precipitation_sum,temperature_2m_mean,wind_speed_10m_max,et0_fao_evapotranspiration"
    "&timezone=Asia/Dhaka"
)
...
rain = round(sum(p for p in d["precipitation_sum"] if p), 1)
temp = round(sum(d["temperature_2m_mean"]) / max(len(d["temperature_2m_mean"]), 1), 1)
wind = round(max(d["wind_speed_10m_max"]), 1)
```
`soil_moisture` for these 15 rows is **not fetched from any API** — it is a synthetic formula of rainfall plus deterministic jitter (`expand_training_data.py:166`): `35.0 + rainfall * 0.12 + jitter(±3.0)`.

**Important caveat found while tracing this**: `expand_training_data.py:213` assigns `data_quality = "real_sar"` to any new event with year ≥ 2017, **without checking whether its VV/VH/ndwi/upstream_vv came from real Sentinel-1**. But `build_row()` (lines 145-176) *always* generates VV/VH/vv_vh_ratio/ndwi/upstream_vv from the hardcoded `SAR_PROFILES` synthetic profiles plus deterministic jitter (lines 96-102, 146-152) — it never calls Sentinel-1/GEE. So **15 of the 77 rows flagged `real_sar` in this export (dates: 2018-04-20, 2018-07-01, 2019-04-20, 2019-09-01, 2021-04-15, 2021-09-10, 2022-09-10, 2022-10-20, 2023-06-15, 2023-08-10, 2023-09-05, 2023-11-10, 2024-04-25, 2024-09-10, 2024-10-15) have synthetic/proxy SAR features despite the "real_sar" label**, even though their rainfall/temp/wind are real Open-Meteo archive values. Only the remaining 62 rows (from `collect_honest_data.py`) have genuinely GEE-fetched Sentinel-1 VV/VH.
