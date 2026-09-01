# Data Dictionary — dataset.csv

Source: `data/honest_training_data_v2.csv`, filtered to `data_quality == 'real_sar'` (the 77-event SAR-verified subset). All descriptions inferred from `config.py`, `collect_honest_data.py`, `expand_training_data.py`, and `gee_scripts/`.

| Column | Dtype | Unit | Min | Max | % Missing | Description |
|---|---|---|---|---|---|---|
| date | object (str, YYYY-MM-DD) | date | 2014-12-10 | 2024-10-15 | 0.0 | Event date; hardcoded per-event in `collect_honest_data.py`/`expand_training_data.py` event lists. |
| flood_label | int64 | binary (0/1) | 0 | 1 | 0.0 | Flood (1) vs. non-flood (0) label, sourced independently from FFWC/BWDB reports and peer-reviewed literature (not derived from the satellite features in this row). |
| source | object (str) | — | — | — | 0.0 | Citation string for the label (e.g. FFWC report, published paper), 40 distinct sources across 77 rows. |
| VV | float64 | dB | -22.35 | -8.76 | 0.0 | Sentinel-1 GRD VV-polarization backscatter over the haor AOI, mean value (`collect_honest_data.py::get_s1()`, 30 m scale). |
| VH | float64 | dB | -29.24 | -15.23 | 0.0 | Sentinel-1 GRD VH (cross-pol) backscatter over the haor AOI. |
| vv_vh_ratio | float64 | ratio (unitless) | 0.5237 | 0.8021 | 0.0 | VV/VH ratio; flood-discrimination feature (lower ratio indicates greater surface water). |
| rainfall | float64 | mm | 0.0 | 266.8 | 0.0 | 7-day cumulative precipitation preceding the event, from ERA5-Land / Open-Meteo archive. |
| soil_moisture | float64 | % volumetric | 14.2 | 69.7 | 0.0 | ERA5-Land layer-1 volumetric soil moisture. |
| temp | float64 | °C | 16.5 | 30.1 | 0.0 | ERA5-Land 2 m mean air temperature. |
| wind | float64 | km/h | 0.0 | 21.7 | 0.0 | 10 m wind speed. |
| slope | float64 | degrees | 1.91 | 1.91 | 0.0 | Static terrain slope for the haor AOI from USGS SRTM 30 m DEM (constant across all events; hardcoded in `config.py`). |
| twi | float64 | unitless (TWI index) | 17.185 | 17.185 | 0.0 | Static Topographic Wetness Index for the haor AOI (constant; hardcoded in `config.py`). |
| forecast_rain_next_12h | float64 | mm | 0.0 | 40.4 | 0.0 | Open-Meteo forecast precipitation for the next 12 hours from the event date. |
| ndwi | float64 | unitless (-1 to 1) | -0.6428 | 0.2701 | 0.0 | Normalized Difference Water Index, (Green−NIR)/(Green+NIR), from Sentinel-2 (20 m), used as a satellite water-extent proxy. |
| upstream_vv | float64 | dB | -20.86 | -7.12 | 0.0 | Sentinel-1 VV backscatter at the upstream Barak River monitoring point (Silchar, ~24.80°N/92.95°E), used as an early-warning upstream signal. |
| forecast_rain_72h | float64 | mm | 0.0 | 277.4 | 0.0 | Open-Meteo 3-day cumulative rainfall forecast. |
| data_quality | object (str) | — | — | — | 0.0 | Provenance flag; constant `"real_sar"` in this filtered file, meaning the row's SAR/optical features were fetched from real satellite imagery (event year ≥ 2014) rather than synthesized from event-type profiles. |
| temp_anomaly | float64 | °C | -1.93 | 3.06 | 0.0 | Temperature minus its monthly climatological mean (`config.py::TEMP_CLIMATOLOGY`), used to remove seasonal confounding from raw temperature. |

**Rows:** 77. **Columns:** 18. No missing values in any column.
