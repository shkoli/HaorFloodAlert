# Refetch Report — 15 Repaired Rows

Real Sentinel-1 (VV/VH/upstream_vv), Sentinel-2 (NDWI), and ERA5-Land (soil_moisture) values fetched via Google Earth Engine to replace the synthetic `SAR_PROFILES` proxy values that `expand_training_data.py` had written into these 15 rows despite flagging them `data_quality="real_sar"`.

| Date | VV old→new | VH old→new | NDWI old→new | upstream_vv old→new | soil_moisture old→new | S1 window | S1 images found | Within ±6d of event? |
|---|---|---|---|---|---|---|---|---|
| 2018-04-20 | -14.21 → -10.02 | -20.25 → -15.8 | -0.4052 → -0.1 (fallback default, no cloud-free S2 scene) | -19.67 → -8.82 | 42.5 → 40.6 | 2018-04-13..2018-04-20 | 3 | yes |
| 2018-07-01 | -17.69 → -16.85 | -24.38 → -26.29 | 0.1192 → -0.1 (fallback default, no cloud-free S2 scene) | -15.81 → -9.27 | 51.5 → 41.9 | 2018-06-24..2018-07-01 | 3 | yes |
| 2019-04-20 | -15.1 → -10.2 | -21.41 → -16.42 | -0.3145 → -0.4181 | -20.86 → -7.88 | 36.5 → 37.6 | 2019-04-13..2019-04-20 | 2 | yes |
| 2019-09-01 | -11.89 → -15.35 | -18.34 → -23.36 | -0.3708 → -0.1 (fallback default, no cloud-free S2 scene) | -11.31 → -9.04 | 37.6 → 41.2 | 2019-08-25..2019-09-01 | 2 | yes |
| 2021-04-15 | -15.47 → -10.91 | -20.85 → -16.47 | -0.3093 → -0.1286 | -19.8 → -9.25 | 39.0 → 32.9 | 2021-04-08..2021-04-15 | 2 | yes |
| 2021-09-10 | -10.75 → -16.26 | -17.46 → -23.29 | -0.4073 → 0.0664 | -9.35 → -8.02 | 39.9 → 41.2 | 2021-09-03..2021-09-10 | 3 | yes |
| 2022-09-10 | -10.86 → -16.45 | -17.84 → -23.23 | -0.477 → 0.2016 | -11.77 → -8.97 | 55.5 → 41.0 | 2022-09-03..2022-09-10 | 2 | yes |
| 2022-10-20 | -11.54 → -17.86 | -17.21 → -24.19 | -0.4281 → 0.1895 | -9.59 → -8.63 | 38.3 → 38.0 | 2022-10-13..2022-10-20 | 3 | yes |
| 2023-06-15 | -17.68 → -9.01 | -23.09 → -16.33 | 0.1482 → -0.2135 | -15.81 → -9.04 | 69.7 → 40.1 | 2023-06-08..2023-06-15 | 1 | yes |
| 2023-08-10 | -18.65 → -16.45 | -24.47 → -22.77 | 0.106 → -0.1 (fallback default, no cloud-free S2 scene) | -17.7 → -7.94 | 63.6 → 42.5 | 2023-08-03..2023-08-10 | 2 | yes |
| 2023-09-05 | -11.75 → -17.29 | -18.56 → -26.29 | -0.4494 → 0.1599 | -9.23 → -8.98 | 43.8 → 39.3 | 2023-08-29..2023-09-05 | 2 | yes |
| 2023-11-10 | -11.87 → -15.87 | -17.59 → -24.0 | -0.4172 → -0.1 (fallback default, no cloud-free S2 scene) | -11.35 → -8.71 | 37.3 → 21.7 | 2023-11-03..2023-11-10 | 3 | yes |
| 2024-04-25 | -15.35 → -10.09 | -20.93 → -16.12 | -0.4064 → -0.3461 | -19.87 → -9.11 | 34.3 → 36.9 | 2024-04-18..2024-04-25 | 3 | yes |
| 2024-09-10 | -11.23 → -17.69 | -17.93 → -26.69 | -0.4738 → 0.2103 | -10.8 → -9.17 | 42.2 → 40.6 | 2024-09-03..2024-09-10 | 2 | yes |
| 2024-10-15 | -11.28 → -16.43 | -18.74 → -26.23 | -0.4235 → 0.1666 | -9.21 → -7.83 | 40.7 → 40.5 | 2024-10-08..2024-10-15 | 2 | yes |

## Sentinel-1 image-date notes
All 15 dates had at least one Sentinel-1 image within ±6 days of the event date.

## Sanity check on dataset_v2.csv

- No missing values
- All 77 dates unique
- VV/VH/upstream_vv within plausible Sentinel-1 backscatter ranges
- rain_24h / rain_96h non-negative
