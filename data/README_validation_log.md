# `daily_validation_log.csv`

A prospective log of this system's live forecasts, appended to by `daily_validation.py` each time it runs (scheduled and manual runs both write to it). Each row is one forecast issued at that timestamp — not a backtest, not a simulation. It is the source for the paper's Fig. 4.

## Schema drift

The file's header row (`date,time,flood_probability,risk_level,sw269_water_level,forecast_rain_72h,soil_moisture,barak_discharge`, 8 fields) predates two columns — `danger_level` and `wl_status` — that were added to `CSV_COLUMNS` in `daily_validation.py` partway through logging. The header was never rewritten when that change shipped. As a result:

- The earliest row (2026-05-26 23:38) has 8 fields, matching the header.
- Every row from 2026-05-26 23:55 onward has 10 fields — `danger_level` and `wl_status` appended — even though the header still only names 8 columns.

This is left as-is rather than corrected, since the file is evidence of what the running system actually produced, not a curated dataset. A reader parsing it should use `CSV_COLUMNS` in `daily_validation.py` (10 fields) for any row after the first, not the file's own header.

## Multiple readings per day

The script can be triggered more than once in a calendar day (scheduled run plus manual re-runs), so a date can appear on several rows with different timestamps. The paper's Fig. 4 plots the **last** logged reading of each day, not an average or the first reading.

## The log extends past the paper's window

A scheduled job is still running and still appending to this file, so it now covers more than the paper's study period. Fig. 4 uses only the pre-monsoon prospective run, **26 May – 5 June 2026**. Rows after 5 June 2026 are post-monsoon-onset: a different hydrological season, generated after the paper's numbers were finalised, and never peer-reviewed. They were deliberately not added to the paper's figure — their presence here is not an update to Fig. 4, and they shouldn't be read as one.

This file will keep growing in this repository for as long as the scheduled job keeps running. If that's not wanted, stop the scheduled task that invokes `daily_validation.py`.
