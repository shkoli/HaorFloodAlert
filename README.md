# HaorFloodAlert

A machine-learning early-warning system for pre-monsoon flash floods in the Sunamganj Haor, northeast Bangladesh. It combines Sentinel-1 SAR backscatter, rainfall forecasts, soil moisture, and a modelled upstream Barak river discharge proxy to produce a 72-hour flood probability forecast.

## Paper

Accepted at **IEEE COMPAS 2026**, University of Dhaka, 9–10 October 2026 (Paper ID 102).

**"HaorFloodAlert: A 72-Hour Machine Learning Early Warning System for Flash Floods in Bangladesh's Haor Wetlands"**

Preprint: arXiv:2605.20167 (a v2 update is pending).

## Results

Source: `analysis_v2/results/loocv_perfold_stds_summary.txt`, `analysis_v2/results/loyo_summary.txt`.

| Protocol | Accuracy | Precision | Recall | F1 | Specificity | AUC |
|---|---|---|---|---|---|---|
| LOOCV, 77 real-SAR events (primary) | 89.6% | 87.5% | 87.5% | 87.5% | 91.1% | 0.939 |
| Leave-one-year-out, 2017–2024, 8 folds, 70 events | 82.9% | — | 75.0% | — | — | 0.914 |
| Stratified 60/40 holdout, 5 seeds | 81.3% ± 6.6% | — | — | — | — | 0.918 ± 0.049 |

Primary confusion matrix (77 real-SAR events, threshold 0.50): TN=41, FP=4, FN=4, TP=28.

The leave-one-year-out result is the stricter test: each fold predicts a year the model has never trained on, rather than a single held-out event drawn from an overlapping pool of years. Compared to LOOCV, the degradation is concentrated in recall (87.5% → 75.0%), not spread evenly across metrics — the costly direction for an early-warning system, since it means missing a larger share of real floods when the model faces a genuinely unseen year.

## Data

131-event inventory: 77 real-SAR events (2014 onward, genuine Sentinel-1 extraction) plus 54 pre-Sentinel-1 proxy events (physics-calibrated backscatter estimates for dates before Sentinel-1 existed). Headline metrics are computed on the 77 real-SAR events only — evaluating on the proxy events would be circular, since those proxies are themselves constructed from post-2014 statistics.

## Model

Ensemble: Random Forest (weight 0.5) + XGBoost (weight 0.5), 10 active features, decision threshold 0.50. Training-fold augmentation: 8x Gaussian noise, sigma 0.04, applied only within each training fold, seed 42.

An LSTM component was also trained but is **excluded from all published results**. Its walk-forward evaluation showed 100% accuracy (AUC 1.000) at this data scale, which was diagnosed as a memorisation artefact rather than genuine predictive skill, not a result worth reporting.

`streamlit_app/pages/1_Prediction.py` is a separate, exploratory dashboard page that does blend the LSTM in at weight 0.20 alongside RF (0.45) and XGBoost (0.35). Its live output is a different computation from the paper's model — numbers shown there are not the paper's results and should not be cited as such.

## Reproducing the paper

```
python analysis_v2/loocv_perfold_stds.py   # camera-ready LOOCV, 89.6%
python analysis_v2/loyo_validation.py      # leave-one-year-out, 82.9%
python eval/run_paper_results.py           # arXiv v1 protocol, 90.9%
```

`eval/run_paper_results.py` computes the per-feature noise scale for augmentation once over all 77 events, before the leave-one-out fold loop begins. The camera-ready protocol (`analysis_v2/loocv_perfold_stds.py`) recomputes that same noise scale inside each fold, from the 76 training rows only. The difference flips the predicted class of one event out of 77; AUC, specificity, TN, and FP are unchanged between the two runs.

## Alert delivery

There is no automatic path from a computed flood probability to a sent alert anywhere in this repository. The scheduled forecast script (`daily_validation.py`) logs its prediction to `data/daily_validation_log.csv` (generated locally; not distributed with this repo) and does not send anything. The Alerts page (`streamlit_app/pages/3_Alerts.py`) sends real SMS via the BulkSMSBD API and real email via Gmail SMTP — both actually execute, not just a template preview — but the probability it sends comes from a manual slider set by whoever is using the page, and sending requires a button press — it is not wired to any live model output. `alerts/email_alert.py` provides a `send_flood_alert()` function as a standalone reference implementation; it has no callers anywhere in this repository — the live Alerts page uses its own inline `smtplib` code instead.

## Repository layout

- `streamlit_app/` — the Streamlit dashboard (Prediction, Map, Alerts, About, Validation, CropDamage pages); `streamlit_app/app.py` is the actual entry point (`streamlit run streamlit_app/app.py`)
- `utils/` — feature-fetching, prediction, and discharge-modelling helpers used by the live app
- `eval/` — the frozen evaluation script that produced the arXiv v1 results (`run_paper_results.py`)
- `analysis_v2/` — the camera-ready validation scripts (LOOCV per-fold diagnostic, leave-one-year-out) and their results
- `alerts/` — standalone email-alert reference module (`send_flood_alert()`); not called from anywhere in this repo — the live Alerts page has its own inline sending code
- `gee_scripts/` — standalone Google Earth Engine JavaScript scripts (SAR flood detection, NDWI mapping, rainfall analysis)
- `data/` — training and validation CSVs, and the live-run validation log (`daily_validation_log.csv`, see `data/README_validation_log.md` for its schema)
- `models/` — saved model artifacts (RF, XGBoost, active feature list, LSTM); the LSTM files are kept only because the exploratory Prediction page loads them — they are excluded from all published results
- `results/` — evaluation outputs, reports, and figures
- `docs/` — the paper draft, extracted model-info notes, and two September 2026 audit records (`PAPER_VERIFICATION.md`, `CODE_AUDIT.md`) documenting discrepancies found between the paper and the code, and the corrections made in response
- `legacy/` — historical development scripts (early training/data-collection/figure-generation code, plus the excluded LSTM's own artifacts in `legacy/lstm/`) — not used by the paper's evaluation or the live app; see `legacy/README.md`

## Data sources

- River gauge data: Bangladesh Water Development Board (BWDB) / Flood Forecasting and Warning Centre (FFWC), station SW269 (Sunamganj)
- SAR and optical imagery: ESA Copernicus Sentinel-1 (SAR backscatter) and Sentinel-2 (NDWI), accessed via Google Earth Engine
- Rainfall: CHIRPS Daily
- Soil moisture and temperature climatology: ECMWF ERA5-Land — "ERA5-Land hourly data from 1950 to present," Copernicus Climate Change Service, ECMWF, 2023, doi: 10.24381/cds.e2161bac
- Weather forecast and archive: Open-Meteo
- Discharge reanalysis: GloFAS, via the Open-Meteo Flood API
- Compute platform: Google Earth Engine

## Acknowledgements

The authors thank the Flood Forecasting and Warning Centre (FFWC), Bangladesh; the ESA Copernicus Programme; Google Earth Engine; the Open-Meteo and GloFAS teams; and the Bangladesh Rice Research Institute (BRRI) for public data access. The haor farming and fishing communities of Sunamganj are the motivation and intended beneficiaries of this research.

## Licence

MIT. See [`LICENSE`](LICENSE).

## Note on the validation log

`data/daily_validation_log.csv` is included in this repository — it is the source for the paper's Fig. 4, a prospective log of live forecasts that cannot be regenerated after the fact. It can log more than one reading per calendar day (the scheduled script can be run more than once, or re-run manually); the paper's Fig. 4 plots the last logged reading of each day, not an average or the first reading. See `data/README_validation_log.md` for a schema note (the header predates two columns added partway through logging).
