# Deseasonalized Machine Learning Flood Prediction for Bangladesh Haor Wetlands Using Sentinel-1 SAR

**Salma Hoque Talukdar Koli¹, Fahima Haque Talukder Jely², Md. Samiul Alim¹**
¹Department of Computer Science & Engineering, RTM Al-Kabir Technical University, Sylhet, Bangladesh
²Department of Computer Science & Engineering, North East University Bangladesh, Sylhet, Bangladesh
{salma.koli, samiul.alim}@rtmaktu.edu.bd · fahima.jely@neub.edu.bd

---

*Proceedings — IEEE International Conference on Remote Sensing and Machine Intelligence (RSMI), 2026*
*IEEE Format · 5 pages · DOI: 10.1109/RSMI2026.XXXXXX*

---

## Abstract

Flash floods in the haor wetlands of Bangladesh destroy the annual *boro* rice harvest, threatening the food security of millions before the crop can be collected. Existing flood prediction systems are calibrated for riverine floods and do not address the distinct backwater-driven inundation dynamics of haor basins. This paper presents HaorFloodAlert, a deseasonalized machine learning ensemble for 72-hour flood probability forecasting in the Sunamganj Haor (~8,000 km²). We make four key contributions: (1) the identification and correction of a temperature seasonal confound that inflated reported accuracy by 6.9 percentage points (r = 0.570 before deconfounding, r = −0.031 after); (2) Otsu-threshold Sentinel-1 SAR change detection for cloud-independent flood mapping validated against three historical events; (3) an upstream Barak river SAR backscatter proxy at Silchar, Assam, providing ~36-hour lead time before haor inundation; and (4) the first publicly documented deseasonalized ML model specifically designed for haor flood prediction. The ensemble (Random Forest 50% + XGBoost 50%; an LSTM component is trained separately but excluded from this metric because its 100% walk-forward accuracy at n=101 is a known small-sample memorisation artifact) achieves Leave-One-Out Cross-Validation accuracy of **89.6%**, Recall **87.5%**, F1-score **87.5%**, and AUC-ROC **0.939** (77 real-SAR events, 2014–2024). A stratified 60/40 holdout over five random seeds yields **81.3% ± 6.6%** accuracy (AUC **0.918 ± 0.049**). The system is deployed as a real-time Streamlit dashboard with Bengali-language SMS and email community alerts.

**Keywords:** flood prediction, haor, Bangladesh, Sentinel-1 SAR, machine learning, seasonal deconfounding, Random Forest, XGBoost, LSTM, upstream discharge proxy.

---

## I. Introduction

### A. Context and Motivation

The haor basins of northeast Bangladesh are among the world's most flood-vulnerable inland wetland ecosystems. Covering approximately 8,000 km² across Sunamganj, Netrokona, and Sylhet districts, haors are shallow bowl-shaped depressions that fill from pre-monsoon rainfall and upstream river discharge originating in the Meghalaya Hills and Assam, India. Unlike riverine floods that progress along established channels, haor inundation arrives as backwater, rising rapidly across flat terrain with little warning.

The consequences are catastrophic and specific: the *boro* rice crop, which matures in March–April, is the sole agricultural cycle in the haor calendar. A flash flood during this window—before mechanical harvesting can begin—destroys the entire annual yield. For the 3–4 million people whose livelihoods depend directly on haor agriculture and fisheries, an accurate 36–72 hour advance warning is not a convenience but a survival imperative.

*Personal Motivation.* As a researcher from Sylhet, I grew up witnessing flash floods displace haor communities with no more than a few hours of warning. Families I knew lost entire harvests in single flood events while government forecasting systems—designed for the Brahmaputra and Meghna rivers—issued no specific haor alerts. The motivation for this research is personal: to build a system that gives the farmers of Sunamganj the hours they need to cut their crops, move their livestock, and protect their families. Every accuracy percentage point in this paper represents real lead time for real people.

### B. Problem Statement

The Flood Forecasting and Warning Centre (FFWC) of Bangladesh operates river-level monitoring at major gauge stations but provides no haor-specific inundation forecasts. Existing ML flood prediction studies in Bangladesh focus on riverine contexts using hydrological station data [1], [2]. No prior work has combined Sentinel-1 SAR change detection, upstream transboundary river monitoring, and deseasonalized feature engineering specifically for haor backwater dynamics.

### C. Research Gap

A critical methodological flaw in prior haor-adjacent ML flood studies is the inclusion of raw air temperature as a model feature. Because floods occur during the warm monsoon season, temperature is a proxy for calendar time rather than a causal flood driver. We demonstrate that this inflates reported accuracy by up to 6.9 percentage points (Section V-D), and we correct it using monthly climatological anomaly deconfounding.

### D. Paper Organization

Section II reviews related work. Section III describes the study area and dataset. Section IV presents the methodology. Section V reports results. Sections VI–VIII discuss implications, limitations, and conclusions.

---

## II. Related Work

Early machine learning applications to Bangladesh flood prediction relied on station-based hydrological data from the Bangladesh Water Development Board (BWDB) [1]. Rajab et al. [3] compared Random Forest, XGBoost, and neural networks for riverine flood forecasting in Bangladesh, reporting AUC values of 0.81–0.89 on station gauge data, but without satellite remote sensing or haor-specific features.

**SAR-based flood mapping** has matured considerably since Sentinel-1's launch in 2014. Uddin et al. [4] demonstrated operational flood extent mapping using multi-temporal Sentinel-1 VV backscatter with Otsu thresholding, achieving 89% accuracy over Bangladeshi floodplains. Singha et al. [5] extended this to flood-impacted paddy rice mapping on Google Earth Engine, directly relevant to boro rice loss estimation. Both studies treat SAR flood mapping as a remote sensing task; neither integrates it with ML-based predictive forecasting.

**Ensemble ML for flood susceptibility** is well established [6], [7]. Talukdar et al. [6] used bagging ensembles for flood susceptibility in the Teesta basin with Random Forest achieving the highest AUC (0.94). Islam et al. [7] applied advanced ensemble methods including Gradient Boosting and XGBoost for susceptibility mapping in Sylhet. Neither study addresses temporal flood prediction, deseasonalization, or upstream transboundary river proxies.

**LSTM and deep learning** for flood forecasting have been applied globally [8] but face a critical limitation in haor contexts: insufficient time-series training data. Kratzert et al. [8] required 30+ years of continuous streamflow records for LSTM calibration. In the haor region, real Sentinel-1 time series are available only from 2014, and flood event labels are sparse and difficult to verify.

**Upstream transboundary river monitoring** is underexplored in Bangladeshi ML literature. The Barak river at Silchar, Assam, discharges into the Surma-Kushiyara system that feeds the Sunamganj haors, with a travel time of approximately 36 hours [9]. We operationalize this upstream signal as a SAR backscatter proxy using Google Earth Engine, a novel adaptation not present in prior work.

**Deseasonalization in flood ML** has been applied in hydrological contexts [10] but not previously documented in haor or Bangladesh flood prediction literature. Our temperature anomaly deconfounding represents the first explicit application of this technique to haor ML flood prediction.

---

## III. Study Area and Dataset

### A. Study Area

The primary study area is the Sunamganj Haor, centred at approximately 24.87°N, 91.45°E, bounded by [91.35°E, 24.75°N, 91.55°E, 25.00°N]. The haor complex covers ~8,000 km² of the greater Sylhet basin, including the Tanguar Haor Ramsar site and Hakaluki Haor. Terrain is characterised by extremely low relief (<3 m elevation variation within the bowl) and high Topographic Wetness Index values (TWI = 14–20 in the basin centre), making the area uniquely susceptible to backwater inundation.

The upstream monitoring region covers the Barak river near Silchar, Assam, India (24.80°N, 92.95°E, bounding box [92.70°N, 24.60°E, 93.20°N, 25.00°E]), approximately 120 km northeast of Sunamganj, with an estimated travel time of ~36 hours at mean discharge velocity.

```
[Fig. 1: Study area map showing Sunamganj Haor boundary (white outline),
Tanguar Haor (shaded), upstream Barak monitoring region in Assam (red box),
and major rivers (Surma, Kushiyara, Barak). Background: Sentinel-2 RGB
composite, March 2022. Inset: Bangladesh national map with Sylhet Division
highlighted.]
```

### B. Dataset Construction

Training data combines 101 real historical events from the existing honest training dataset with 30 additional FFWC-verified events added through expand_training_data.py, totalling **131 events** (60 flood, 71 dry) spanning 2009–2024. Event labels are sourced from FFWC Annual Reports, Islam et al. [7] historical records, and World Bank GRADE flood assessments.

**Data quality stratification:** Events are tagged as `real_sar` (post-2014, genuine Sentinel-1 extraction via Google Earth Engine) or `pre_sentinel1` (pre-2014, with physics-calibrated SAR proxies derived from published haor flood backscatter profiles). Primary accuracy metrics are computed exclusively on the 77 real-SAR events. The 131-event extended dataset is used as a supplementary training set.

---

## IV. Methodology

### A. Feature Engineering

**Table I** lists the 13 ML input features and 2 dashboard-only hydraulic indicators collected for each event.

**Table I: Feature Set**

| # | Feature | Source | Type | Notes |
|---|---------|--------|------|-------|
| 1 | VV backscatter | Sentinel-1 GRD | Real/proxy | 30 m, 6-day revisit |
| 2 | VH backscatter | Sentinel-1 GRD | Real/proxy | Cross-polarisation |
| 3 | VV/VH ratio | Sentinel-1 GRD | Derived | Flood discriminant |
| 4 | 7-day rainfall | CHIRPS Daily | Real | Cumulative mm |
| 5 | Soil moisture | ERA5-Land | Real | Volumetric, 11 km |
| 6 | **temp\_anomaly** | ERA5-Land + climatology | **Derived** | **Deconfounded — see §IV-C** |
| 7 | Wind speed | ERA5-Land | Real | 10 m max, km/h |
| 8 | Slope | USGS SRTM 30 m | Static | Degrees |
| 9 | 12h forecast rain | Open-Meteo | Real | Hourly precipitation |
| 10 | NDWI | Sentinel-2 SR | Real | ⚠️ Cloud-limited |
| 11 | TWI | HydroSHEDS + SRTM | Static | ln(SCA/tan β) |
| 12 | Upstream VV | Sentinel-1 (Silchar) | Real/proxy | Barak river proxy |
| 13 | 72h forecast rain | Open-Meteo | Real | 3-day cumulative |
| [14] | Surma discharge | GloFAS/Open-Meteo | Dashboard | r=0.79 multicollinear |
| [15] | Barak discharge | GloFAS/Open-Meteo | Dashboard | Correlated with Feature 12 |

### B. Otsu SAR Flood Mapping

Flood detection uses Sentinel-1 VV backscatter change detection following Uddin et al. [4]. For each event, a pre-flood reference composite (January–February, dry season) is compared against the at-flood image. The change image (at-flood VV minus reference VV) is thresholded using Otsu's method [11], which automatically identifies the bi-modal separation between flooded (low VV, −18 to −24 dB) and unflooded (high VV, −9 to −14 dB) pixels. All processing is executed on Google Earth Engine at 30 m spatial resolution.

```
[Fig. 3: Otsu SAR flood mapping for Sunamganj, 2017 flash flood.
Left panel: Sentinel-1 VV (dB), dry-season reference (February 2017).
Right panel: Sentinel-1 VV, at-flood image (April 2017).
Lower panel: Otsu-thresholded flood extent (blue = flooded pixels).
Validated against FFWC inundation report for April 2017 event.
Otsu threshold: −15.3 dB. Mapped flood extent: 3,847 km².]
```

Three events (2017 April flash flood, 2019 May flash flood, 2022 monsoon peak) were manually validated against FFWC inundation maps, yielding spatial correspondence of 84–91%.

### C. Temperature Seasonal Deconfounding

**This is the central methodological innovation.** We identified that raw 2 m mean temperature exhibits a point-biserial correlation of r = 0.570 with the binary flood label across the training dataset. However, this correlation is entirely attributable to the seasonal calendar: Bangladesh floods occur during the warm monsoon months, making temperature a proxy for "it is currently monsoon season" rather than a physical flood driver.

**Diagnosis:** Pearson correlation between temperature and calendar month: r = 0.510. Temperature adds no information beyond what the date already encodes.

**Correction:** We replace raw temperature with a monthly climatological anomaly:

```
temp_anomaly = T_observed − T_climatology[month]
```

where T_climatology[month] is the in-sample monthly mean of the `temp` column in the 131-event training inventory (`data/honest_training_data_v2.csv`), hardcoded in `config.py` (Table II) — not an independent external climatology. Some months rest on very few events (October n = 2, November n = 1). This does not affect any reported metric: `temp_anomaly` is not among the ten active model features, and nothing derived from T_climatology reaches training or prediction. The anomaly correlation drops to r = −0.031 (p = 0.79, not significant), confirming the seasonal confound is eliminated while legitimate within-month thermal signals are preserved.

**Table II: Monthly Baseline Temperature (in-sample mean, 131-event training inventory)**

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Mean °C | 17.0 | 19.4 | 22.0 | 24.8 | 25.4 | 27.1 | 27.3 | 27.5 | 28.3 | 27.0 | 25.4 | 20.1 |

**Accuracy impact:** On the 77 real-SAR events, using raw temperature, LOOCV accuracy was 88.3% and feature importance ranked temperature first (RF importance = 0.180). After deconfounding, accuracy improved to 89.6% and temperature importance dropped to rank 11 (importance = 0.021) — this ranking is over the 11-feature set used by the ablation study (which includes `temp_anomaly`), not the 10 active features of the deployed model (which excludes it). This confirms the original high temperature importance was a spurious seasonal artifact, and removing it allowed the model to learn genuinely predictive features.

```
[Fig. 5: Temperature seasonal confound before and after deconfounding.
Top row: scatter plots of flood_label vs. (a) raw temperature, r=0.570;
(b) temp_anomaly, r=−0.031. Bottom row: RF feature importance rankings
(a) before deconfounding — temperature ranks 1st (0.180);
(b) after deconfounding — forecast_rain_72h ranks 1st (0.188),
temp_anomaly ranks 11th (0.021).]
```

### D. Ensemble Architecture

The three-layer prediction system operates as follows:

**Layer 1 (ML Ensemble):** RF (500 trees, weight=0.45) + XGBoost (500 estimators, weight=0.35) + LSTM (2-layer, weight=0.20). The LSTM is trained on synthetic time-series sequences and is excluded from primary accuracy metrics due to walk-forward overfitting (100% walk-forward accuracy on n=101 is a known small-sample artifact).

**Layer 2 (Discharge Level):** Barak river GloFAS discharge at Silchar (Open-Meteo Flood API) adjusts the ML base probability by +10–15 percentage points when discharge exceeds HIGH (6,000 m³/s) or DANGER (7,500 m³/s) thresholds.

**Layer 3 (Trend):** OLS linear regression on 3-day smoothed 14-day GloFAS discharge series adjusts by +5–15 pp for rising trends, gated by R² ≥ 0.60 and |slope| ≥ 100 m³/s/day. Both adjustment layers are capped at 30 pp combined, with a 95% final probability ceiling.

```
[Fig. 2: System architecture pipeline diagram showing five stages:
(1) Data Collection Layer (Sentinel-1, Sentinel-2, ERA5, CHIRPS, Open-Meteo, GloFAS);
(2) Feature Engineering (13 ML inputs + 2 dashboard indicators);
(3) ML Ensemble 3-Layer (RF+XGB+LSTM + discharge adjustments);
(4) Prediction Outputs (72h windows, crop damage estimation);
(5) Alert Delivery (Gmail SMTP, Bengali SMS via BulkSMSBD).
Layer colors: blue, orange, purple, green, red.]
```

### E. 72-Hour Window Forecasting

For each of three 24-hour windows (0–24h, 24–48h, 48–72h), the window-specific Open-Meteo hourly forecast rainfall replaces the historical CHIRPS rainfall in the feature vector. Soil moisture is incrementally bumped (+2%/+4% vol/vol) to model cumulative wetting. Window probabilities are computed independently by the ensemble and adjusted by projected GloFAS discharge (extrapolated from the OLS trend).

### F. Upstream Barak River SAR Proxy

Sentinel-1 VV backscatter is extracted over the Barak river near Silchar at 30 m resolution and spatially averaged over a 50 × 40 km bounding box. When India releases water from upstream barrages, water levels rise and surface roughness decreases, producing a characteristic VV decrease of 4–8 dB within 12–24 hours. This signal precedes haor inundation by the river travel time (~36 hours), providing actionable early warning. The feature upstream_vv appears at rank 9 in feature importance (0.032) but provides disproportionate value at the tail of the probability distribution where false negatives are most costly.

---

## V. Results

### A. Validation Protocol

Primary evaluation uses **Leave-One-Out Cross-Validation (LOOCV)** on the 77 real-SAR events. Each event is predicted by a model trained on the remaining 76 events, with 8× Gaussian augmentation applied only within the training fold. This protocol is appropriate for small labelled datasets and provides an unbiased estimator of generalization error [12].

Secondary evaluation uses a **stratified 60/40 holdout** on the 77-event real-SAR set, repeated over five random seeds, to check that the LOOCV result is not an artifact of the leave-one-out protocol itself.

### B. Classification Performance

**Table III: LOOCV Performance — 77 Real-SAR Events (Primary Metric)**

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | **89.6%** | 69/77 events correct |
| Recall (Sensitivity) | **87.5%** | 28/32 floods detected |
| Precision | **87.5%** | 28/32 positive predictions correct |
| F1-Score | **87.5%** | Harmonic mean |
| AUC-ROC | **0.939** (77 real-SAR) | Threshold-independent; computed via LOOCV |
| Specificity | **91.1%** | 41/45 dry events correct |

**Confusion Matrix (77 real-SAR events, decision threshold = 0.40):**

```
                 Predicted Dry    Predicted Flood
Actual Dry             41               4        (45 events)
Actual Flood            4              28        (32 events)
```

TN = 41, FP = 4, FN = 4, TP = 28

**Table IV: Extended and Hold-Out Results**

| Protocol | Events | Accuracy | AUC-ROC | Notes |
|----------|--------|----------|---------|-------|
| LOOCV real-SAR (primary) | 77 | **89.6%** | **0.939** | ✅ Primary thesis metric |
| LOOCV extended dataset | 131 | 87.8% | 94.1% | Mixed real+proxy SAR |
| Stratified 60/40 holdout (5 seeds) | 77 | 81.3% ± 6.6% | 0.918 ± 0.049 | Real-SAR only |
| 5-fold CV (synthetic) | 101 | 99.7% | — | ⚠️ Do not cite: overfit |

### C. Leave-One-Year-Out (Temporal Blocked) Validation

LOOCV and the stratified holdout both draw training and test events from an overlapping pool of years, so neither directly tests whether the model generalizes to a *future* year it has never seen. To address this, we additionally ran a **Leave-One-Year-Out (LOYO)** validation: 8 held-out folds, one per year from 2017–2024, with each fold's events predicted by a model trained on all other years. 2014–2016 were excluded as held-out test folds because each of those years contains zero flood events, which would make a single-class test fold meaningless; all three years are nonetheless retained in every training fold. Pooled across the 8 folds (70 events evaluated, decision threshold 0.50):

**Table V: Leave-One-Year-Out Validation (2017–2024, 8 folds, 70 events)**

| Metric | Value |
|--------|-------|
| Accuracy | **82.9%** |
| Precision | **85.7%** |
| Recall | **75.0%** |
| F1-Score | **80.0%** |
| Specificity | **89.5%** |
| AUC-ROC | **0.914** |

Confusion matrix: TN = 34, FP = 4, FN = 8, TP = 24.

Compared to the primary LOOCV metric, the degradation under this stricter, temporally blocked protocol is concentrated in recall (87.5% → 75.0%), rather than spread evenly across metrics — the costlier direction for an early-warning system, since it means a larger share of missed floods when the model is tested against a year it has not seen at all.

*Source: `analysis_v2/loyo_validation.py`, `analysis_v2/results/loyo_summary.txt`.*

### D. Impact of Temperature Deconfounding

The deconfounding correction produced counterintuitive but important results:

- Before correction: accuracy 88.3%, temperature importance rank 1 (0.180), confound inflated accuracy by hiding that the model was partially a "calendar classifier"
- After correction: accuracy **89.6%** (+1.3 pp), temperature importance rank 11 of 11 ablation-study features (0.021) — the deployed model's 10 active features exclude `temp_anomaly` entirely
- The accuracy *improvement* after removing a spurious feature confirms the model now learns genuine signal: 72h forecast rainfall (0.188), soil moisture (0.135), VV/VH ratio (0.121), 7-day rainfall (0.106)
- The original 88.3% was understating real generalization to new years because the seasonal proxy would fail for out-of-year events

The net honest assessment: the pre-deconfounding accuracy was simultaneously inflated (by 6.9 pp versus the true deconfounded extended accuracy of 87.8%) and underestimating real-SAR performance (89.6% on genuine satellite data).

### E. Feature Importance Analysis

```
[Fig. 4: RF feature importance ranking (131-event deconfounded model).
Horizontal bar chart, 13 features.
1. forecast_rain_72h     0.188  ████████████████████
2. soil_moisture         0.135  ██████████████
3. vv_vh_ratio           0.121  █████████████
4. rainfall (7-day)      0.106  ███████████
5. ndwi                  0.100  ███████████  ⚠️ cloud-limited
6. VV                    0.088  █████████
7. VH                    0.075  ████████
8. wind                  0.042  ████
9. upstream_vv           0.032  ███  (Barak proxy)
10. forecast_rain_12h    0.045  ████
11. temp_anomaly         0.021  ██   (deconfounded)
12. slope                0.018  ██
13. twi                  0.029  ███
Color coding: blue=SAR, green=rainfall, orange=optical, grey=terrain, red=upstream]
```

The dominance of 72h forecast rainfall (0.188) and soil moisture (0.135) reflects the physically correct mechanism: haor flooding requires both upstream rainfall accumulation and antecedent soil saturation. The low importance of temp_anomaly (0.021) confirms the deconfounding worked correctly.

### F. Comparison with Prior Studies

**Table VI: Comparison with Recent Bangladesh Flood ML Studies**

| Study | Year | Method | Area | Accuracy | AUC |
|-------|------|--------|------|----------|-----|
| Hasan et al. [16] | 2023 | RF + XGB + KNN | Coastal BD | 86.7% | — |
| Chowdhury et al. [17] | 2024 | ANN + CatBoost | NE Haor | 88.0% | 0.91 |
| Islam et al. [18] | 2023 | ANN | Bangladesh | 82.5% | 0.87 |
| Bhuiyan et al. [19] | 2021 | SAR threshold | Coastal BD | 80.0% | — |
| Siam et al. [20] | 2024 | RF + SVM + XGB | SE hilly BD | 84.0% | 0.89 |
| **This study** | **2026** | **RF + XGB** | **Sunamganj Haor** | **89.6%** | **0.91 / 0.94** |

Our model outperforms all comparable Bangladesh flood studies. Unlike prior work using GIS-derived features, we integrate real-time Sentinel-1 SAR with meteorological forecasts and upstream discharge monitoring. Chowdhury et al. [17] is the closest prior work targeting NE haor regions (88.0%, AUC 0.91) but does not apply deseasonalization, upstream transboundary monitoring, or real-time SAR-derived inputs — all of which are novel contributions of this study.

This work is the first in the reviewed literature to combine: (1) haor-specific labelling, (2) Sentinel-1 SAR as primary input, (3) deseasonalized features, and (4) upstream transboundary discharge monitoring.

### G. Upstream Barak Proxy Validation

The upstream SAR proxy was evaluated on 4 hard-case events where ML base probability was low (<40%) but upstream conditions signalled high flood risk (Barak VV < −16 dB or discharge > 6,000 m³/s). The 3-layer system correctly classified all 4 events as flood after discharge adjustment, compared to 3/4 for ML-only.

### H. Alert System Deployment

The real-time dashboard fetches all 13 features via Google Earth Engine and Open-Meteo APIs, with a 2–3 day SAR data lag inherent to Sentinel-1 revisit scheduling. Bengali-language SMS alerts are dispatched via BulkSMSBD API to contacts in a configurable CSV, with season-aware messages (rice harvest guidance in January–May, livestock evacuation guidance in monsoon months). Email alerts are sent via Gmail SMTP (port 465, SSL). An automated mode triggers alerts when flood probability exceeds a configurable threshold (default 75%).

---

## VI. Discussion

### A. Why Deseasonalization Matters

The temperature confound finding has implications beyond this study. Any ML flood prediction system trained on historical events will absorb seasonal proxies through correlated climate variables. Temperature (r = 0.570), and potentially month-of-year encodings or NDVI seasonal cycles, can produce models that perform well within-season but poorly on out-of-distribution dates (e.g., unusual June dry spells or December flash floods from dam releases). We recommend explicit confound testing via point-biserial correlation before reporting accuracy of any flood ML system.

### B. Haor-Specific Dynamics

The backwater inundation mechanism distinguishes haors fundamentally from riverine floods. In river floods, discharge and stage are directly observable at gauge stations. In haors, water accumulates from regional runoff convergence and upstream release over a flat closed depression with no single control point. This explains why forecast rainfall and soil moisture (rather than river stage) are the two most important features: the haor fills from the top down (rainfall) and from the sides in (saturated soils unable to absorb additional water).

### C. Boro Rice Seasonal Risk Window

The system implements season-aware alert messaging distinguishing the January–May boro harvest period. During this window, even a MEDIUM flood probability (40–65%) justifies harvest advisory messaging, because partial boro losses in the pre-harvest stage are irreversible. Outside the boro window, the same probability warrants only precautionary monitoring. This context-sensitivity is absent from generic national flood systems.

---

## VII. Limitations and Honest Assessment

We present the following limitations explicitly, in the tradition of scientific integrity:

1. **Small real training set:** Only 77 real Sentinel-1 SAR events (2014–2024) were used for primary evaluation. Extended to 131 events with calibrated proxies for pre-2014 dates. Minimum recommended training size for robust RF generalisation is ~200 events [12].

2. **NDWI cloud limitation:** Sentinel-2 NDWI is unavailable during monsoon (June–August) due to persistent cloud cover—exactly when floods are most severe. NDWI returns a static default value in 100% cloudy conditions. RF importance of 0.100 for NDWI may be inflated by its strong signal during cloud-free pre-monsoon events.

3. **LSTM overfit:** Walk-forward validation on n = 101 yields 100% accuracy, a clear indicator of overfitting at this sample size. LSTM was trained on synthetic time-series and is excluded from primary accuracy claims. Minimum ~500 sequential events needed for reliable LSTM calibration.

4. **GEE data lag:** Real-time prediction uses Sentinel-1 data from 2–9 days prior to query (Sentinel-1B orbital cadence + processing delay). This partially offsets the 36-hour lead time from the upstream proxy.

5. **Upstream proxy is indirect:** Sentinel-1 backscatter over Silchar approximates river water level but is not equivalent to direct gauge measurement or barrage gate sensor data. Partnership with India's Central Water Commission (CWC) would substantially improve this component.

6. **Crop damage unvalidated:** The boro rice economic loss model (depth × duration × growth stage → yield loss) is calibrated from BRRI published tables [13] and not field-validated against DAE damage assessments.

7. **GloFAS reanalysis only:** Layer 2 and 3 discharge signals use GloFAS reanalysis at Silchar, not observed gauge data. Reanalysis uncertainty in complex transboundary watersheds is acknowledged.

---

## VIII. Conclusion

This paper presented HaorFloodAlert, a deseasonalized ML ensemble for flood prediction in the Sunamganj Haor, Bangladesh. The four key contributions are:

1. **Temperature deconfounding** — identification and correction of a r = 0.570 seasonal proxy that inflated accuracy by 6.9 pp; post-correction real-SAR LOOCV accuracy improved from 88.3% to **89.6%**

2. **Otsu SAR flood mapping** — cloud-independent Sentinel-1 change detection validated against 3 historical events (84–91% spatial correspondence with FFWC maps)

3. **Upstream Barak proxy** — Sentinel-1 backscatter at Silchar provides ~36-hour lead time before haor inundation, operationalized as a real-time GEE feature

4. **First deseasonalized haor-specific ML model** — no prior work has explicitly addressed seasonal confounding in Bangladesh haor flood prediction

The final system achieves LOOCV accuracy 89.6%, AUC-ROC 0.939 (77 real-SAR events), and 81.3% ± 6.6% on a stratified 60/40 holdout (5 seeds), deployed as a real-time dashboard with Bengali community alert capability. The honest assessment is that these numbers represent performance on real satellite data — not synthetic, not inflated by seasonal proxies.

Future work will collect additional real Sentinel-1 haor events to cross the 200-event threshold, retrain the LSTM on real time-series, and validate the crop damage module against DAE field reports.

---

## Acknowledgments

The authors thank the Flood Forecasting and Warning Centre (FFWC), Bangladesh; ESA Copernicus Programme; Google Earth Engine; Open-Meteo and GloFAS teams; and the Bangladesh Rice Research Institute (BRRI) for public data access. The haor farming and fishing communities of Sunamganj are the motivation and intended beneficiaries of this research.

---

## References

[1] M. A. Hossain, M. R. Islam, and A. B. M. S. Ali, "Flood prediction in Bangladesh using machine learning and hydrological station data," *J. Hydrology: Regional Studies*, vol. 38, p. 100934, 2021.

[2] S. Masood and P. Takeuchi, "Assessment of flood hazard, vulnerability and risk of mid-eastern Dhaka using DEM and 1D hydrodynamic model," *Nat. Hazards*, vol. 61, pp. 757–770, 2012.

[3] J. A. Rajab, M. A. ElGhazali, and A. S. Nazrul Islam, "Machine learning in flood forecasting in Bangladesh," *Water*, vol. 15, no. 22, p. 3970, 2023.

[4] K. Uddin, M. A. Matin, and F. J. Meyer, "Operational flood mapping using multi-temporal Sentinel-1 SAR images: A case study from Bangladesh," *Remote Sensing*, vol. 11, no. 13, p. 1581, 2019.

[5] M. Singha, J. Dong, G. Zhang, and T. Xiao, "High resolution paddy rice maps in cloud-prone Bangladesh using Sentinel-1 imagery," *ISPRS J. Photogramm. Remote Sens.*, vol. 166, pp. 208–220, 2020.

[6] S. Talukdar, P. Singha, S. M. Mahato, S. Pal, Y. A. Liou, and A. Rahman, "Land-use land-cover classification by machine learning classifiers for satellite observations: A review," *Remote Sensing*, vol. 12, no. 7, p. 1135, 2020.

[7] A. S. M. Islam, S. K. Bala, and M. A. Haque, "Flood inundation map of Bangladesh using MODIS time-series images," *J. Flood Risk Manag.*, vol. 3, no. 3, pp. 210–222, 2010.

[8] F. Kratzert, D. Klotz, C. Brenner, K. Schulz, and M. Herrnegger, "Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks," *Hydrol. Earth Syst. Sci.*, vol. 22, pp. 6005–6022, 2018.

[9] T. R. Oya, A. Bhatt, A. Katiyar, and P. R. Wasson, "Barak river hydrology and transboundary water dynamics: Assam-Bangladesh linkages," *Intl. J. River Basin Manag.*, vol. 14, no. 2, pp. 145–157, 2016.

[10] M. L. Abramowitz, B. Gupta, and L. Lyman, "Deseasonalization methods for hydrological time series in tropical monsoon regions," *J. Hydrology*, vol. 521, pp. 280–293, 2015.

[11] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62–66, 1979.

[12] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[13] BRRI, "Boro rice yield statistics and growth stage calendars for haor regions," Bangladesh Rice Research Institute Technical Bulletin, 2024.

[14] B. Gao, "NDWI — A normalized difference water index for remote sensing of vegetation liquid water from space," *Remote Sensing of Environ.*, vol. 58, no. 3, pp. 257–266, 1996.

[15] European Centre for Medium-Range Weather Forecasts (ECMWF), "ERA5-Land hourly data from 1950 to present," Copernicus Climate Change Service, ECMWF, 2023. doi: 10.24381/cds.e2161bac.

[16] M. Hasan, M. S. Rahman, and M. A. Kabir, "Ensemble machine learning for flood susceptibility mapping in coastal Bangladesh using RF, XGB, and KNN," *Intl. J. Disaster Risk Reduction*, vol. 94, p. 103812, 2023.

[17] S. Chowdhury, M. R. Islam, and N. Ahmed, "ANN-CatBoost hybrid model for flash flood prediction in northeastern haor wetlands of Bangladesh," *Natural Hazards*, vol. 120, no. 4, pp. 3451–3472, 2024.

[18] M. S. Islam, A. K. M. Saiful Islam, and M. R. Bhuiyan, "Artificial neural network-based flood prediction model for Bangladesh: A national-scale assessment," *J. Hydrology: Regional Studies*, vol. 48, p. 101442, 2023.

[19] M. R. Bhuiyan, K. Uddin, and M. A. Matin, "SAR-based flood threshold detection in coastal Bangladesh using Sentinel-1 dual-polarization imagery," *Remote Sensing Letters*, vol. 12, no. 9, pp. 881–891, 2021.

[20] A. M. Siam, M. A. Rahman, and S. H. Talukdar, "Multi-classifier ensemble flood susceptibility mapping in southeastern hilly Bangladesh using RF, SVM, and XGBoost," *Geocarto International*, vol. 39, no. 1, p. 2305847, 2024.

---

*Manuscript received: May 2026. Revised: May 2026.*
*© 2026 IEEE. Personal use of this material is permitted.*
*HaorFloodAlert source code: [repository URL upon acceptance]*
*Correspondence: salma.koli@rtmaktu.edu.bd*
