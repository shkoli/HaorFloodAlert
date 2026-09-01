"""
4_About.py — Project Overview with Honest Limitations
HaorFloodAlert — 15-feature system (13 ML inputs + Surma discharge + Barak discharge dashboards)
"""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About — HaorFloodAlert")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Project Overview

    **HaorFloodAlert** is an undergraduate thesis prototype that delivers real-time
    flood risk predictions and community alerts for the haor wetland regions of
    Sylhet–Sunamganj, Bangladesh — one of the world's most flood-vulnerable areas.

    Haors are shallow, bowl-shaped wetlands covering roughly **8,000 km²** in northeast
    Bangladesh. Flash floods triggered by pre-monsoon rainfall from upstream India
    frequently destroy the *boro* rice harvest before it can be collected, threatening
    both food security and the livelihoods of millions.

    Existing government systems (BWDB/FFWC) focus on riverine flood levels and lack
    haor-specific inundation forecasts. This system addresses that gap with a
    **15-feature system** combining SAR, optical, topographic,
    meteorological, upstream river, and hydraulic discharge data.
    13 features are used as ML model inputs; Surma GloFAS discharge and
    Barak river GloFAS discharge are retained as real-time hydraulic
    dashboard indicators (both tested as ML features and excluded due to
    multicollinearity with existing model variables).

    ---

    ## Novel Contributions

    | # | Contribution | Technical Detail |
    |---|---|---|
    | 1 | **Otsu SAR change detection** | Pre-flood (Jan–Feb) vs at-flood Sentinel-1 VV change image; Otsu bi-modal threshold separates flooded from dry pixels — methodology of Uddin et al. (2019) and Singha et al. (2020) |
    | 2 | **Upstream Barak river SAR proxy** | Sentinel-1 VV over Silchar (Assam) detects barrage release ~36h before haor inundation |
    | 3 | **NDWI (Sentinel-2)** | Optical water index combined with SAR for dual-sensor flood detection. ⚠️ **Cloud limitation:** Sentinel-2 is blocked by cloud cover during peak monsoon — NDWI falls back to a default value exactly when floods are most intense. SAR (cloud-penetrating) is the primary sensor; NDWI is secondary. |
    | 4 | **TWI (HydroSHEDS)** | Topographic Wetness Index captures haor bowl-shape susceptibility |
    | 5 | **72-hour rainfall forecast** | 3-day cumulative enables pre-monsoon early warning |
    | 6 | **Flood lead time estimation** | ~36h lead time from upstream VV signal — first haor-specific model |
    | 7 | **Boro rice crop damage (BRRI-calibrated)** | Depth × duration × growth stage damage model with upazila-level economic loss |
    | 8 | **ML uncertainty quantification** | 95% CI from 500 RF tree predictions — rare in undergraduate haor thesis |
    | 9 | **3-model ensemble** | RF + XGBoost + LSTM — most haor studies use only one model |
    | 10 | **Surma GloFAS discharge (dashboard)** | Open-Meteo GloFAS reanalysis at Sunamganj (24.87°N, 91.40°E); evaluated as ML Feature 14, excluded due to multicollinearity (r=0.79 with soil moisture); retained as real-time hydraulic indicator alongside ML prediction |
    | 11 | **Barak river discharge monitoring** | GloFAS discharge at Silchar (24.82°N, 92.79°E) via Open-Meteo Flood API; 14-day trend + 72h projection; DANGER/HIGH/RISING thresholds (7500/6000/4000 m³/s); discharge-adjusted probability shown alongside ML base probability; 4/4 correct classification on validation hard cases |

    ---

    ## Methodology

    1. **Data Collection** — Sentinel-1 SAR (VV/VH), Sentinel-2 NDWI, ERA5-Land soil
       moisture, CHIRPS 7-day rainfall, SRTM slope, HydroSHEDS TWI, Barak river
       upstream VV, Open-Meteo 12h/72h forecast — all via Google Earth Engine + APIs.
    2. **Feature Engineering** — 15 features collected; 13 used as ML inputs.
       Surma GloFAS discharge excluded (multicollinearity r=0.79 with soil moisture).
       Barak GloFAS discharge excluded (likely correlated with upstream_vv; pending
       multicollinearity test — see `add_barak_discharge.py`). Both retained as
       real-time hydraulic dashboard indicators.
    3. **Model Training** — RF (500) + XGBoost (500) trained on 72 real-SAR events
       (LOOCV with 8× Gaussian augmentation). LSTM (2-layer) trained separately on
       synthetic time-series sequences; **excluded from primary metric** — walk-forward
       validation on 101 rows yields 100% accuracy (overfit due to small n).
       LSTM included in the ensemble only when its model file is present; primary
       performance claims are for RF + XGBoost only.
    4. **Primary Validation** — **Leave-One-Out Cross-Validation (LOOCV) on 72 real,
       post-Sentinel-1 events with genuine satellite data (2014–2024):**
       Accuracy **88.9%**, Recall **93.8%**, F1 **88.2%**, AUC-ROC **89.9%**.
       This is the thesis's main reported metric (100% real satellite inputs).
    5. **Supplementary Validation** — 45 historical events (2017–2024) using Open-Meteo
       archived rainfall + physics-calibrated SAR/NDWI proxies (not live GEE values).
       Results are indicative, not independent of training data.
    6. **Dashboard** — Streamlit app with live map, 5-day forecast, 72-hour
       multi-window forecast, upstream Barak discharge monitoring, flood duration,
       crop damage estimation, and Gmail email community alerts.

    ---

    ## Accuracy & Generalization Analysis

    > **For thesis reviewers:** This section explains what each accuracy figure actually
    > measures, so results are not over-stated.

    | Metric | Value | What it measures | Reliability |
    |--------|-------|-----------------|-------------|
    | **LOOCV — 77 real-SAR events** | **89.6%** | Real Sentinel-1/CHIRPS/ERA5 data (2014–2024). `temp_anomaly` replaces raw temp — seasonal confound removed. | ✅ **Primary metric** |
    | Recall / Precision / F1 | **87.5% / 87.5% / 87.5%** | Balanced: 28/32 floods caught, 4/45 dry misclassified | ✅ |
    | AUC-ROC | **93.6%** | Area under ROC curve — threshold-independent | ✅ |
    | LOOCV — 131 events (extended) | **87.8%** | Adds 30 FFWC-verified events (2009–2024) with real Open-Meteo rainfall + calibrated SAR proxies. Seasonal confound removed. | ✅ Extended (mixed real+proxy SAR) |
    | Extended F1 / AUC | 86.4% / 94.1% | From 131-event LOOCV; currently deployed models | ✅ Extended |
    | **45-event hold-out validation** | **86.7% (5-seed stratified)** | 2017–2024 events; Open-Meteo archive + calibrated SAR proxies; independent complement to LOOCV | ⚠️ Proxy-based |
    | 5-fold CV on synthetic data | 99.7% ± 0.2% | Trained and tested on same distribution — confirms separability, **not** real-world accuracy | ⚠️ Inflated — do not cite |

    > **Temperature deconfounding:** Raw temperature had r=0.570 with flood label — the model
    > was learning "warm months = monsoon = flood" rather than causal flood physics. Replacing
    > with `temp_anomaly = observed − monthly_climatological_mean` reduced this to r=−0.031.
    > The previous inflated score (94.7%) dropped to an honest 87.8%, and the real-SAR LOOCV
    > *improved* from 88.3% to 89.6% — the model now learns genuine signal, not a calendar proxy.

    **LOOCV Confusion Matrix — real-SAR only (77 events, deconfounded):**
    TN = 41 · FP = 4 · FN = 4 · TP = 28

    **LOOCV Confusion Matrix — extended (131 events, deconfounded):**
    TN = 64 · FP = 7 · FN = 9 · TP = 51

    For a flood alert system, **FN (missed floods) are more dangerous than FP (false
    alarms)**. Lowering the decision threshold to 0.40 improves recall at
    the cost of more false alarms — tunable on the Validation page.

    ---

    ## ⚠️ Honest Limitations

    > Acknowledging limitations is a sign of scientific maturity, not weakness.

    | # | Limitation | Impact | Future Fix |
    |---|---|---|---|
    | 1 | **Training data is synthetic** | Model calibrated from literature, not real GEE samples | Collect real GEE data (see `collect_real_haor_data.py`) |
    | 2 | **Validation uses SAR proxies** | VV/NDWI/soil are physics-calibrated, not actual satellite values for historical dates | Live GEE calls for each event (20+ min wait) |
    | 3 | **Upstream proxy is indirect** | Sentinel-1 over Silchar ≠ actual barrage gate sensors | Partner with India CWC/BWDB for real gauge data |
    | 4 | **NDWI unusable during peak floods** | Sentinel-2 is 100% cloud-blocked during June–August monsoon. NDWI returns a static default value at the exact time floods are most severe — making it a weak feature for the highest-risk events. RF importance weight 0.200 may be inflated by dry-season correlation. | Use Sentinel-1 SAR (cloud-penetrating) as primary; collect cloud-free NDWI only for pre-monsoon flash flood window (Mar–May) |
    | 5 | **Crop damage numbers are estimates** | Yield (45t/km²), price (28k BDT), and historical data are literature-based approximations | Field surveys with BRRI/BBS |
    | 6 | **GEE has 2–3 day data lag** | Real-time prediction actually uses data from 2–9 days ago | Subscribe to Copernicus DIAS for faster access |
    | 7 | **TWI is terrain-static** | HydroSHEDS TWI does not change with season or cultivation | Re-compute with seasonal DEM updates |
    | 8 | **LSTM overfit — excluded from primary metric** | Walk-forward validation on n=101 rows yields 100% accuracy — a clear sign of overfit, not true generalisation. LSTM was trained on synthetic time-series sequences; real haor SAR sequences are unavailable in sufficient quantity (minimum ~500 needed). LSTM contributes weight=0.20 when loaded but its contribution is cosmetic at current data size. | Collect 5+ years of daily Sentinel-1 time-series; retrain LSTM; validate with proper temporal hold-out |
    | 9 | **Small real training dataset** | Only 12 real GEE rows collected (all with rainfall=120 hardcoded) | Recollect with `collect_real_haor_data.py` |
    | 10 | **Flood duration model is empirical** | Duration formula is calibrated from BWDB records, not physics-based | Train dedicated LSTM on BWDB gauge time-series |
    | 11 | **TWI and slope are constant across all real training rows** | Both features = same value for every event (TWI=17.185, slope=1.91); dropped in LOOCV model, but kept in synthetic model — creates train/test feature mismatch | Collect per-event TWI from GEE at event dates or remove from feature set |

    ---

    ## Data Sources & Honesty

    | Data | Type | Source | Quality |
    |------|------|--------|---------|
    | SAR VV/VH | ✅ Real (live) | Sentinel-1 GRD / GEE | High — 30m, 6-day |
    | NDWI | ⚠️ Real (cloud-limited) | Sentinel-2 / GEE | **Low during monsoon** — cloud cover blocks Sentinel-2 Jun–Aug; returns default value during peak flood season |
    | Soil moisture | ✅ Real (live) | ERA5-Land / GEE | Medium — 11km coarse |
    | Rainfall | ✅ Real | CHIRPS + Open-Meteo | High |
    | Forecast | ✅ Real | Open-Meteo | High |
    | Slope / TWI | ✅ Real (static) | SRTM + HydroSHEDS | High |
    | Upstream VV | ✅ Real (live) | Sentinel-1 Silchar (92.79°E) | Indirect proxy |
    | Surma discharge | ✅ Real (dashboard) | GloFAS / Open-Meteo Flood API | Dashboard only — multicollinear |
    | Barak discharge | ✅ Real (dashboard) | GloFAS / Open-Meteo Flood API | Dashboard + heuristic adjustment |
    | Training data | ⚠️ Synthetic | Physics-calibrated | Literature-based |
    | Validation SAR | ⚠️ Proxy | Calibrated from profiles | Not actual satellite |
    | Crop damage | ⚠️ Estimated | BRRI/BBS literature | Approximate |

    ---

    ## Tech Stack

    | Component | Technology |
    |-----------|------------|
    | SAR data | Copernicus Sentinel-1 GRD via Google Earth Engine |
    | Optical data | Copernicus Sentinel-2 SR via GEE (NDWI) |
    | Rainfall | CHIRPS Daily / Open-Meteo archive & forecast |
    | Soil moisture | ECMWF ERA5-Land |
    | Terrain | USGS SRTM DEM + WWF HydroSHEDS (TWI) |
    | Upstream proxy | Sentinel-1 over Barak river, Silchar, Assam |
    | ML models | scikit-learn RandomForest + XGBoost + PyTorch LSTM |
    | Dashboard | Streamlit + Folium |
    | Community alerts | Gmail SMTP (primary) · Telegram Bot + WhatsApp API (tested) |
    | Language | Python 3.11 |

    ---

    ## References

    - Islam et al. (2021). Flood susceptibility modelling using advanced ensemble ML. *Geoscience Frontiers*.
    - Singha et al. (2020). Sentinel-1 flood mapping on GEE in Bangladesh. *ISPRS J. Photogramm.*
    - Uddin et al. (2019). Multi-temporal Sentinel-1 operational flood mapping. *Remote Sensing*.
    - Talukdar et al. (2020). Ensemble bagging for flood susceptibility, Teesta basin. *Model Earth Syst Environ.*
    - Rajab et al. (2023). ML in flood forecasting in Bangladesh. *Water*, 15(22), 3970.
    - Gao (1996). NDWI — A normalized difference water index. *Remote Sensing of Environment.*
    - Conrad et al. (2011). SAGA GIS for geoscientific analyses. *Trans. GIS.*
    - BRRI (2024). Boro rice yield statistics. Bangladesh Rice Research Institute.
    - BWDB (2024). Flood damage assessment reports 2017–2024. Bangladesh Water Development Board.

    ---

    ## 🙏 Acknowledgments

    The author gratefully acknowledges the following individuals and organisations:

    - **Md. Samiul Alim** (Supervisor, RTM Al-Kabir Technical University) — for guidance,
      technical direction, and academic mentorship throughout the thesis research.
    - **FFWC — Flood Forecasting & Warning Centre, Bangladesh** — for publicly available
      flood records and hydrological station data that informed model calibration.
    - **ESA Copernicus Programme** — for free access to Sentinel-1 SAR and Sentinel-2
      optical imagery that forms the backbone of this system's real-time sensing capability.
    - **Google Earth Engine** — for providing cloud-based satellite image processing
      infrastructure without which real-time prediction at this scale would be infeasible.
    - **Open-Meteo and GloFAS teams** — for open-access weather forecast and discharge APIs.
    - **BRRI (Bangladesh Rice Research Institute)** — for published Boro rice phenological
      calendars and yield statistics used in the crop damage estimation module.
    - **The haor farming and fishing communities of Sunamganj** — whose vulnerability to
      flash floods is the entire motivation for this research.

    ---

    """)

    st.markdown("## 🏗️ System Architecture")

    _arch_html = """
<div style="font-family:sans-serif;max-width:700px;margin:8px 0 20px 0">

  <div style="border-radius:12px;overflow:hidden;margin-bottom:4px;box-shadow:0 2px 10px rgba(0,0,0,0.4)">
    <div style="background:#1a5fa0;padding:10px 18px;display:flex;align-items:center;gap:10px">
      <span style="font-size:1.1rem">📡</span>
      <span style="color:#fff;font-weight:700;font-size:0.93rem;letter-spacing:0.6px;text-transform:uppercase">Data Collection Layer</span>
    </div>
    <div style="background:#0d2d4a;padding:12px 18px;color:#9ecfee;font-size:0.87rem;line-height:1.75">
      Sentinel-1 SAR (VV / VH) &nbsp;·&nbsp; Sentinel-2 NDWI &nbsp;·&nbsp; ERA5-Land Soil Moisture &nbsp;·&nbsp; CHIRPS Rainfall<br>
      SRTM Slope &nbsp;·&nbsp; HydroSHEDS TWI &nbsp;·&nbsp; Open-Meteo 12h / 72h Forecast<br>
      GloFAS Barak Discharge &nbsp;·&nbsp; GloFAS Surma Discharge
    </div>
  </div>

  <div style="text-align:center;padding:3px 0;color:#607d8b;font-size:0.8rem;letter-spacing:1px">▼ &nbsp;15 features collected</div>

  <div style="border-radius:12px;overflow:hidden;margin-bottom:4px;box-shadow:0 2px 10px rgba(0,0,0,0.4)">
    <div style="background:#a96b14;padding:10px 18px;display:flex;align-items:center;gap:10px">
      <span style="font-size:1.1rem">⚙️</span>
      <span style="color:#fff;font-weight:700;font-size:0.93rem;letter-spacing:0.6px;text-transform:uppercase">Feature Engineering</span>
    </div>
    <div style="background:#2b1800;padding:12px 18px;font-size:0.87rem;line-height:1.75">
      <span style="color:#e8c880">VV/VH ratio &nbsp;·&nbsp; NDWI &nbsp;·&nbsp; TWI &nbsp;·&nbsp; 72h forecast &nbsp;·&nbsp; upstream VV &nbsp;·&nbsp; temp_anomaly &nbsp;·&nbsp; soil moisture &nbsp;·&nbsp; wind</span><br>
      <span style="color:#4fc3f7;font-weight:600">13 features → ML models</span>
      &nbsp;&nbsp;<span style="color:#455a64">|</span>&nbsp;&nbsp;
      <span style="color:#ffb74d">2 features → dashboard only</span>
      <span style="color:#546e7a;font-size:0.8rem"> (Surma + Barak — multicollinear)</span>
    </div>
  </div>

  <div style="text-align:center;padding:3px 0;color:#607d8b;font-size:0.8rem">▼</div>

  <div style="border-radius:12px;overflow:hidden;margin-bottom:4px;box-shadow:0 2px 10px rgba(0,0,0,0.4)">
    <div style="background:#6b37a0;padding:10px 18px;display:flex;align-items:center;gap:10px">
      <span style="font-size:1.1rem">🤖</span>
      <span style="color:#fff;font-weight:700;font-size:0.93rem;letter-spacing:0.6px;text-transform:uppercase">ML Ensemble — 3-Layer Prediction</span>
    </div>
    <div style="background:#160a26;padding:12px 18px;font-size:0.87rem;line-height:1.9">
      <span style="color:#d4b8f8"><b style="color:#e8d8ff">Layer 1:</b> RF (w=0.45) + XGBoost (w=0.35) + LSTM (w=0.20) = ML base probability</span><br>
      <span style="color:#b499e0"><b style="color:#cc99ff">Layer 2:</b> Barak discharge level (DANGER / HIGH) → +0–15 percentage points</span><br>
      <span style="color:#9a7fc8"><b style="color:#b088ee">Layer 3:</b> 14-day rising trend (R²-gated OLS) → +0–15 percentage points</span><br>
      <span style="color:#5a5070;font-size:0.8rem">Combined discharge cap: 30 pp &nbsp;·&nbsp; Final probability cap: 95%</span>
    </div>
  </div>

  <div style="text-align:center;padding:3px 0;color:#607d8b;font-size:0.8rem;letter-spacing:1px">▼ &nbsp;Final flood probability (%)</div>

  <div style="border-radius:12px;overflow:hidden;margin-bottom:4px;box-shadow:0 2px 10px rgba(0,0,0,0.4)">
    <div style="background:#1e8048;padding:10px 18px;display:flex;align-items:center;gap:10px">
      <span style="font-size:1.1rem">📊</span>
      <span style="color:#fff;font-weight:700;font-size:0.93rem;letter-spacing:0.6px;text-transform:uppercase">Prediction Outputs</span>
    </div>
    <div style="background:#071a10;padding:12px 18px;color:#90dba8;font-size:0.87rem;line-height:1.75">
      72-hour 3-window forecast &nbsp;(0–24h &nbsp;·&nbsp; 24–48h &nbsp;·&nbsp; 48–72h)<br>
      5-day rainfall-driven forecast animation<br>
      Upstream discharge DANGER / HIGH / RISING alert &nbsp;(~36 h lead time)<br>
      Boro rice crop damage estimation &nbsp;(BDT crore &nbsp;·&nbsp; tons)
    </div>
  </div>

  <div style="text-align:center;padding:3px 0;color:#607d8b;font-size:0.8rem">▼</div>

  <div style="border-radius:12px;overflow:hidden;margin-bottom:2px;box-shadow:0 2px 10px rgba(0,0,0,0.4)">
    <div style="background:#a02020;padding:10px 18px;display:flex;align-items:center;gap:10px">
      <span style="font-size:1.1rem">🚨</span>
      <span style="color:#fff;font-weight:700;font-size:0.93rem;letter-spacing:0.6px;text-transform:uppercase">Alert Delivery</span>
    </div>
    <div style="background:#260808;padding:12px 18px;color:#f0aaaa;font-size:0.87rem;line-height:1.75">
      Gmail SMTP email &nbsp;(Bengali + English)<br>
      Streamlit dashboard &nbsp;(farmers &nbsp;·&nbsp; DDMC focal points &nbsp;·&nbsp; union parishad leaders)
    </div>
  </div>

</div>
"""
    st.markdown(_arch_html, unsafe_allow_html=True)
    st.markdown("""

    ---

    ## 📖 How to Cite

    **Thesis citation (APA 7th edition):**

    > Talukdar Koli, S. H. (2026). *Flood prediction in the haor regions of Bangladesh
    > using machine learning and satellite-based data* [Undergraduate thesis].
    > Department of Computer Science & Engineering,
    > RTM Al-Kabir Technical University, Sylhet, Bangladesh.

    **System/software citation:**

    > Talukdar Koli, S. H. (2026). HaorFloodAlert v2.0 — Haor Flood Early Warning System
    > [Computer software]. RTM Al-Kabir Technical University.

    **Key methods cited:**

    > Uddin, K., Matin, M. A., & Meyer, F. J. (2019). Multi-temporal Sentinel-1 SAR based
    > operational flood mapping. *Remote Sensing*, 11(13), 1581.

    > Singha, M., et al. (2020). Flood and flood impacted paddy rice mapping using
    > Sentinel-1 imagery on Google Earth Engine in Bangladesh. *ISPRS Journal of
    > Photogrammetry and Remote Sensing*, 166, 278–293.
    """)

with col2:
    st.markdown("""
    ## Developer

    **Salma Hoque Talukdar Koli**
    Student, Computer Science & Engineering
    RTM Al-Kabir Technical University

    ---

    ## Contributors

    **Salma Hoque Talukdar Koli**
    RTM Al-Kabir Technical University, Sylhet

    **Fahima Haque Talukder Jely**
    North East University Bangladesh, Sylhet

    ---

    ## Project Info

    **Title:**
    Flood Prediction in the Haor Regions
    of Bangladesh Using Machine Learning
    and Satellite-Based Data

    **Supervisor:** Md. Samiul Alim
    **Institution:** RTM Al-Kabir Technical University
    **Department:** Computer Science & Engineering
    **Year:** 2026

    ---

    ## Study Area

    **Region:** Sunamganj Haor
    **Bbox:** 91.35–91.55°E, 24.75–25.00°N
    **Key haors:** Tanguar, Hakaluki
    **Total area:** ~8,000 km²

    **Upstream proxy:**
    Barak river, Silchar (Assam, India)
    Travel time to Haor: **~36 hours**

    ---

    ## Model Performance

    **Primary (thesis) metric:**
    LOOCV on **77 real-SAR events** (2014–2024)
    **Accuracy: 88.3%**
    Recall: 87.5% | F1: 86.2%
    AUC-ROC: 94.3%

    **Extended dataset (131 events, 2009–2024):**
    Accuracy: 94.7% | F1: 94.2% | AUC: 96.7%
    *(includes proxy SAR for pre-2017 events)*

    | Metric | Value | Data |
    |--------|-------|------|
    | LOOCV Accuracy | **88.3%** | 77 real-SAR events ✅ |
    | Recall | 87.5% | Real ✅ |
    | F1 Score | 86.2% | Real ✅ |
    | AUC-ROC | 94.3% | Real ✅ |
    | Extended LOOCV | **94.7%** | 131 events (mixed) ✅ |
    | 45-event hold-out | **86.7% (5-seed stratified)** | Proxy ⚠️ |
    | 5-fold CV | 99.7% | Synthetic ⚠️ |

    Features collected: **15** · ML inputs: **11 active** | Training rows: **131**

    ---

    ## Feature Importance (RF, 131-event model)

    1. Temperature (0.180)
    2. Forecast 72h rain (0.158) ★
    3. VV/VH ratio (0.113)
    4. Soil moisture (0.105)
    5. VV backscatter (0.091)
    6. 7-day rainfall (0.081)
    7. NDWI — Sentinel-2 (0.077) ★⚠️
    8. VH backscatter (0.075)
    9. Forecast 12h rain (0.045)
    10. Wind speed (0.042)
    11. Upstream VV — Barak (0.032) ★

    ★ = Novel feature · ⚠️ cloud-limited during monsoon

    ---

    ## Alert System

    - **Gmail SMTP** — primary (email alerts)
    - **Demo mode** — thesis presentation
    - **Languages** — Bangla + English
    - *(Telegram Bot + WhatsApp API: implemented and tested; removed from production page for simplicity)*

    ---

    ## Quick Links

    🔮 [Prediction page](Prediction) — live forecast
    📊 [Validation page](Validation) — accuracy details
    🌾 [CropDamage page](CropDamage) — agricultural impact
    """)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#888;text-align:center;line-height:1.7'>"
        "🌊 <b>HaorFloodAlert v2.0</b><br>"
        "© 2026 Salma Hoque Talukdar Koli<br>"
        "RTM Al-Kabir Technical University<br>"
        "CSE Thesis Project"
        "</div>",
        unsafe_allow_html=True,
    )
