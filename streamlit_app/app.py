import streamlit as st

st.set_page_config(
    page_title="HaorFloodAlert",
    layout="wide",
    page_icon="🌊",
)

# Professional Header
st.markdown(
    """
    <div style='background:linear-gradient(135deg,#0a3d62 0%,#1a5276 50%,#1a7a4a 100%);
    padding:28px 36px;border-radius:14px;margin-bottom:20px;
    box-shadow:0 4px 16px rgba(0,0,0,0.18)'>
    <h1 style='color:white;margin:0;font-size:2.4rem;font-weight:700;letter-spacing:-0.5px'>
    🌊 HaorFloodAlert
    </h1>
    <p style='color:#b8e4f9;margin:7px 0 4px 0;font-size:1.2rem;font-weight:500'>
    সুনামগঞ্জ হাওর বন্যা পূর্বাভাস ব্যবস্থা
    </p>
    <p style='color:#d4f1e4;margin:0;font-size:1.0rem'>
    Haor Flood Early Warning System &nbsp;·&nbsp; Sunamganj, Bangladesh
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Developer & System Stats row
col_dev, col_sep, col_stats = st.columns([3, 0.1, 2])

with col_dev:
    st.markdown("""
### 👩‍💻 Project Information

**Developed by:** Salma Hoque Talukdar Koli

**Contributors:**
Salma Hoque Talukdar Koli (RTM Al-Kabir Technical University, Sylhet) &
Fahima Haque Talukder Jely (North East University Bangladesh, Sylhet)

**Department:** Computer Science & Engineering
**Institution:** RTM Al-Kabir Technical University, Sylhet, Bangladesh
**Supervisor:** Md. Samiul Alim
**Thesis Year:** 2026

**Title:** *Flood Prediction in the Haor Regions of Bangladesh Using
Machine Learning and Satellite-Based Data*
    """)

with col_stats:
    st.markdown("### 📊 Validated Performance")
    m1, m2 = st.columns(2)
    m1.metric("LOOCV Accuracy", "89.6%",
              delta="77 real-SAR events (primary)",
              delta_color="off",
              help="Leave-One-Out CV on 77 real Sentinel-1 SAR events (2014–2024). "
                   "temp_anomaly replaces raw temp — seasonal confound removed.")
    m2.metric("Hold-out Accuracy", "86.7%",
              delta="5-seed stratified mean",
              delta_color="off",
              help="Independent hold-out on 45 historical events (2017–2024). Conservative real-world estimate.")
    m3, m4 = st.columns(2)
    m3.metric("Recall / F1", "87.5% / 87.5%",
              delta="Real-SAR LOOCV (77 events)",
              delta_color="off",
              help="Recall: 28/32 real floods correctly detected. F1 = harmonic mean.")
    m4.metric("AUC-ROC", "93.6%", delta_color="off",
              help="Real-SAR LOOCV (77 events). Extended 131-event LOOCV: 87.8% acc, AUC 94.1%.")

st.divider()

# Key Features
st.markdown("### 🚀 System Capabilities")
feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown("""
🛰️ **Multi-sensor flood detection**
Sentinel-1 SAR (Otsu change detection) + Sentinel-2 NDWI + HydroSHEDS TWI.
First haor model combining SAR change detection with optical and topographic features.

---

📊 **72-hour multi-window forecast**
ML ensemble (RF + XGBoost + LSTM) runs for 0–24 h, 24–48 h, 48–72 h windows using
Open-Meteo hourly precipitation forecast. Outputs per-window flood probability.

---

🌊 **3-layer discharge adjustment**
Layer 1: ML ensemble base probability (satellite + rainfall)
Layer 2: Barak river current level (DANGER/HIGH classification, +10–15 pp)
Layer 3: 14-day rising trend (R²-gated reliability, +5–15 pp)

---

🌾 **Boro rice crop damage estimation**
First haor-specific model: depth × duration × BRRI growth stage → yield loss (tons) +
economic loss (BDT crore) at upazila level. Covers ~8,000 km² Sunamganj haor.
    """)

with feat_col2:
    st.markdown("""
🔺 **Upstream Barak river monitoring**
GloFAS discharge at Silchar, Assam (24.82°N 92.79°E) via Open-Meteo Flood API.
DANGER >7,500 m³/s · HIGH >6,000 m³/s · WARNING >4,000 m³/s.
Provides **~36 h lead time** before haor inundation.

---

📈 **14-day discharge trend analysis**
OLS linear regression on 3-day smoothed GloFAS series.
R² reliability gate: trend suppressed if R² < 0.60 or |slope| < 100 m³/s/day.
72-hour discharge projection with flat-line fallback for noisy signals.

---

🔬 **ML uncertainty quantification**
95% confidence interval from 500 RF tree predictions.
High-uncertainty warnings trigger when CI spans > 20 percentage points.

---

📧 **Community alert system**
Gmail SMTP email alerts in Bengali + English for Sunamganj haor communities.
Target: farmers, fishers, union parishad leaders, DDMC focal points.
    """)

st.divider()

# Navigation
st.markdown("### 🗂️ Application Pages")
nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.markdown("""
| Page | What you'll find |
|------|-----------------|
| 🔮 **Prediction** | Live 72 h forecast · 3-layer breakdown · 5-day rainfall animation · discharge trend chart |
| 🗺️ **Map** | Sentinel-1 SAR water mask · Otsu flood extent · Tanguar Haor overlay |
| 🚨 **Alerts** | Gmail community alert system · Bengali + English templates · demo mode |
    """)
with nav_col2:
    st.markdown("""
| Page | What you'll find |
|------|-----------------|
| ℹ️ **About** | Methodology · Honest accuracy analysis · Acknowledgments · Citation |
| 📊 **Validation** | 45-event accuracy · 3-layer hard cases · threshold tuning · confusion matrix |
| 🌾 **CropDamage** | Boro rice impact · flood duration · upazila economic breakdown |
    """)

st.divider()

# Bengali community section
st.markdown("### 🇧🇩 বাংলা সতর্কতা নির্দেশিকা — সুনামগঞ্জ হাওর")
st.markdown(
    """
এই সিস্টেম স্যাটেলাইট ও আবহাওয়া তথ্য ব্যবহার করে সুনামগঞ্জ হাওরে **৭২ ঘণ্টার বন্যার পূর্বাভাস** দেয়।
উজানের বরাক নদীর প্রবাহ পর্যবেক্ষণ করে আগাম সতর্কতা প্রদান করে।
কৃষক, মৎসজীবী এবং স্থানীয় প্রশাসনের জন্য তৈরি।

| ঝুঁকির মাত্রা | শতকরা হার | অর্থ | করণীয় |
|---|---|---|---|
| 🔴 **অত্যন্ত বিপদজনক** | ৮৫%+ | তাৎক্ষণিক বন্যার সম্ভাবনা | এখনই নিরাপদ স্থানে যান, ফসল সরান |
| 🟠 **উচ্চ ঝুঁকি** | ৬৫%–৮৪% | বন্যার উচ্চ সম্ভাবনা | প্রস্তুত থাকুন, গবাদিপশু সরান |
| 🟡 **মাঝারি ঝুঁকি** | ৪০%–৬৪% | বন্যার কিছু সম্ভাবনা | সতর্ক থাকুন, জিনিসপত্র উঁচুতে রাখুন |
| 🟢 **স্বাভাবিক** | ৪০% এর কম | বন্যার সম্ভাবনা কম | নিয়মিত পরিস্থিতি পর্যবেক্ষণ করুন |

**মনে রাখুন:** উজানে বরাক নদীতে পানি বাড়লে প্রায় **৩৬ ঘণ্টার মধ্যে** হাওরে বন্যা আসতে পারে।
FFWC (বন্যা পূর্বাভাস ও সতর্কতা কেন্দ্র) এর তথ্য দিয়েও সবসময় যাচাই করুন।
    """
)

st.divider()

# Technical stack
st.markdown("### 🔧 Technical Stack")
tc1, tc2 = st.columns(2)
with tc1:
    st.markdown("""
**Satellite data:** Copernicus Sentinel-1 GRD (SAR) · Sentinel-2 SR (NDWI) via Google Earth Engine
**Rainfall / Forecast:** CHIRPS Daily · Open-Meteo hourly forecast & archive
**Soil / Climate:** ECMWF ERA5-Land soil moisture
**Terrain:** USGS SRTM DEM slope · WWF HydroSHEDS (TWI)
**Discharge:** GloFAS reanalysis via Open-Meteo Flood API (Barak + Surma)
    """)
with tc2:
    st.markdown("""
**ML models:** scikit-learn RandomForest (500 trees, w=0.45) + XGBoost (500 est., w=0.35) + PyTorch LSTM (w=0.20, synthetic training ⚠️ — excluded from primary accuracy metric)
**Features:** 15 collected · 13 ML inputs · 2 hydraulic dashboard indicators
**Primary validation:** LOOCV on 72 real-SAR events (2014–2024)
**Dashboard:** Streamlit · Folium · Plotly
**Alerts:** Gmail SMTP · Telegram Bot (tested) · WhatsApp API (tested)
**Language:** Python 3.11
    """)

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#999;font-size:12px;padding:8px 0'>"
    "© 2026 Salma Hoque Talukdar Koli &nbsp;·&nbsp; HaorFloodAlert v2.0 &nbsp;·&nbsp; "
    "RTM Al-Kabir Technical University &nbsp;·&nbsp; CSE Thesis Project &nbsp;·&nbsp; "
    "Sunamganj Haor, Bangladesh"
    "</div>",
    unsafe_allow_html=True,
)

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
