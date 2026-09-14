import streamlit as st

st.set_page_config(
    page_title="HaorFloodAlert",
    layout="wide",
    page_icon="🌊",
)

st.title("🌊 HaorFloodAlert")
st.subheader("Real-Time Flood Prediction & Community Alert System — Sunamganj Haor, Bangladesh")

st.markdown(
    """
    **Undergraduate Thesis Project** | RTM Al-Kabir Technical University
    Department of Computer Science & Engineering | Developer: Salma Hoque Talukdar Koli

    This system uses **13 satellite and meteorological features** — including Sentinel-1 SAR,
    MODIS NDWI, ERA5 soil moisture, CHIRPS rainfall, upstream Barak river proxy, and
    Open-Meteo 72h forecast — to predict flash flood risk in the haor wetlands of
    Sylhet-Sunamganj with an ensemble of Random Forest, XGBoost, and LSTM.

    **Model Performance (LOOCV on 77 real Sentinel-1 SAR events, 2014-2024):**
    - Accuracy: 89.6% | F1: 87.5% | AUC-ROC: 0.939
    - Labels from: FFWC Annual Reports, Mondal et al. (2021), Bhuiyan et al. (2024), World Bank GRADE (2024)
    - Evaluation: Leave-One-Out Cross-Validation (LOOCV) + 8x Gaussian augmentation

    ---

    | Page | Description |
    |------|-------------|
    | Prediction | Live 3-7 day flood risk forecast with 5-day rainfall animation |
    | Map | Real-time Sentinel-1 SAR water mask (Google Earth Engine) |
    | Alerts | SMS (BulkSMSBD) + Email (SMTP) + WhatsApp-ready template community alert system |
    | About | Project methodology, novel contributions, references |
    | Validation | Historical accuracy validation — 131-event inventory (77 real-SAR + 54 pre-Sentinel-1 proxy), 2009-2024 |
    | CropDamage | Boro rice crop damage + flood duration prediction |

    ---

    **Novel contributions:**
    - Upstream Barak river SAR proxy (India barrage release ~36h lead time)
    - MODIS NDWI for cloud-robust water detection in monsoon season
    - Pre-harvest flash flood detection (March-April) — first haor-specific model
    - 72-hour rainfall forecast for early warning
    - Boro rice crop damage estimation

    **Data sources:** Google Earth Engine | Sentinel-1 SAR | MODIS | ERA5-Land |
    Open-Meteo | USGS SRTM | FFWC/BWDB

    **Models:** RF (500 trees) + XGBoost (500 est.) + LSTM ensemble |
    10 active features | LOOCV evaluated
    """
)
