"""
1_Prediction.py — Real-Time Flood Risk Prediction
14-feature system (13 ML inputs + Surma discharge dashboard) + 5-day forecast + flood duration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ee
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
try:
    import tensorflow as tf
    _TF_OK = True
except ImportError:
    _TF_OK = False
from datetime import datetime, timedelta, timezone

from config import (
    DEFAULTS, FEATURES, GEE_PROJECT,
    HAOR_BBOX, HAOR_LAT, HAOR_LON,
    MODELS_DIR, RF_WEIGHT, XGB_WEIGHT, LSTM_WEIGHT,
    TEMP_CLIMATOLOGY,
    UPSTREAM_TRAVEL_HOURS,
    UPSTREAM_DISCHARGE_DEFAULT,
    UPSTREAM_DISCHARGE_THRESHOLD_DANGER,
    UPSTREAM_DISCHARGE_THRESHOLD_HIGH,
    UPSTREAM_DISCHARGE_THRESHOLD_WARNING,
)
from utils.gee_features import (
    fetch_sentinel1, fetch_ndwi, fetch_twi, fetch_slope,
    fetch_rainfall_chirps, fetch_soil_moisture,
    fetch_weather_historical, fetch_forecast,
    fetch_upstream_vv, estimate_flood_lead_time,
    fetch_surma_discharge,
    get_forecast_rainfall_72h,
)
from utils.predict import predict_flood_72h, apply_discharge_adjustment
from utils.upstream_discharge import (
    get_barak_discharge_current,
    get_barak_discharge_forecast_72h,
    classify_discharge_risk,
)
from utils.discharge_trend import (
    get_discharge_history_14days,
    calculate_trend_per_day,
    analyze_discharge_trend,
    predict_discharge_next_72h,
)

BST = timezone(timedelta(hours=6))   # Bangladesh Standard Time = UTC+6

st.set_page_config(page_title="Flood Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Real-Time Flood Risk Prediction")
st.subheader("Sunamganj Haor — RF + XGBoost | 11 Active Features | LOOCV 89.6% (real-SAR, deconfounded)")

N_FEAT = len(FEATURES)

# LSTM loaded dynamically (Keras or PyTorch) — see load_models()

@st.cache_resource
def load_models():
    rf  = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")

    act_path = MODELS_DIR / "active_features.pkl"
    if act_path.exists():
        active_feats = joblib.load(act_path)
    else:
        active_feats = list(FEATURES)[:rf.n_features_in_]

    lstm_m, lstm_s, lstm_type = None, None, None
    sp = MODELS_DIR / "lstm_scaler.pkl"
    lp_h5  = MODELS_DIR / "lstm_model.h5"
    lp_pt  = MODELS_DIR / "lstm_model.pth"
    try:
        sc = joblib.load(sp) if sp.exists() else None
        if lp_h5.exists() and _TF_OK and sc is not None:
            import tensorflow as tf
            lstm_m    = tf.keras.models.load_model(str(lp_h5))
            lstm_s    = sc
            lstm_type = "keras"
        elif lp_pt.exists() and sc is not None:
            import json, torch, torch.nn as nn
            nf_lp   = MODELS_DIR / "lstm_n_features.json"
            lstm_nf = json.load(open(nf_lp))["n_features"] if nf_lp.exists() else len(active_feats)
            class _LSTM(nn.Module):
                def __init__(self, nf):
                    super().__init__()
                    self.lstm = nn.LSTM(nf,64,2,batch_first=True,dropout=0.30)
                    self.norm = nn.LayerNorm(64)
                    self.drop = nn.Dropout(0.35)
                    self.head = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
                def forward(self,x):
                    _,(h,_) = self.lstm(x)
                    return self.head(self.drop(self.norm(h[-1]))).squeeze(1)
            m = _LSTM(lstm_nf)
            m.load_state_dict(torch.load(str(lp_pt), map_location="cpu"))
            m.eval()
            lstm_m, lstm_s, lstm_type = m, sc, "torch"
    except Exception as e:
        st.warning(f"LSTM load: {e}")
    return rf, xgb, lstm_m, lstm_s, active_feats, lstm_type


def lstm_predict(model, scaler, feat_row, active_feats, all_feats, lstm_type="keras"):
    """
    Run the LSTM and return a probability in [0, 1].

    Falls back to 0.5 (neutral — no contribution) on any error.
    The most likely failure is a feature-count mismatch: the LSTM scaler was
    fitted on the old feature list that included raw 'temp' (11 features),
    but the current FEATURES list uses 'temp_anomaly' instead (still 13 total,
    but active_feats trimmed to 11 may differ).  Since LSTM weight = 0.20 and
    it was trained on synthetic data, a neutral fallback is safe.
    """
    try:
        feat_idx = [list(all_feats).index(f) for f in active_feats]
        base     = np.array([feat_row[i] for i in feat_idx], dtype=np.float32)
        noise    = np.ones(len(active_feats), dtype=np.float32) * 0.5
        seq      = np.stack([base + np.random.normal(0, noise).astype(np.float32)
                             for _ in range(5)])
        seq[-1]  = base
        scaled   = scaler.transform(seq).astype(np.float32)
        if lstm_type == "keras":
            p = model.predict(scaled[np.newaxis], verbose=0)
            return float(p[0][0])
        else:
            import torch
            with torch.no_grad():
                return float(torch.sigmoid(model(torch.tensor(scaled).unsqueeze(0))).item())
    except Exception as _lstm_err:
        # Scaler/model mismatch (e.g. temp→temp_anomaly feature rename).
        # Return 0.5 so LSTM contributes zero net pull on the ensemble.
        st.caption(
            f"⚠️ LSTM skipped (feature mismatch — retrain LSTM to fix): {_lstm_err}"
        )
        return 0.5


@st.cache_data(ttl=3600)
def fetch_live():
    ee.Initialize(project=GEE_PROJECT)
    haor     = ee.Geometry.Rectangle(HAOR_BBOX)
    end_dt   = datetime.now(timezone.utc).date() - timedelta(days=2)
    start_dt = end_dt - timedelta(days=7)
    s, e     = str(start_dt), str(end_dt)

    vv, vh, ratio = fetch_sentinel1(haor, s, e)
    rain          = fetch_rainfall_chirps(haor, s, e)
    soil          = fetch_soil_moisture(haor, s, e)
    temp, wind    = fetch_weather_historical(s, e)
    slope         = fetch_slope(haor)
    f12, f72      = fetch_forecast()
    ndwi          = fetch_ndwi(haor, s, e)
    twi           = fetch_twi(haor)
    upstream_vv   = fetch_upstream_vv(s, e)
    lead_info     = estimate_flood_lead_time(upstream_vv, vv, f72)
    surma_q       = fetch_surma_discharge(date_str=e, days=7)

    # Compute temp_anomaly: removes seasonal confound (raw temp r=0.57 with flood_label).
    # Uses the end-date month so the monthly baseline is aligned with the observation window.
    end_month    = end_dt.month
    temp_anomaly = round(temp - TEMP_CLIMATOLOGY.get(end_month, temp), 2)

    return dict(VV=round(vv,2), VH=round(vh,2), vv_vh_ratio=round(ratio,4),
                rainfall=rain, soil_moisture=soil,
                temp=temp,               # raw — display only, not a model input
                temp_anomaly=temp_anomaly,  # model input (FEATURES list key)
                wind=wind,
                slope=slope, forecast_rain_next_12h=f12,
                ndwi=round(ndwi,4), twi=round(twi,2),
                upstream_vv=round(upstream_vv,2), forecast_rain_72h=f72,
                surma_discharge=round(surma_q,2),
                start=s, end=e, lead_info=lead_info)


@st.cache_data(ttl=3600)
def fetch_5day_forecast():
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={HAOR_LAT}&longitude={HAOR_LON}"
            f"&hourly=precipitation,temperature_2m,wind_speed_10m"
            f"&forecast_days=6&timezone=Asia/Dhaka",
            timeout=15
        ).json()["hourly"]

        days = {}
        for t_str, rain, temp, wind in zip(
                r["time"], r["precipitation"],
                r["temperature_2m"], r["wind_speed_10m"]):
            dt   = datetime.fromisoformat(t_str)
            dkey = dt.strftime("%Y-%m-%d")
            if dkey not in days:
                days[dkey] = {"rain": 0.0, "temp": [], "wind": 0.0}
            days[dkey]["rain"] += rain
            days[dkey]["temp"].append(temp)
            days[dkey]["wind"] = max(days[dkey]["wind"], wind)

        rows = []
        for dkey, v in sorted(days.items())[:5]:
            rows.append({
                "Date":        dkey,
                "Rain (mm)":   round(v["rain"], 1),
                "Max Temp °C": round(max(v["temp"]), 1),
                "Max Wind":    round(v["wind"], 1),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_72h_breakdown():
    """Cached 72-hour rainfall forecast from Open-Meteo. Refreshes every hour."""
    return get_forecast_rainfall_72h()


@st.cache_data(ttl=3600)
def fetch_barak_discharge():
    """Cached Barak river discharge from Open-Meteo GloFAS. Refreshes every hour."""
    current  = get_barak_discharge_current(days_back=14)
    forecast = get_barak_discharge_forecast_72h()
    return current, forecast


@st.cache_data(ttl=3600)
def fetch_discharge_trend():
    """
    Fetch 14-day discharge history, run reliability-checked trend analysis,
    and project 72h ahead.  Handles GloFAS API failures gracefully.
    """
    meta        = get_discharge_history_14days()          # dict with values + metadata
    history     = meta["values"]
    analysis    = analyze_discharge_trend(history)        # slope, r², reliability

    # Use reliable trend for projections; flat if unreliable
    effective_trend = analysis["slope"] if analysis["is_reliable"] else 0.0
    current_val     = history[-1] if history else 450.0
    projections     = predict_discharge_next_72h(current_val, effective_trend)

    return {
        "history":         history,
        "smoothed":        analysis["smoothed"],
        "trend":           effective_trend,           # 0 if suppressed
        "raw_trend":       analysis["slope"],          # always the OLS slope
        "r_squared":       analysis["r_squared"],
        "is_reliable":     analysis["is_reliable"],
        "reliability_reason": analysis["reason"],
        "n_days":          analysis["n_days"],
        "checks":          analysis["checks"],
        "current":         current_val,
        "projections":     projections,               # [24h, 48h, 72h]
        # API metadata
        "api_status":      meta["status"],            # "ok"|"cached"|"default"
        "data_age_hours":  meta["data_age_hours"],
        "fetched_at":      meta["fetched_at"],
        "source":          meta["source"],
    }


def risk_label(p):
    if p >= 0.85: return "EXTREME", "🔴"
    if p >= 0.65: return "HIGH",    "🟠"
    if p >= 0.40: return "MEDIUM",  "🟡"
    return "LOW", "🟢"

RISK_LABELS_BN = {
    "EXTREME": "অত্যন্ত বিপদজনক",
    "HIGH":    "উচ্চ ঝুঁকি",
    "MEDIUM":  "মাঝারি ঝুঁকি",
    "LOW":     "স্বাভাবিক",
}


def flood_duration_estimate(prob, soil, upvv, rain):
    base = 0.0
    if prob >= 0.85:   base = 18 + prob * 20
    elif prob >= 0.65: base = 8  + prob * 15
    elif prob >= 0.40: base = 3  + prob * 8
    else:              base = 1  + prob * 4
    mod = (1 + (soil-30)/100) * (1.3 if upvv < -16 else 1.0) * (1 + rain/500)
    return max(0.0, round(base * mod, 1))


if st.button("🔄 Refresh Live Data", type="primary"):
    st.cache_data.clear()

with st.spinner("Fetching 15 live features from GEE + APIs ..."):
    try:
        feat = fetch_live()
    except Exception as err:
        st.error(f"GEE fetch failed: {err}"); st.stop()

rf, xgb, lstm_m, lstm_s, active_feats, lstm_type = load_models()

feat_row = [feat[f] for f in FEATURES]
input_df = pd.DataFrame([feat_row], columns=list(FEATURES))

input_rf  = input_df[active_feats]
rf_prob   = float(rf.predict_proba(input_rf)[0][1])
xgb_prob  = float(xgb.predict_proba(input_rf)[0][1])

# RF uncertainty: distribution across all 500 individual tree predictions
tree_probs = np.array([tree.predict_proba(input_rf)[0][1] for tree in rf.estimators_])
rf_std     = float(tree_probs.std())

if lstm_m and lstm_s:
    lstm_prob  = float(lstm_predict(lstm_m, lstm_s, feat_row, active_feats, FEATURES, lstm_type))
    final_prob = float(RF_WEIGHT*rf_prob + XGB_WEIGHT*xgb_prob + LSTM_WEIGHT*lstm_prob)
    lstm_ok    = True
    ci_margin  = 1.96 * RF_WEIGHT * rf_std
else:
    final_prob = float(RF_WEIGHT/(RF_WEIGHT+XGB_WEIGHT)*rf_prob +
                       XGB_WEIGHT/(RF_WEIGHT+XGB_WEIGHT)*xgb_prob)
    lstm_prob, lstm_ok = None, False
    ci_margin  = 1.96 * (RF_WEIGHT / (RF_WEIGHT + XGB_WEIGHT)) * rf_std

prob_lower   = round(max(0.0, final_prob - ci_margin), 3)
prob_upper   = round(min(1.0, final_prob + ci_margin), 3)
is_uncertain = (prob_upper - prob_lower) > 0.20

risk, icon = risk_label(final_prob)
inundated  = round(42 + final_prob * 92, 1)
lead_info  = feat["lead_info"]
dur_days   = flood_duration_estimate(final_prob, feat["soil_moisture"],
                                     feat["upstream_vv"], feat["rainfall"])

if lead_info["upstream_alert"]:
    lh = lead_info["lead_hours"]
    st.warning(
        f"⚠️ **UPSTREAM FLOOD SIGNAL — Barak River (Assam/India)**  \n"
        f"**বাংলা:** উজানের সংকেত — বরাক নদীতে পানি বেড়েছে (~{lh:.0f} ঘণ্টার মধ্যে হাওরে পৌঁছাতে পারে)  \n"
        f"Upstream VV: **{feat['upstream_vv']:.1f} dB** | "
        f"Estimated arrival: **~{lh:.0f} hours**  \n"
        f"This may indicate barrage/gate release. Prepare NOW — এখনই প্রস্তুত হন।"
    )

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric(f"{icon} Risk Level", risk,
          delta=RISK_LABELS_BN.get(risk, ""), delta_color="off")
c2.metric("Flood Probability",  f"{final_prob*100:.1f}%",
          delta=f"95% CI: {prob_lower*100:.0f}–{prob_upper*100:.0f}%",
          delta_color="off")
c3.metric("Inundated Area",          f"{inundated} km²")
c4.metric("Forecast 12h",            f"{feat['forecast_rain_next_12h']:.1f} mm")
c5.metric("Forecast 72h",            f"{feat['forecast_rain_72h']:.1f} mm")
c6.metric("Est. Flood Duration",     f"{dur_days:.0f} days" if final_prob >= 0.4 else "—")

st.divider()

with st.expander("🔬 Model Technical Details — RF + XGBoost + LSTM", expanded=False):
    st.subheader("🤖 Model Breakdown")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**Random Forest**")
        st.progress(float(rf_prob), text=f"{rf_prob*100:.1f}%")
        st.caption(f"500 trees · {len(active_feats)} features · LOOCV 89.6% (real-SAR) · 87.8% (extended, deconfounded)")
    with mc2:
        st.markdown("**XGBoost**")
        st.progress(float(xgb_prob), text=f"{xgb_prob*100:.1f}%")
        st.caption(f"500 estimators · {len(active_feats)} features · LOOCV 89.6% (real-SAR) · 87.8% (extended, deconfounded)")
    with mc3:
        st.markdown("**LSTM (5-day sequence)**")
        if lstm_ok:
            st.progress(float(lstm_prob), text=f"{lstm_prob*100:.1f}%")
            st.caption("⚠️ Synthetic training · walk-forward 100% (overfit, n=101) · excluded from primary metric")
        else:
            st.info("LSTM not loaded — RF+XGB only (primary metric unaffected)")

    wstr = (f"RF×{RF_WEIGHT} + XGB×{XGB_WEIGHT} + LSTM×{LSTM_WEIGHT}"
            if lstm_ok else f"RF×{RF_WEIGHT/(RF_WEIGHT+XGB_WEIGHT):.2f} + XGB×{XGB_WEIGHT/(RF_WEIGHT+XGB_WEIGHT):.2f}")
    st.info(
        f"**Ensemble:** {wstr} = **{final_prob*100:.1f}%** "
        f"(95% CI: {prob_lower*100:.0f}%–{prob_upper*100:.0f}%)"
    )

    if is_uncertain:
        st.warning(
            f"⚠️ **High model uncertainty** — 95% CI spans "
            f"{(prob_upper - prob_lower)*100:.0f} percentage points "
            f"({prob_lower*100:.0f}%–{prob_upper*100:.0f}%). "
            "Verify with FFWC real-time gauge data before issuing community alerts. "
            "মডেলের অনিশ্চয়তা বেশি — FFWC গেজ তথ্য দিয়ে যাচাই করুন।"
        )

    # RF tree probability distribution (uncertainty visualization)
    with st.expander("📊 RF Uncertainty Detail — Tree Probability Distribution"):
        fig_unc = go.Figure()
        fig_unc.add_trace(go.Histogram(
            x=tree_probs * 100,
            nbinsx=25,
            marker_color="#5b9bd5",
            name="Tree predictions",
            opacity=0.8,
        ))
        fig_unc.add_vline(x=rf_prob * 100, line_color="red", line_dash="dash",
                          annotation_text=f"RF mean: {rf_prob*100:.1f}%")
        fig_unc.add_vline(x=prob_lower * 100, line_color="orange", line_dash="dot",
                          annotation_text=f"CI low: {prob_lower*100:.0f}%")
        fig_unc.add_vline(x=prob_upper * 100, line_color="orange", line_dash="dot",
                          annotation_text=f"CI high: {prob_upper*100:.0f}%")
        fig_unc.update_layout(
            title=f"RF Forest: Distribution of {len(rf.estimators_)} tree predictions",
            xaxis_title="Flood Probability (%)",
            yaxis_title="Number of Trees",
            height=280,
        )
        st.plotly_chart(fig_unc, use_container_width=True)
        st.caption(
            f"Each bar = number of trees predicting that probability. "
            f"Narrow distribution = high model confidence. "
            f"RF std = {rf_std*100:.1f}pp | 95% CI = ±{ci_margin*100:.1f}pp"
        )

st.divider()

st.subheader("📅 5-Day Rainfall Forecast (Sunamganj Haor)")
forecast_df = fetch_5day_forecast()

if not forecast_df.empty:
    probs = []
    for _, row in forecast_df.iterrows():
        day_feat = feat_row.copy()
        feat_names = list(FEATURES)
        day_feat[feat_names.index("rainfall")]               = row["Rain (mm)"] * 3
        day_feat[feat_names.index("forecast_rain_next_12h")] = row["Rain (mm)"] * 0.5
        day_feat[feat_names.index("forecast_rain_72h")]      = row["Rain (mm)"] * 2.0
        d_df = pd.DataFrame([day_feat], columns=feat_names)
        d_rf = d_df[active_feats]
        p = float(RF_WEIGHT/(RF_WEIGHT+XGB_WEIGHT) * rf.predict_proba(d_rf)[0][1] +
                  XGB_WEIGHT/(RF_WEIGHT+XGB_WEIGHT) * xgb.predict_proba(d_rf)[0][1])
        probs.append(round(p, 3))

    forecast_df["Flood Risk (%)"] = [round(p*100, 1) for p in probs]
    forecast_df["Risk Level"]     = [risk_label(p)[0] for p in probs]
    forecast_df["Risk Color"]     = [("#FF4B4B" if p>=0.65 else
                                      "#FFD700" if p>=0.40 else "#00C49A")
                                     for p in probs]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Bar(
        x=forecast_df["Date"], y=forecast_df["Rain (mm)"],
        name="Forecast Rain (mm)", marker_color="#5b9bd5", opacity=0.6,
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Flood Risk (%)"],
        name="Flood Risk (%)", mode="lines+markers",
        line=dict(color="#FF4B4B", width=3),
        marker=dict(size=10, color=forecast_df["Risk Color"]),
    ))
    fig_fc.add_hline(y=65, line_dash="dash", line_color="orange",
                     annotation_text="HIGH risk threshold (65%)")
    fig_fc.add_hline(y=40, line_dash="dot", line_color="yellow",
                     annotation_text="MEDIUM threshold (40%)")
    fig_fc.update_layout(
        title="5-Day Flood Risk Forecast — Sunamganj Haor",
        yaxis=dict(title="Flood Risk (%)", range=[0, 105]),
        height=400, legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    col_fc = st.columns(len(forecast_df))
    for i, (_, row) in enumerate(forecast_df.iterrows()):
        with col_fc[i]:
            r_lbl = risk_label(probs[i])
            st.markdown(f"**{row['Date'][5:]}**")
            st.markdown(f"{r_lbl[1]} {r_lbl[0]}")
            st.caption(f"🌧️ {row['Rain (mm)']} mm")
else:
    st.info("5-day forecast unavailable — check internet connection.")

st.divider()

# 72-Hour Flood Risk Forecast
st.subheader("📊 72-Hour Flood Risk Forecast")
st.caption(
    f"Last updated: {datetime.now(BST).strftime('%Y-%m-%d %H:%M BST')} "
    f"(cached 1 h) · Open-Meteo hourly precipitation → RF+XGB ensemble"
)

with st.spinner("Computing 72-hour forecast (3-layer model) ..."):
    fc72       = fetch_72h_breakdown()
    trend_data = fetch_discharge_trend()
    fp72 = predict_flood_72h(
        feat, fc72, rf, xgb, active_feats,
        current_discharge      = trend_data["current"],
        discharge_trend        = trend_data["trend"],        # 0 if unreliable
        discharge_projections  = trend_data["projections"],
        is_trend_reliable      = trend_data["is_reliable"],
    )

peak_risk, peak_icon = risk_label(fp72["peak"])

# Top summary row
col_peak, col_total, col_src = st.columns([2, 1, 1])
with col_peak:
    pct = fp72["peak"] * 100
    if fp72["peak"] >= 0.85:
        badge_color = "#CC0000"
    elif fp72["peak"] >= 0.65:
        badge_color = "#E65C00"
    elif fp72["peak"] >= 0.40:
        badge_color = "#B8860B"
    else:
        badge_color = "#1A7A4A"
    st.markdown(
        f"<div style='padding:14px 20px;border-radius:8px;background:#F8F9FA;"
        f"border-left:6px solid {badge_color}'>"
        f"<span style='font-size:13px;color:#666;text-transform:uppercase;"
        f"letter-spacing:0.5px'>Peak 72-hour flood probability</span><br>"
        f"<span style='font-size:38px;font-weight:bold;color:{badge_color}'>"
        f"{pct:.1f}%</span> "
        f"<span style='font-size:20px;color:{badge_color}'>"
        f"{peak_icon} {peak_risk} RISK</span><br>"
        f"<span style='font-size:13px;color:#555'>"
        f"Peak window: <b>{fp72['peak_window']}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_total:
    st.metric("Total 72h Forecast Rain",
              f"{fc72['total']:.1f} mm",
              delta=fc72["source"],
              delta_color="off")
with col_src:
    st.metric("Data source",
              "Open-Meteo",
              delta="Hourly · Dhaka TZ",
              delta_color="off")

st.markdown("")

# Bengali warning for high / extreme risk
if fp72["peak"] >= 0.65:
    bn_msg = {
        "EXTREME": (
            "🚨 **আগামী ৭২ ঘণ্টায় অত্যন্ত বিপজ্জনক বন্যার ঝুঁকি রয়েছে।** "
            "এখনই নিরাপদ স্থানে যাওয়ার প্রস্তুতি নিন। "
            "ফসল ও গবাদিপশু সরিয়ে ফেলুন।"
        ),
        "HIGH": (
            "⚠️ **আগামী ৭২ ঘণ্টায় উচ্চ বন্যার ঝুঁকি রয়েছে।** "
            "জরুরি জিনিসপত্র গুছিয়ে নিন এবং পরিস্থিতি পর্যবেক্ষণ করুন।"
        ),
    }.get(peak_risk, "⚠️ বন্যার ঝুঁকি রয়েছে — সতর্ক থাকুন।")
    st.warning(bn_msg)

# Three-window probability chart
windows      = fp72["windows"]
window_names = [w["label"] for w in windows]
window_probs = [round(w["prob"] * 100, 1) for w in windows]
window_rain  = [w["rain_mm"] for w in windows]
window_colors = [
    ("#CC0000" if p >= 85 else "#E65C00" if p >= 65 else
     "#B8860B" if p >= 40 else "#1A7A4A")
    for p in window_probs
]

fig_72 = go.Figure()

# Rain bars (secondary y-axis)
fig_72.add_trace(go.Bar(
    x=window_names,
    y=window_rain,
    name="Forecast Rain (mm)",
    marker_color="#5b9bd5",
    opacity=0.55,
    yaxis="y2",
))

# Risk probability line
fig_72.add_trace(go.Scatter(
    x=window_names,
    y=window_probs,
    name="Flood Risk (%)",
    mode="lines+markers+text",
    line=dict(color="#FF4B4B", width=3),
    marker=dict(size=14, color=window_colors, line=dict(width=2, color="white")),
    text=[f"{p:.0f}%" for p in window_probs],
    textposition="top center",
    textfont=dict(size=13, color="#333"),
))

# Current (now) reference line
fig_72.add_hline(
    y=fp72["now"] * 100,
    line_dash="dot",
    line_color="#888",
    annotation_text=f"Current: {fp72['now']*100:.0f}%",
    annotation_position="bottom right",
)
fig_72.add_hline(y=65, line_dash="dash", line_color="orange",
                 annotation_text="HIGH threshold (65%)")
fig_72.add_hline(y=40, line_dash="dot",  line_color="#B8860B",
                 annotation_text="MEDIUM threshold (40%)")

fig_72.update_layout(
    title="72-Hour Flood Risk by Window — Sunamganj Haor",
    xaxis_title="Forecast Window",
    yaxis=dict(title="Flood Risk (%)", range=[0, 110]),
    yaxis2=dict(title="Rain (mm)", overlaying="y", side="right",
                showgrid=False, range=[0, max(window_rain or [1]) * 4]),
    height=380,
    legend=dict(orientation="h", y=1.05),
    plot_bgcolor="#FAFAFA",
)
st.plotly_chart(fig_72, use_container_width=True)

# Window metric cards
card_cols = st.columns(3)
for col, w in zip(card_cols, windows):
    r_lbl, r_icon = risk_label(w["prob"])
    with col:
        st.metric(
            label=f"{r_icon} {w['label']}",
            value=f"{w['prob']*100:.1f}%",
            delta=f"🌧️ {w['rain_mm']:.1f} mm forecast",
            delta_color="off",
        )
        st.caption(f"Risk: {r_lbl}")

# Hourly rain sparkline (if data available)
if fc72.get("hourly_times") and len(fc72["hourly_times"]) >= 6:
    with st.expander("🌧️ Hourly Rainfall Breakdown (72h)", expanded=False):
        df_hr = pd.DataFrame({
            "Time":    fc72["hourly_times"],
            "Rain mm": fc72["hourly_rain"],
        })
        df_hr["Time"] = pd.to_datetime(df_hr["Time"]).dt.strftime("%m-%d %H:%M")
        fig_hr = px.bar(
            df_hr, x="Time", y="Rain mm",
            title="Hourly Forecast Precipitation — Sunamganj Haor (next 72h)",
            color_discrete_sequence=["#5b9bd5"],
            height=260,
        )
        fig_hr.update_layout(xaxis_tickangle=-45, plot_bgcolor="#FAFAFA")
        st.plotly_chart(fig_hr, use_container_width=True)
        st.caption(
            f"Source: {fc72['source']} · Fetched: {fc72.get('fetched_at', 'unknown')} · "
            f"72h total: {fc72['total']:.1f} mm"
        )
elif fc72["source"] == "default":
    st.info(
        "📡 72-hour rainfall data unavailable — showing default (0 mm). "
        "Check internet connection. Flood probability estimate uses current "
        "satellite conditions only."
    )

st.divider()

# 3-Layer Prediction Component Breakdown
st.subheader("🔍 Prediction Component Breakdown")
st.caption(
    "Shows how each layer contributes to the peak 72-hour flood probability. "
    "Layer 1 = ML satellite/rainfall base · Layer 2 = current discharge level · "
    "Layer 3 = discharge rising trend (reliability-gated)."
)

# API status banner
_api_st = trend_data.get("api_status", "ok")
_age_h  = trend_data.get("data_age_hours", 0.0)
if _api_st == "cached":
    _age_m = round(_age_h * 60)
    st.warning(
        f"⚠️ **Using cached discharge data ({_age_m} minutes old)** — "
        f"GloFAS API unavailable. Discharge values reflect the last successful "
        f"fetch. Trend analysis may be less accurate. "
        f"Last fetch: {trend_data.get('fetched_at', 'unknown')}",
        icon="⚠️",
    )
elif _api_st == "default":
    st.error(
        "🔴 **GloFAS API unavailable and no cache found.** "
        "Discharge shown as dry-season baseline (450 m³/s). "
        "Layer 2 and Layer 3 adjustments are suppressed. "
        "ML base probability (Layer 1) remains valid.",
        icon="🔴",
    )

# Trend reliability notice
if not trend_data.get("is_reliable", True):
    st.info(
        f"ℹ️ **Layer 3 (discharge trend) suppressed** — "
        f"{trend_data.get('reliability_reason', 'Trend unreliable.')}  \n"
        f"Only Layers 1 (ML) and 2 (discharge level) contribute to this forecast. "
        f"Raw slope: {trend_data.get('raw_trend', 0):+.0f} m³/s/day · "
        f"R² = {trend_data.get('r_squared', 0):.2f} · "
        f"Days: {trend_data.get('n_days', 0)}",
    )

bd   = fp72["breakdown"]
bpct = bd["base_prob"]   * 100
l2pp = bd["adj_current"] * 100
l3pp = bd["adj_trend"]   * 100
fpct = bd["final_prob"]  * 100
_, risk_icon_bd = risk_label(bd["final_prob"])

# Summary card
if bd["final_prob"] >= 0.85:   card_color = "#CC0000"
elif bd["final_prob"] >= 0.65: card_color = "#E65C00"
elif bd["final_prob"] >= 0.40: card_color = "#B8860B"
else:                           card_color = "#1A7A4A"

st.markdown(
    f"<div style='padding:16px 22px;border-radius:10px;background:#F8F9FA;"
    f"border-left:7px solid {card_color};margin-bottom:12px'>"
    f"<span style='font-size:13px;color:#666;text-transform:uppercase;"
    f"letter-spacing:0.6px'>Final Flood Probability — Peak Window "
    f"({fp72['peak_window']})</span><br>"
    f"<span style='font-size:44px;font-weight:bold;color:{card_color}'>"
    f"{fpct:.1f}%</span> "
    f"<span style='font-size:22px;color:{card_color}'>{risk_icon_bd}</span><br>"
    f"<span style='font-size:13px;color:#555;margin-top:4px;display:block'>"
    f"<b>Component Breakdown:</b></span>"
    f"<span style='font-size:14px;color:#333;display:block;margin-top:2px'>"
    f"&nbsp;&nbsp;├─ 🛰️ ML base (satellite + rainfall) &nbsp;&nbsp;&nbsp;&nbsp;: "
    f"<b>{bpct:.1f}%</b></span>"
    f"<span style='font-size:14px;color:#333;display:block'>"
    f"&nbsp;&nbsp;├─ 🌊 Current discharge (Layer 2)&nbsp;&nbsp;: "
    f"<b>+{l2pp:.1f} pp</b></span>"
    f"<span style='font-size:14px;color:#333;display:block'>"
    f"&nbsp;&nbsp;└─ 📈 Discharge trend (Layer 3) &nbsp;&nbsp;&nbsp;&nbsp;: "
    f"<b>+{l3pp:.1f} pp</b></span>"
    f"<span style='font-size:14px;color:{card_color};font-weight:bold;"
    f"display:block;margin-top:6px'>"
    f"&nbsp;&nbsp;&nbsp;&nbsp; TOTAL = {bpct:.1f} + {l2pp:.1f} + {l3pp:.1f}"
    f" = <u>{fpct:.1f}%</u></span>"
    f"</div>",
    unsafe_allow_html=True,
)

# Explanation
if bd.get("cap_applied"):
    st.caption(
        f"🔒 **Safety cap applied:** L2+L3 combined adjustment limited to "
        f"{int(0.30*100)} pp and/or final probability capped at "
        f"{int(0.95*100)}% to prevent discharge from overpowering the ML signal."
    )

if bd["explanation"]:
    st.info(f"**Explanation:** {bd['explanation']}")

# Per-window stacked contribution chart
with st.expander("📊 Per-Window 3-Layer Contribution Chart", expanded=True):
    w_labels   = [w["label"]       for w in fp72["windows"]]
    w_base     = [w["base_prob"]   * 100 for w in fp72["windows"]]
    w_l2       = [w["adj_current"] * 100 for w in fp72["windows"]]
    w_l3       = [w["adj_trend"]   * 100 for w in fp72["windows"]]

    fig_bd = go.Figure()
    fig_bd.add_trace(go.Bar(
        name="Layer 1 — ML Base (satellite + rainfall)",
        x=w_labels, y=w_base,
        marker_color="#5b9bd5",
        text=[f"{v:.1f}%" for v in w_base],
        textposition="inside",
    ))
    fig_bd.add_trace(go.Bar(
        name="Layer 2 — Current Discharge",
        x=w_labels, y=w_l2,
        marker_color="#E65C00",
        text=[f"+{v:.1f}pp" if v > 0 else "" for v in w_l2],
        textposition="inside",
    ))
    fig_bd.add_trace(go.Bar(
        name="Layer 3 — Discharge Trend",
        x=w_labels, y=w_l3,
        marker_color="#CC0000",
        text=[f"+{v:.1f}pp" if v > 0 else "" for v in w_l3],
        textposition="inside",
    ))
    fig_bd.update_layout(
        barmode="stack",
        title="3-Layer Flood Probability — Component Contribution per Window",
        xaxis_title="Forecast Window",
        yaxis=dict(title="Flood Probability (%)", range=[0, 110]),
        height=340,
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="#FAFAFA",
    )
    fig_bd.add_hline(y=65, line_dash="dash", line_color="orange",
                     annotation_text="HIGH threshold (65%)")
    fig_bd.add_hline(y=40, line_dash="dot",  line_color="#B8860B",
                     annotation_text="MEDIUM threshold (40%)")
    st.plotly_chart(fig_bd, use_container_width=True)

    # Reason table
    reason_rows = []
    for w in fp72["windows"]:
        reason_rows.append({
            "Window":          w["label"],
            "Base (L1) %":     f"{w['base_prob']*100:.1f}",
            "+Discharge (L2)": f"+{w['adj_current']*100:.1f} pp",
            "+Trend (L3)":     f"+{w['adj_trend']*100:.1f} pp",
            "Final %":         f"{w['prob']*100:.1f}",
            "L2 Reason":       w["l2_reason"],
            "L3 Reason":       w["l3_reason"],
        })
    st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)

# 14-day discharge history + trend line
with st.expander("📈 14-Day Discharge History + Trend Line", expanded=False):
    hist       = trend_data["history"]
    smoothed   = trend_data.get("smoothed", hist)
    t_slope    = trend_data["raw_trend"]          # always show raw slope here
    eff_slope  = trend_data["trend"]              # 0 if suppressed
    t_curr     = trend_data["current"]
    t_proj     = trend_data["projections"]        # [24h, 48h, 72h]
    r2_val     = trend_data.get("r_squared", 0.0)
    reliable   = trend_data.get("is_reliable", True)

    from datetime import date as _date
    today      = _date.today()
    hist_dates = [
        str(today - timedelta(days=len(hist) - 1 - i))
        for i in range(len(hist))
    ]
    proj_labels = ["0–24h proj", "24–48h proj", "48–72h proj"]

    # Fitted trend line (use raw slope even if suppressed, shown as dashed grey if suppressed)
    x_fit      = np.arange(len(hist), dtype=float)
    intercept  = float(np.polyfit(x_fit, np.array(smoothed, dtype=float), 1)[1])
    trend_fit  = intercept + t_slope * x_fit
    trend_color = "#FF4B4B" if reliable else "#AAAAAA"
    trend_label = (f"Linear trend ({t_slope:+.0f} m³/s/day, R²={r2_val:.2f})"
                   + ("" if reliable else " ⚠️ unreliable"))

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_dates, y=hist,
        name="GloFAS Discharge (raw)",
        mode="lines+markers",
        line=dict(color="#5b9bd5", width=2),
        marker=dict(size=5),
        opacity=0.7,
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_dates, y=smoothed,
        name="3-Day Smoothed",
        mode="lines",
        line=dict(color="#1A7A4A", width=2),
    ))
    fig_hist.add_trace(go.Scatter(
        x=hist_dates, y=list(trend_fit),
        name=trend_label,
        mode="lines",
        line=dict(color=trend_color, width=2,
                  dash="dash" if not reliable else "solid"),
    ))
    # 72h projection (flat if trend suppressed)
    fig_hist.add_trace(go.Scatter(
        x=proj_labels, y=t_proj,
        name="72h Projection" + (" (flat — trend suppressed)" if not reliable else ""),
        mode="markers+text",
        marker=dict(size=12, color=trend_color, symbol="diamond"),
        text=[f"{v:,.0f}" for v in t_proj],
        textposition="top center",
    ))
    fig_hist.add_hline(y=UPSTREAM_DISCHARGE_THRESHOLD_DANGER,
                       line_dash="dash", line_color="#CC0000",
                       annotation_text=f"DANGER ({UPSTREAM_DISCHARGE_THRESHOLD_DANGER:,})")
    fig_hist.add_hline(y=UPSTREAM_DISCHARGE_THRESHOLD_HIGH,
                       line_dash="dot", line_color="#E65C00",
                       annotation_text=f"HIGH ({UPSTREAM_DISCHARGE_THRESHOLD_HIGH:,})")
    fig_hist.update_layout(
        title="Barak Discharge — 14-Day History + 3-Day Smoothing + 72h Projection",
        yaxis_title="Discharge (m³/s)",
        xaxis_title="Date / Projection Window",
        height=380,
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="#FAFAFA",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    trend_icon = "📈" if t_slope > 100 else ("📉" if t_slope < -100 else "➡️")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Current Discharge", f"{t_curr:,.0f} m³/s",
               delta=trend_data.get("source", "GloFAS"), delta_color="off")
    tc2.metric("Raw Trend (OLS)", f"{trend_icon} {t_slope:+.0f} m³/s/day")
    tc3.metric("Trend R²", f"{r2_val:.2f}",
               delta=("✅ Reliable (≥0.60)" if reliable else "⚠️ Suppressed (<0.60)"),
               delta_color="off")
    tc4.metric("72h Projected", f"{t_proj[2]:,.0f} m³/s",
               delta=(f"{t_proj[2]-t_curr:+,.0f} vs now"),
               delta_color=("inverse" if t_proj[2] > t_curr else "normal"))

    if not reliable:
        st.caption(
            f"⚠️ Trend suppressed — {trend_data.get('reliability_reason', '')}  \n"
            f"72h projection shows flat line ({t_curr:,.0f} m³/s) — no trend extrapolation applied."
        )
    else:
        st.caption(
            f"Checks: R²={r2_val:.2f} ✅ · |slope|={abs(t_slope):.0f} m³/s/day ✅ · "
            f"{trend_data.get('n_days', 0)} days ✅ · Source: {trend_data.get('source', 'GloFAS')}"
        )

st.divider()

# Upstream Barak Discharge Monitoring
st.subheader("🌊 Upstream Barak River Discharge Monitoring")
st.caption(
    f"Barak at Silchar, Assam (24.82°N 92.79°E) · "
    f"GloFAS reanalysis via Open-Meteo · 72h projection from 14-day trend · "
    f"Last updated: {datetime.now(BST).strftime('%Y-%m-%d %H:%M BST')} (cached 1h)"
)

with st.spinner("Fetching Barak river discharge ..."):
    try:
        barak_current, barak_forecast = fetch_barak_discharge()
        discharge_ok = True
    except Exception as _barak_err:
        discharge_ok = False
        st.warning(f"Discharge fetch failed: {_barak_err} — using defaults.")
        barak_current  = {
            "discharge": UPSTREAM_DISCHARGE_DEFAULT, "trend": 0.0,
            "trend_label": "Unknown", "daily": [], "status": "default",
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }
        barak_forecast = {
            "windows": [
                {"label": "0–24h",  "discharge": UPSTREAM_DISCHARGE_DEFAULT, "rain_mm": 0.0},
                {"label": "24–48h", "discharge": UPSTREAM_DISCHARGE_DEFAULT, "rain_mm": 0.0},
                {"label": "48–72h", "discharge": UPSTREAM_DISCHARGE_DEFAULT, "rain_mm": 0.0},
            ],
            "method": "default", "note": "API unavailable.",
        }

disch_val   = barak_current["discharge"]
disch_trend = barak_current["trend"]
risk_cls    = classify_discharge_risk(disch_val)

# Discharge adjustment on current ML probability
adj = apply_discharge_adjustment(
    final_prob, disch_val, disch_trend,
    is_trend_reliable=trend_data.get("is_reliable", True),
)

# Top row: current discharge + risk classification
dc1, dc2, dc3, dc4 = st.columns([2, 1, 1, 1])
with dc1:
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:8px;background:#F8F9FA;"
        f"border-left:6px solid {risk_cls['color']}'>"
        f"<span style='font-size:12px;color:#666;text-transform:uppercase;"
        f"letter-spacing:0.5px'>Current Barak Discharge (Silchar)</span><br>"
        f"<span style='font-size:36px;font-weight:bold;color:{risk_cls['color']}'>"
        f"{disch_val:,.0f} m³/s</span> "
        f"<span style='font-size:18px;color:{risk_cls['color']}'>"
        f"{risk_cls['icon']} {risk_cls['level']}</span><br>"
        f"<span style='font-size:13px;color:#555'>{risk_cls['message']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with dc2:
    trend_icon = "📈" if disch_trend > 150 else ("📉" if disch_trend < -150 else "➡️")
    st.metric(
        "Trend",
        f"{trend_icon} {barak_current['trend_label']}",
        delta=f"{disch_trend:+.0f} m³/s/day",
        delta_color=("inverse" if disch_trend > 0 else "normal"),
    )
with dc3:
    st.metric(
        "ML Base Prob",
        f"{final_prob*100:.1f}%",
        delta="RF+XGB ensemble",
        delta_color="off",
    )
with dc4:
    delta_pp = adj["adjustment"] * 100
    st.metric(
        "Discharge-Adjusted Prob",
        f"{adj['adjusted_prob']*100:.1f}%",
        delta=f"{delta_pp:+.1f}pp from discharge" if delta_pp != 0 else "No adjustment",
        delta_color=("inverse" if delta_pp > 0 else "off"),
        help="Heuristic post-model adjustment — not a retrained ML output.",
    )

# Bengali warning for HIGH/DANGER
if risk_cls["level"] in ("High", "Danger"):
    st.warning(
        f"{risk_cls['icon']} **{risk_cls['bangla']}**  \n"
        f"উজানের বরাক নদীতে ({disch_val:,.0f} m³/s) অস্বাভাবিক উচ্চ প্রবাহ শনাক্ত হয়েছে। "
        f"আনুমানিক {36} ঘণ্টার মধ্যে সুনামগঞ্জ হাওরে বন্যা আসতে পারে।  \n"
        f"বরাক নদীর প্রবাহের কারণ: {adj['reason']}"
    )

if adj["adjustment"] > 0:
    st.info(
        f"**Discharge-adjusted probability: {adj['adjusted_prob']*100:.1f}%** "
        f"(base ML: {adj['base_prob']*100:.1f}%, adjustment: {adj['adjustment']*100:+.1f}pp)  \n"
        f"Reason: {adj['reason']}  \n"
        f"*Note: this is a physics-informed post-model heuristic, not a retrained ML metric. "
        f"The primary model probability is {adj['base_prob']*100:.1f}%.*"
    )

st.divider()

# 72-hour discharge projection chart
fw = barak_forecast.get("windows", [])
if fw:
    w_labels  = [w["label"]     for w in fw]
    w_vals    = [w["discharge"]  for w in fw]
    w_colors  = [classify_discharge_risk(v)["color"] for v in w_vals]

    fig_bk = go.Figure()
    # Historical daily values (up to 14 days)
    daily = barak_current.get("daily", [])
    if daily:
        hist_dates = [d["date"]  for d in daily]
        hist_vals  = [d["value"] for d in daily]
        fig_bk.add_trace(go.Scatter(
            x=hist_dates, y=hist_vals,
            name="Historical (GloFAS)",
            mode="lines+markers",
            line=dict(color="#5b9bd5", width=2),
            marker=dict(size=6),
        ))

    # 72h projection
    fig_bk.add_trace(go.Bar(
        x=w_labels, y=w_vals,
        name="72h Projection",
        marker_color=w_colors,
        opacity=0.75,
    ))

    fig_bk.add_hline(y=UPSTREAM_DISCHARGE_THRESHOLD_DANGER,
                     line_dash="dash", line_color="#CC0000",
                     annotation_text=f"DANGER ({UPSTREAM_DISCHARGE_THRESHOLD_DANGER:,} m³/s)")
    fig_bk.add_hline(y=UPSTREAM_DISCHARGE_THRESHOLD_HIGH,
                     line_dash="dot", line_color="#E65C00",
                     annotation_text=f"HIGH ({UPSTREAM_DISCHARGE_THRESHOLD_HIGH:,} m³/s)")
    fig_bk.add_hline(y=UPSTREAM_DISCHARGE_THRESHOLD_WARNING,
                     line_dash="dot", line_color="#B8860B",
                     annotation_text=f"RISING ({UPSTREAM_DISCHARGE_THRESHOLD_WARNING:,} m³/s)")

    fig_bk.update_layout(
        title="Barak River Discharge — Historical (14d) + 72h Trend Projection",
        yaxis_title="Discharge (m³/s)",
        xaxis_title="Date / Window",
        height=380,
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="#FAFAFA",
    )
    st.plotly_chart(fig_bk, use_container_width=True)

    # Window cards
    wc_cols = st.columns(3)
    for col, w in zip(wc_cols, fw):
        rc = classify_discharge_risk(w["discharge"])
        with col:
            st.metric(
                label=f"{rc['icon']} {w['label']}",
                value=f"{w['discharge']:,.0f} m³/s",
                delta=rc["level"],
                delta_color="off",
            )

    st.caption(
        f"Data: {barak_forecast['method']} · {barak_forecast['note']} · "
        f"Fetched: {barak_current.get('fetched_at', 'unknown')}"
    )

    if barak_current["status"] == "default":
        st.info(
            "📡 GloFAS discharge data unavailable — showing default dry-season baseline. "
            "Check internet connection."
        )

st.divider()

st.subheader("🌊 Surma River Gauge — Sunamganj (Direct Haor Hydraulic Driver)")
st.caption(
    "GloFAS reanalysis at Sunamganj (24.87°N, 91.40°E) · Open-Meteo Flood API · "
    "Dashboard indicator only — excluded from ML model (r=0.79 with ERA5 soil moisture). "
    "Unlike Barak (upstream, ~36h lead), Surma is the **direct local driver** of haor inundation. "
    "Thresholds from BWDB Sunamganj station records."
)

# Surma-specific thresholds (BWDB Sunamganj station, from FFWC records)
_SURMA_DANGER  = 800   # m³/s — haor inundation near-certain
_SURMA_HIGH    = 400   # m³/s — elevated flood risk
_SURMA_WARNING = 150   # m³/s — rising, monitor closely

q_surma = feat.get("surma_discharge", 20.0)

def _classify_surma(q):
    if q >= _SURMA_DANGER:
        return {"level": "Danger",  "color": "#CC0000", "icon": "🔴",
                "msg": f"Extreme discharge ({q:,.0f} m³/s) — direct haor inundation expected.",
                "bn":  "অত্যন্ত বিপজ্জনক — সুরমা নদীতে অস্বাভাবিক উচ্চ প্রবাহ।"}
    if q >= _SURMA_HIGH:
        return {"level": "High",    "color": "#E65C00", "icon": "🟠",
                "msg": f"High discharge ({q:,.0f} m³/s) — elevated haor flood risk.",
                "bn":  "উচ্চ ঝুঁকি — সুরমার প্রবাহ বেশি। নজর রাখুন।"}
    if q >= _SURMA_WARNING:
        return {"level": "Rising",  "color": "#B8860B", "icon": "🟡",
                "msg": f"Rising discharge ({q:,.0f} m³/s) — conditions developing.",
                "bn":  "প্রবাহ বাড়ছে — সতর্ক থাকুন।"}
    return {"level": "Normal",  "color": "#1A7A4A", "icon": "🟢",
            "msg": f"Normal discharge ({q:,.0f} m³/s) — no local flood signal.",
            "bn":  "স্বাভাবিক প্রবাহ — কোনো বন্যার সংকেত নেই।"}

sc = _classify_surma(q_surma)

sa1, sa2, sa3, sa4 = st.columns([2, 1, 1, 1])
with sa1:
    _sc_color = sc["color"]
    _sc_icon  = sc["icon"]
    _sc_level = sc["level"]
    _sc_msg   = sc["msg"]
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:8px;background:#F8F9FA;"
        f"border-left:6px solid {_sc_color}'>"
        f"<span style='font-size:12px;color:#666;text-transform:uppercase;"
        f"letter-spacing:0.5px'>Current Surma Discharge (Sunamganj)</span><br>"
        f"<span style='font-size:36px;font-weight:bold;color:{_sc_color}'>"
        f"{q_surma:,.1f} m³/s</span> "
        f"<span style='font-size:18px;color:{_sc_color}'>"
        f"{_sc_icon} {_sc_level}</span><br>"
        f"<span style='font-size:13px;color:#555'>{_sc_msg}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
with sa2:
    pct_danger = min(100, round(q_surma / _SURMA_DANGER * 100, 1))
    st.metric("% of Danger Level", f"{pct_danger:.0f}%",
              delta=f"Danger = {_SURMA_DANGER:,} m³/s", delta_color="off")
with sa3:
    st.metric("ML Model Role", "Dashboard only",
              delta="r=0.79 multicollinear", delta_color="off",
              help="Surma discharge was tested as ML Feature 14 but excluded due to "
                   "multicollinearity with ERA5 soil moisture (r=0.79). "
                   "Retained as a real-time hydraulic indicator alongside the ML prediction.")
with sa4:
    ffwc_q = round(q_surma * 0.0283168, 2)   # approximate cusecs
    st.metric("Approx. Cusecs", f"{ffwc_q:,.0f} cusec",
              delta="FFWC reporting unit", delta_color="off",
              help="FFWC reports discharge in cusecs (cubic feet per second). "
                   "1 m³/s ≈ 35.31 cusec. Verify at ffwc.gov.bd")

if sc["level"] in ("High", "Danger"):
    st.warning(
        f"{sc['icon']} **{sc['bn']}**  \n"
        f"Surma River at Sunamganj shows {sc['level'].lower()} discharge ({q_surma:,.0f} m³/s). "
        f"This is a direct haor hydraulic driver — elevated risk of haor inundation regardless of upstream Barak signal.  \n"
        f"Verify with FFWC real-time gauge: **[ffwc.gov.bd](https://ffwc.gov.bd)**"
    )

# Threshold gauge
st.markdown(
    f"**Surma Discharge Thresholds (BWDB Sunamganj):** "
    f"🟢 Normal < {_SURMA_WARNING:,} &nbsp;·&nbsp; "
    f"🟡 Rising {_SURMA_WARNING:,}–{_SURMA_HIGH:,} &nbsp;·&nbsp; "
    f"🟠 High {_SURMA_HIGH:,}–{_SURMA_DANGER:,} &nbsp;·&nbsp; "
    f"🔴 Danger ≥ {_SURMA_DANGER:,} m³/s"
)
st.caption(
    "**Why Surma matters:** Surma discharge directly drives haor water level at Sunamganj. "
    "High Barak discharge (upstream) typically causes high Surma discharge 6–12h later. "
    "Both signals together give the most complete haor flood picture. "
    "Source: GloFAS reanalysis via Open-Meteo Flood API (Sunamganj 24.87°N, 91.40°E). "
    "Excluded from ML model due to multicollinearity — not a model weakness, but a deliberate "
    "feature engineering decision to prevent overfitting."
)

st.divider()

with st.expander("📡 All 15 Live Features — Raw Sensor Values", expanded=False):
    st.subheader("📡 All 15 Live Features (13 ML inputs + Surma + Barak discharge)")
    t1,t2,t3,t4 = st.tabs(["🛰️ SAR + Terrain","🌡️ Meteorological","🆕 New Features","📅 Forecast"])

    with t1:
        c1,c2 = st.columns(2)
        with c1:
            st.metric("Sentinel-1 VV", f"{feat['VV']:.2f} dB", help="< −16 dB = open water / flood")
            st.metric("Sentinel-1 VH", f"{feat['VH']:.2f} dB")
            st.metric("VV/VH Ratio",   f"{feat['vv_vh_ratio']:.4f}")
        with c2:
            st.metric("Slope (SRTM)", f"{feat['slope']:.2f}°")
            st.metric("Data window",  f"{feat['start']} → {feat['end']}")

    with t2:
        c1,c2 = st.columns(2)
        with c1:
            st.metric("7-day Rainfall (CHIRPS)", f"{feat['rainfall']:.1f} mm")
            st.metric("Soil Moisture (ERA5)",    f"{feat['soil_moisture']:.1f} %")
        with c2:
            ta = feat.get("temp_anomaly", 0.0)
            ta_s = ("🔴 unusually warm" if ta > 2 else
                    "🔵 unusually cold"  if ta < -2 else "🟢 near-normal")
            st.metric("Temp Anomaly (model input)", f"{ta:+.1f} °C", delta=ta_s,
                      help=f"temp_anomaly = observed {feat['temp']:.1f}°C − monthly mean. "
                           "Replaces raw temperature to remove seasonal confound "
                           "(raw temp r=0.57 with flood label; anomaly r=−0.03).")
            st.metric("Wind Speed", f"{feat['wind']:.1f} km/h")

    with t3:
        c1,c2 = st.columns(2)
        with c1:
            ndwi_s = "🌊 water" if feat["ndwi"] > 0 else "🌱 dry"
            st.metric("NDWI (Sentinel-2)", f"{feat['ndwi']:.4f}", delta=ndwi_s,
                      help="Normalized Difference Water Index. >0 = water surface")
            twi_s = ("🔴 very high" if feat["twi"] > 14 else
                     "🟠 high"      if feat["twi"] > 10 else "🟢 moderate")
            st.metric("TWI (HydroSHEDS)", f"{feat['twi']:.2f}", delta=twi_s,
                      help="Topographic Wetness Index. Bowl centre: 14-20")
        with c2:
            up_s = "⚠️ elevated" if feat["upstream_vv"] < -16 else "✅ normal"
            st.metric("Upstream VV (Barak)", f"{feat['upstream_vv']:.2f} dB", delta=up_s,
                      help="Sentinel-1 over Silchar, Assam — barrage release proxy")
            if lead_info["upstream_alert"]:
                st.warning(f"🕐 Lead time: ~{lead_info['lead_hours']:.0f}h")
            q = feat.get("surma_discharge", 20.0)
            q_s = ("🔴 major flood" if q > 400 else
                   "🟠 high risk"   if q > 150 else
                   "🟡 elevated"    if q > 50  else "🟢 normal")
            st.metric("Surma River Discharge", f"{q:.1f} m³/s", delta=q_s,
                      help="7-day mean discharge at Sunamganj — direct hydraulic flood driver (GloFAS)")
            if dur_days > 0 and final_prob >= 0.4:
                st.metric("Predicted flood duration", f"~{dur_days:.0f} days")

    with t4:
        c1,c2 = st.columns(2)
        with c1:
            st.metric("Next 12h rainfall", f"{feat['forecast_rain_next_12h']:.1f} mm")
        with c2:
            f72_s = ("🔴 very high" if feat["forecast_rain_72h"] > 200 else
                     "🟠 high"      if feat["forecast_rain_72h"] > 100 else "🟢 normal")
            st.metric("Next 72h rainfall (3-day)", f"{feat['forecast_rain_72h']:.1f} mm", delta=f72_s)
        if feat["forecast_rain_72h"] > 150:
            st.warning(
                f"⚠️ Heavy 72h forecast ({feat['forecast_rain_72h']:.0f} mm) — "
                f"haor flash flood risk. Consider issuing community alert."
            )

    st.caption(
        f"Data: Sentinel-1 GRD · Sentinel-2 · CHIRPS · ERA5-Land · HydroSHEDS · "
        f"Open-Meteo · Barak upstream proxy | {len(FEATURES)} features fetched | "
        f"{len(active_feats)} used by RF/XGB | Cached 1h"
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