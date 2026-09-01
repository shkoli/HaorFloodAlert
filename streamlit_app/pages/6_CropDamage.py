"""
6_CropDamage.py — Boro Rice Crop Damage + Flood Duration
First haor-specific agricultural impact model integrated with ML flood prediction.

DATA HONESTY:
  - Yield constants (45 t/km², 28k BDT/t): REAL from BRRI/BBS 2024 publications
  - Historical impact (2017/2019/2022/2024): ESTIMATED from BWDB reports (approximate)
  - Calculation model: empirical formula, not field-measured
  - Flood duration: calibrated from BWDB gauge records, not physics simulation
  - Slider inputs: user-defined scenario (for exploration), not measured values
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from utils.boro_damage import (
    BORO_YIELD_KM2, BORO_PRICE_BDT, estimate_duration, estimate_damage,
    upazila_breakdown,
)

st.set_page_config(page_title="Crop Damage", page_icon="🌾", layout="wide")
st.title("🌾 Boro Rice Crop Damage Estimation")
st.subheader("Sunamganj Haor — Flood Impact on Agriculture + Flood Duration Prediction")

# Data transparency banner
st.info(
    "**Data transparency:** Yield constants (45 t/km², ৳28,000/ton) are from "
    "published BRRI/BBS 2024 statistics. Historical impact figures are approximate "
    "estimates from BWDB flood damage assessment reports. The damage calculation "
    "is an empirical academic model, not a field-measured result. "
    "Flood duration is calibrated from BWDB Sunamganj gauge records (2017–2024)."
)

# Sunamganj Haor agricultural constants and estimate_duration()/estimate_damage()
# now live in utils/boro_damage.py (shared with eval/run_journal_extension.py so
# the model exists in exactly one place). BORO_YIELD_KM2/BORO_PRICE_BDT are
# imported above for use in the upazila breakdown table below.


# Boro rice phenological calendar
st.markdown("### 📅 Boro Rice Phenological Calendar — Flood Vulnerability by Month")
st.caption(
    "Source: Bangladesh Rice Research Institute (BRRI) 2024. "
    "Haor boro rice grows Oct–May; floods before harvest cause the highest losses."
)

PHENO = pd.DataFrame({
    "Month": ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May"],
    "Month_num": [10, 11, 12, 1, 2, 3, 4, 5],
    "Stage": [
        "Nursery / নার্সারি",
        "Transplanting / রোপণ",
        "Vegetative / কায়িক বৃদ্ধি",
        "Booting / বুটিং",
        "Panicle initiation / শীষ শুরু",
        "Pre-harvest / আগাম কাটার আগে ★",
        "Grain fill / harvest ★★ / দানা পূরণ",
        "Harvest / কাটার সময়",
    ],
    "Flood Damage Rate (%)": [10, 15, 20, 30, 45, 85, 100, 70],
    "Risk": ["Low","Low","Low","Moderate","Moderate","High","Critical","High"],
})
color_map = {"Low": "#00C49A", "Moderate": "#FFD700", "High": "#FF8C00", "Critical": "#FF4B4B"}
fig_pheno = go.Figure()
for _, row in PHENO.iterrows():
    fig_pheno.add_trace(go.Bar(
        x=[row["Month"]], y=[row["Flood Damage Rate (%)"]],
        name=row["Risk"],
        marker_color=color_map[row["Risk"]],
        text=f'{row["Stage"]}<br>{row["Flood Damage Rate (%)"]}%',
        textposition="inside",
        showlegend=False,
    ))
fig_pheno.add_hline(y=70, line_dash="dash", line_color="red",
                    annotation_text="Catastrophic loss threshold (>70%)")
fig_pheno.update_layout(
    title="Boro Rice Flood Damage Rate by Month — Sunamganj Haor",
    yaxis_title="Estimated Crop Damage (%)",
    xaxis_title="Month",
    height=340, showlegend=False,
    bargap=0.15,
)
st.plotly_chart(fig_pheno, use_container_width=True)
st.caption(
    "Pre-monsoon flash floods in **March–May (মার্চ–মে)** strike during grain filling and harvest, "
    "causing 85–100% crop loss. ★★ = Critical window (সবচেয়ে বিপজ্জনক সময়). "
    "This is why Boro harvest in haors is so vulnerable — "
    "floodwaters arrive just weeks before farmers can collect the crop. "
    "Source: Bangladesh Rice Research Institute (BRRI) flood impact assessment methodology."
)

st.divider()

# Input panel
st.markdown("### 🎛️ Flood Scenario Input")
st.caption(
    "Adjust sliders to simulate any flood scenario. "
    "For live flood: use values from the Prediction page."
)

c1, c2, c3 = st.columns(3)
with c1:
    prob            = st.slider("Flood Probability (%)", 0, 100, 71) / 100
    area_km2        = st.slider("Inundated Area (km²)", 0, 500, 107, 5)
    flood_depth_cm  = st.slider("Flood Depth (cm)", 5, 200, 60, 5,
                                help="Average water depth above ground in flooded haor area")
with c2:
    soil  = st.slider("Soil Moisture (%)", 5, 70, 45)
    upvv  = st.slider("Upstream VV (dB)", -25.0, -5.0, -17.5, 0.5)
with c3:
    rain  = st.slider("7-day Rainfall (mm)", 0, 400, 180, 10)
    month = st.selectbox(
        "Month of flood", list(range(1, 13)), index=3,
        format_func=lambda m: datetime(2024, m, 1).strftime("%B")
    )

dur    = estimate_duration(prob, soil, upvv, rain)
damage = estimate_damage(area_km2, prob, dur["days"], month, flood_depth_cm)

st.divider()

# Flood Duration Results
st.markdown("### ⏱️ Flood Duration Prediction")
st.caption(
    "Empirical model calibrated from BWDB Sunamganj gauge station records 2017–2024. "
    "95% CI reflects ±30% empirical uncertainty in haor drainage patterns."
)

d1, d2, d3, d4 = st.columns(4)
d1.metric("Estimated Duration", f"{dur['days']} days")
d2.metric("Duration Category",  dur["category"])
d3.metric("95% CI Lower",       f"{dur['ci_low']} days")
d4.metric("95% CI Upper",       f"{dur['ci_high']} days")

bar_color = ("#FF4B4B" if dur["days"] > 21 else
             "#FF8C00" if dur["days"] > 8  else "#FFD700")

fig_dur = go.Figure()
fig_dur.add_trace(go.Bar(
    x=["Flood Duration Estimate"], y=[dur["days"]],
    error_y=dict(type="data", symmetric=False,
                 array=[dur["ci_high"] - dur["days"]],
                 arrayminus=[max(0, dur["days"] - dur["ci_low"])]),
    marker_color=bar_color, width=0.3,
    name="Duration",
))
fig_dur.add_hline(y=7, line_dash="dash", line_color="red",
                  annotation_text="7 days → 100% boro crop loss threshold (BRRI)")
fig_dur.update_layout(
    height=280, yaxis_title="Days",
    title="Predicted flood duration with 95% confidence interval"
)
st.plotly_chart(fig_dur, use_container_width=True)

st.divider()

# Crop Damage Results
st.markdown("### 🌾 Boro Rice Crop Damage Estimate")

# Fixed: use st.markdown with HTML instead of st.warning ternary (DeltaGenerator bug)
st.markdown(
    f'<div style="background:{damage["color"]}22;border-left:4px solid {damage["color"]};'
    f'padding:12px 16px;border-radius:4px;margin-bottom:12px;">'
    f'<b>{damage["msg"]}</b></div>',
    unsafe_allow_html=True,
)

r1, r2, r3 = st.columns(3)
r1.metric("Boro Area in Flood Zone",   f"{damage['boro_in']} km²",
          help="72% of inundated area is boro rice cultivation (BRRI survey)")
r2.metric("Crop Damaged Area",         f"{damage['damaged']} km²",
          delta=f"{damage['rate_pct']}% damage rate", delta_color="inverse")
r3.metric("Affected Farming Families", f"{damage['farmers']:,}",
          help="2.3 farming families per hectare (BRRI census data)")

r4, r5, r6 = st.columns(3)
r4.metric("Rice Production Lost",  f"{damage['lost_tons']:,.0f} metric tons",
          help="Damaged area × 45 t/km² BRRI average yield")
r5.metric("Economic Loss (BDT)",   f"৳{damage['bdt_crore']:.2f} crore",
          help="Based on ৳28,000/ton market price (BBS 2024)")
r6.metric("Economic Loss (USD)",   f"${damage['usd_million']:.2f} million",
          help="USD 255/ton (BDT/USD ≈ 110)")

# BRRI model factor breakdown
with st.expander("🔬 BRRI Damage Model Decomposition (for thesis)"):
    st.markdown(
        f"**Growth stage:** {damage['stage_name']}  \n"
        f"**Stage max sensitivity:** {damage['sens_max']:.0f}%  \n"
        f"**Depth factor** (flood depth {flood_depth_cm} cm): **{damage['depth_factor']:.2f}**  \n"
        f"**Duration factor** (duration {dur['days']} days): **{damage['dur_factor']:.2f}**  \n"
        f"**Flood probability scale:** {prob*100:.0f}%  \n\n"
        f"**Damage rate = {damage['sens_max']:.0f}% × {damage['depth_factor']:.2f} × "
        f"{damage['dur_factor']:.2f} × {min(prob*1.15,1.0):.2f} = **{damage['rate_pct']:.1f}%****  \n"
        f"**95% uncertainty range:** {damage['rate_low']:.1f}% – {damage['rate_high']:.1f}%  \n\n"
        "Source: Bangladesh Rice Research Institute (BRRI), Flood Impact Assessment Methodology (2022), Table 4.2"
    )
    fig_factors = go.Figure(go.Bar(
        x=["Stage sensitivity", "Depth factor", "Duration factor", "Combined damage rate"],
        y=[damage["sens_max"],
           damage["depth_factor"] * 100,
           damage["dur_factor"] * 100,
           damage["rate_pct"]],
        marker_color=["#5b9bd5", "#FF8C00", "#FFD700", damage["color"]],
        text=[f"{damage['sens_max']:.0f}%",
              f"{damage['depth_factor']*100:.0f}%",
              f"{damage['dur_factor']*100:.0f}%",
              f"{damage['rate_pct']:.1f}%"],
        textposition="outside",
    ))
    fig_factors.update_layout(
        title="BRRI Damage Model Factor Breakdown",
        yaxis_title="Factor value (%)",
        height=280,
    )
    st.plotly_chart(fig_factors, use_container_width=True)

# Area breakdown chart
safe = max(0, damage["boro_in"] - damage["damaged"])
non  = max(0, area_km2 - damage["boro_in"])
fig_pie = go.Figure(go.Pie(
    labels=["Boro crop damaged", "Boro crop safe", "Non-agricultural"],
    values=[damage["damaged"], safe, non],
    marker_colors=["#FF4B4B", "#00C49A", "#888888"],
    hole=0.4,
))
fig_pie.update_layout(title="Flooded Area Breakdown", height=320)
st.plotly_chart(fig_pie, use_container_width=True)

# Data honesty note
st.caption(
    "⚠️ Damage rate is an academic estimation based on BRRI flood impact methodology. "
    "Actual field losses may vary by 20–40% depending on variety, water depth, "
    "and drainage conditions specific to each haor sub-basin."
)

# Upazila-level economic breakdown
st.divider()
st.markdown("### 🏘️ Upazila-Level Economic Impact Breakdown — Sunamganj")
st.caption(
    "Based on BBS Agricultural Statistics 2023 upazila-level Boro rice acreage "
    "and BRRI yield survey data. Each upazila's share of total district damage "
    "is proportional to its Boro cultivation area."
)

UPAZILA_DATA = upazila_breakdown(damage["damaged"]).rename(columns={
    "upazila":           "Upazila",
    "boro_area_km2":     "Boro Area (km²)",
    "share_pct":         "Share (%)",
    "damaged_km2":       "Damaged (km²)",
    "rice_lost_tons":    "Rice Lost (tons)",
    "loss_crore_bdt":    "Loss (crore ৳)",
    "families_affected": "Families Affected",
})
st.dataframe(
    UPAZILA_DATA[["Upazila", "Boro Area (km²)", "Share (%)",
                  "Damaged (km²)", "Rice Lost (tons)", "Loss (crore ৳)", "Families Affected"]],
    use_container_width=True, hide_index=True
)
fig_up = px.bar(
    UPAZILA_DATA, x="Upazila", y="Loss (crore ৳)",
    color="Loss (crore ৳)",
    color_continuous_scale=["#00C49A", "#FFD700", "#FF8C00", "#FF4B4B"],
    title=f"Upazila-Level Economic Loss — {datetime(2024, month, 1).strftime('%B')} flood scenario",
    text="Loss (crore ৳)",
)
fig_up.update_traces(texttemplate="৳%{text:.2f}cr", textposition="outside")
fig_up.update_layout(height=340, coloraxis_showscale=False)
st.plotly_chart(fig_up, use_container_width=True)
st.caption(
    "Source: BBS Agricultural Statistics 2023 · BRRI Upazila Boro Survey 2022 · "
    "DAE (Department of Agricultural Extension) Sunamganj district office. "
    "**Tahirpur and Jamalganj upazilas are most vulnerable** — largest haor Boro areas."
)

# Flood probability → damage sensitivity
st.divider()
st.markdown("### 📈 How Flood Probability Drives Crop Loss — Sensitivity Analysis")
st.caption(
    f"Current month: **{datetime(2024, month, 1).strftime('%B')}** | "
    f"Duration: **{dur['days']} days** | Area: **{area_km2} km²**"
)
prob_range  = [p / 100 for p in range(0, 101, 5)]
loss_bdt    = []
loss_tons_r = []
for p in prob_range:
    _d = estimate_duration(p, soil, upvv, rain)
    _dmg = estimate_damage(area_km2, p, _d["days"], month, flood_depth_cm)
    loss_bdt.append(_dmg["bdt_crore"])
    loss_tons_r.append(_dmg["lost_tons"])

fig_sens = go.Figure()
fig_sens.add_trace(go.Scatter(
    x=[p * 100 for p in prob_range], y=loss_bdt,
    mode="lines", name="Loss (crore BDT)",
    line=dict(color="#FF8C00", width=3),
    fill="tozeroy", fillcolor="rgba(255,140,0,0.15)",
))
fig_sens.add_vline(
    x=prob * 100, line_dash="dash", line_color="white",
    annotation_text=f"Current: {prob*100:.0f}% → ৳{damage['bdt_crore']:.2f} cr",
)
fig_sens.update_layout(
    title=f"Economic Loss vs Flood Probability ({datetime(2024, month, 1).strftime('%B')}, {area_km2} km², {dur['days']} days)",
    xaxis_title="Flood Probability (%)",
    yaxis_title="Economic Loss (crore BDT)",
    height=320,
)
st.plotly_chart(fig_sens, use_container_width=True)
st.caption(
    "This chart shows how the ML model's flood probability output translates directly "
    "into estimated economic loss. Higher flood probability → longer duration → larger "
    "crop damage area → greater economic impact. The ML ensemble provides the probability; "
    "this module converts it into actionable agricultural impact estimates."
)

st.divider()

# Historical Reference
st.subheader("📊 Historical Flood Impact Reference — Sunamganj Haor")
st.caption(
    "⚠️ These figures are **approximate estimates** compiled from BWDB Flood Damage "
    "Assessment Reports (2017–2024). Exact field-verified data is not publicly available. "
    "Values represent district-level aggregates, not haor-specific measurements."
)

hist = pd.DataFrame({
    "Year":              [2017,  2019,  2022,   2024],
    "Inundated (km²)":   [420,   280,   520,    350],
    "Rice Lost (tons)":  [58000, 38000, 72000,  49000],
    "Loss (crore BDT)":  [16.2,  10.6,  20.2,   13.7],
    "Families Affected": [95000, 63000, 118000, 80000],
})
st.dataframe(hist, use_container_width=True, hide_index=True)

fig_hist = px.bar(
    hist, x="Year", y="Rice Lost (tons)",
    color_discrete_sequence=["#FF8C00"],
    title="Boro Rice Production Lost by Year — Sunamganj Haor (Estimated)",
    text="Rice Lost (tons)",
)
fig_hist.update_traces(texttemplate="%{text:,}", textposition="outside")
fig_hist.update_layout(height=320)
st.plotly_chart(fig_hist, use_container_width=True)

st.caption(
    "Source: BWDB Flood Damage Assessment Reports 2017–2024 · "
    "Bangladesh Bureau of Statistics (BBS) Agricultural Statistics · "
    "Bangladesh Rice Research Institute (BRRI) yield data. "
    "**Note: All historical values are estimates from government reports, not field surveys.**"
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

st.divider()

# Novelty statement
st.info(
    "**Novel academic contribution:**  \n"
    "This is the **first haor-specific flood impact model** that integrates ML flood "
    "probability with the Boro rice phenological calendar and BRRI agricultural data "
    "to provide scenario-based economic loss estimates for Sunamganj haor.  \n"
    "Previous haor flood prediction studies (Uddin 2019, Singha 2020, Islam 2021) "
    "focused exclusively on inundation mapping — none quantified agricultural economic impact."
)

# Limitations
with st.expander("⚠️ Model Limitations (important for thesis)"):
    st.markdown("""
    1. **Yield constant (45 t/km²)** is BRRI district average — actual haor yields vary 
       by variety (BR11, BRRI dhan28/29) and field condition (30–55 t/km²)
    2. **Damage rate formula** is academic estimation, not field-calibrated
    3. **Historical impact figures** are BWDB district-level estimates, not haor-specific
    4. **Flood duration model** uses empirical calibration, not hydrodynamic simulation
    5. **2.3 families/hectare** is a survey average — actual density varies by upazila
    6. **Price (28,000 BDT/ton)** is 2024 market average — fluctuates ±15% seasonally
    """)
