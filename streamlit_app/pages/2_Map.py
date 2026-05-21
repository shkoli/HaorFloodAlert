"""
2_Map.py — Sentinel-1 SAR Flood Inundation Map
================================================
Method: Otsu bi-modal thresholding on SAR change detection image.
        Pre-flood dry-season reference vs at-flood composite.

References (from thesis proposal Section 4 — Literature Review):
  Uddin et al. (2019) Remote Sensing 11(13):1581 — operational flood mapping with S1
  Singha et al. (2020) ISPRS Journal 166:278-293 — S1 flood mapping on GEE

Why Otsu instead of fixed threshold:
  A fixed -15 dB threshold fails because dry haor land in Jan–Feb can have VV of
  -10 to -14 dB, meaning genuine floods (VV drops to -14 dB) are missed entirely.
  Otsu automatically finds the optimal threshold for each date by maximising the
  inter-class variance between "changed" (flooded) and "unchanged" (dry) pixels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta, timezone, date

from config import GEE_PROJECT, HAOR_BBOX, HAOR_LAT, HAOR_LON, UPSTREAM_BBOX

st.set_page_config(page_title="Flood Map", page_icon="🗺️", layout="wide")
st.title("🗺️ SAR Flood Inundation Map — Sentinel-1 Otsu Change Detection")
st.subheader(
    "Sunamganj Haor | Copernicus Sentinel-1 GRD via Google Earth Engine | "
    "Pre-flood dry-season reference vs at-flood composite | "
    "Otsu bi-modal thresholding (Uddin et al. 2019; Singha et al. 2020)"
)

# High-risk communities
VILLAGES = [
    {"name": "Sunamganj Sadar",  "lat": 24.8897, "lon": 91.3966},
    {"name": "Tahirpur",         "lat": 24.9348, "lon": 91.4889},
    {"name": "Derai",            "lat": 24.7697, "lon": 91.5280},
    {"name": "Jagannathpur",     "lat": 24.7963, "lon": 91.4678},
    {"name": "Sulla",            "lat": 24.8330, "lon": 91.6230},
    {"name": "Doarabazar",       "lat": 24.8820, "lon": 91.5490},
]


# Otsu threshold (Python, client-side)
def otsu_threshold(counts: list, means: list) -> float:
    """
    Otsu's method: find threshold that maximises inter-class variance.
    Applied to the SAR change histogram (pre_VV - flood_VV in dB).
    Positive change > threshold → flooded pixel.

    Reference: Otsu (1979) IEEE Trans. Systems, Man, Cybernetics 9(1):62-66.
               Applied to SAR flood mapping by Uddin et al. (2019).
    """
    counts = np.array(counts, dtype=np.float64)
    means  = np.array(means,  dtype=np.float64)

    total     = counts.sum()
    if total == 0:
        return 3.0   # fallback: 3 dB change

    sum_total = (counts * means).sum()
    w_b, sum_b, best_var, best_thresh = 0.0, 0.0, 0.0, float(means[0])

    for i in range(len(counts)):
        w_b  += counts[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += counts[i] * means[i]
        m_b    = sum_b / w_b
        m_f    = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * (m_b - m_f) ** 2
        if between_var > best_var:
            best_var   = between_var
            best_thresh = means[i]

    return float(best_thresh)


# GEE: load S1 composite with speckle filtering
def _s1_composite(haor, start: str, end: str):
    """Load Sentinel-1 VV median composite with focal-mean speckle filter."""
    raw = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(haor)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .median()
        .clip(haor)
    )
    # Speckle filter: focal mean (50 m radius, circle kernel)
    return raw.focal_mean(50, "circle", "meters")


def _count_s1(haor, start: str, end: str) -> int:
    return int(
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(haor).filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW")).size().getInfo()
    )


# Main GEE computation
@st.cache_data(ttl=3600, show_spinner=False)
def compute_flood_layers(flood_end_str: str) -> dict:
    """
    Full Otsu change detection pipeline.

    Steps:
      1. Load dry-season reference composite (Jan–Feb = haor driest period)
      2. Load at-flood composite (user-selected window)
      3. Speckle filter both images (focal mean, 50 m)
      4. Compute change image: reference_VV − flood_VV  (positive = VV dropped = water)
      5. Extract histogram client-side, compute Otsu threshold in Python
      6. Apply threshold → raw flood mask
      7. Remove permanent water bodies (JRC Global Surface Water ≥10 months/year)
      8. Estimate flood area (km²)
      9. Return all tile URLs + statistics
    """
    try:
        ee.Initialize(project=GEE_PROJECT)
        haor     = ee.Geometry.Rectangle(HAOR_BBOX)
        upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)

        flood_end   = datetime.strptime(flood_end_str, "%Y-%m-%d").date()
        flood_start = flood_end - timedelta(days=12)

        # Reference: January–February of the same year (lowest water level in haors)
        ref_year  = flood_end.year if flood_end.month >= 3 else flood_end.year - 1
        ref_start = f"{ref_year}-01-01"
        ref_end   = f"{ref_year}-02-28"

        flood_start_str = str(flood_start)

        # 1. Composites
        n_ref   = _count_s1(haor, ref_start, ref_end)
        n_flood = _count_s1(haor, flood_start_str, flood_end_str)

        if n_flood == 0:
            return {"error": f"No Sentinel-1 images found for {flood_start_str} → {flood_end_str}. "
                             "Try a wider date window or different date."}

        s1_ref   = _s1_composite(haor, ref_start, ref_end)
        s1_flood = _s1_composite(haor, flood_start_str, flood_end_str)

        # 2. Change image (dB)
        # Positive change means VV dropped (flooded) — exactly the flood signal
        change = s1_ref.select("VV").subtract(s1_flood.select("VV")).rename("change")

        # 3. Otsu threshold (client-side)
        hist_info = change.reduceRegion(
            reducer=ee.Reducer.histogram(maxBuckets=128, minBucketWidth=0.1),
            geometry=haor,
            scale=20,
            maxPixels=int(1e8),
        ).getInfo()

        hist    = hist_info.get("change", {})
        counts  = hist.get("histogram", [])
        means   = hist.get("bucketMeans", [])

        if len(counts) > 5 and len(means) > 5:
            threshold = otsu_threshold(counts, means)
        else:
            threshold = 3.0   # fallback if histogram is too sparse

        # Enforce minimum: always require at least 2 dB change to call it flooded
        threshold = max(threshold, 2.0)

        # 4. Flood mask
        raw_flood = change.gt(threshold)

        # 5. Remove permanent water (JRC Global Surface Water)
        jrc            = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality")
        permanent_water = jrc.gte(10)             # present ≥10 months/year
        flood_mask      = raw_flood.where(permanent_water, 0).selfMask()

        # 6. Flood area (km²)
        area_km2_img = flood_mask.multiply(ee.Image.pixelArea()).divide(1e6)
        area_val     = area_km2_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=haor, scale=20, maxPixels=int(1e8)
        ).getInfo().get("change", 0) or 0

        # 7. Upstream Barak VV (current)
        s1_up = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(upstream)
            .filterDate(flood_start_str, flood_end_str)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select(["VV"]).median().clip(upstream)
        )
        up_vv_val = s1_up.select("VV").reduceRegion(
            ee.Reducer.mean(), upstream, 20
        ).getInfo().get("VV")
        upstream_vv = round(float(up_vv_val), 2) if up_vv_val else None

        # 8. Tile URLs
        def tile_url(img, vis):
            return img.getMapId(vis)["tile_fetcher"].url_format

        urls = {
            "vv_ref":    tile_url(s1_ref.select("VV"),
                                  {"min": -25, "max": 0,
                                   "palette": ["000000", "888888", "ffffff"]}),
            "vv_flood":  tile_url(s1_flood.select("VV"),
                                  {"min": -25, "max": 0,
                                   "palette": ["000000", "888888", "ffffff"]}),
            "change":    tile_url(change,
                                  {"min": -4, "max": 8,
                                   "palette": ["1a237e", "90caf9", "ffffff",
                                               "ffb74d", "b71c1c"]}),
            "flood":     tile_url(flood_mask,
                                  {"min": 0, "max": 1, "palette": ["0055FF"]}),
            "permanent": tile_url(permanent_water.selfMask(),
                                  {"min": 0, "max": 1, "palette": ["00BCD4"]}),
        }

        return {
            "error":        None,
            "urls":         urls,
            "threshold":    round(threshold, 2),
            "flood_area":   round(float(area_km2_img.reduceRegion(
                                ee.Reducer.sum(), haor, 20, maxPixels=int(1e8)
                            ).getInfo().get("change", area_val) or area_val), 1),
            "n_ref":        n_ref,
            "n_flood":      n_flood,
            "ref_period":   f"{ref_start} → {ref_end}",
            "flood_period": f"{flood_start_str} → {flood_end_str}",
            "upstream_vv":  upstream_vv,
            "hist_counts":  counts,
            "hist_means":   means,
        }

    except Exception as exc:
        return {"error": str(exc)}


# Folium map builder
def build_map(result: dict, show_layers: list) -> folium.Map:
    m = folium.Map(
        location=[HAOR_LAT, HAOR_LON],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # Study area boundary
    folium.Rectangle(
        bounds=[[HAOR_BBOX[1], HAOR_BBOX[0]], [HAOR_BBOX[3], HAOR_BBOX[2]]],
        color="#FF8C00", weight=2,
        fill=True, fill_color="#FF8C00", fill_opacity=0.05,
        tooltip="Study Area — Sunamganj Haor",
    ).add_to(m)

    # Tanguar Haor (northwest basin)
    folium.Polygon(
        locations=[[24.90, 91.42], [24.90, 91.52], [24.98, 91.52], [24.98, 91.42]],
        color="#00BFFF", weight=2,
        fill=True, fill_color="#00BFFF", fill_opacity=0.15,
        tooltip="Tanguar Haor (northern basin)",
    ).add_to(m)

    urls = result.get("urls", {})

    if "VV Reference (dry)" in show_layers and urls.get("vv_ref"):
        folium.TileLayer(
            tiles=urls["vv_ref"], attr="Copernicus/GEE",
            name=f"S1 VV — Reference ({result['ref_period'][:7]})",
            overlay=True, control=True, show=False,
        ).add_to(m)

    if "VV At-flood" in show_layers and urls.get("vv_flood"):
        folium.TileLayer(
            tiles=urls["vv_flood"], attr="Copernicus/GEE",
            name=f"S1 VV — At-flood ({result['flood_period'][:10]})",
            overlay=True, control=True, show=False,
        ).add_to(m)

    if "Change image" in show_layers and urls.get("change"):
        folium.TileLayer(
            tiles=urls["change"], attr="Copernicus/GEE",
            name=f"VV Change (Δ dB) — threshold={result['threshold']} dB",
            overlay=True, control=True, show=False,
        ).add_to(m)

    if "Permanent water (JRC)" in show_layers and urls.get("permanent"):
        folium.TileLayer(
            tiles=urls["permanent"], attr="JRC/GEE",
            name="Permanent water (JRC, excluded)",
            overlay=True, control=True, show=False,
        ).add_to(m)

    if "Flood mask (Otsu)" in show_layers and urls.get("flood"):
        folium.TileLayer(
            tiles=urls["flood"], attr="Copernicus/GEE",
            name=f"Flood extent — Otsu >{result['threshold']} dB",
            overlay=True, control=True, show=True,
        ).add_to(m)

    # Village markers
    for v in VILLAGES:
        folium.CircleMarker(
            location=[v["lat"], v["lon"]],
            radius=9, color="#FF4B4B",
            fill=True, fill_color="#FF4B4B", fill_opacity=0.85,
            tooltip=f"⚠️ {v['name']} — high-risk community",
            popup=folium.Popup(
                f"<b>{v['name']}</b><br>High-risk haor community<br>"
                f"Flood area: {result.get('flood_area', '?')} km²",
                max_width=200
            ),
        ).add_to(m)

    folium.Marker(
        location=[HAOR_LAT, HAOR_LON],
        popup="Sunamganj Haor — Study Centre",
        tooltip="Study Centre (91.45°E, 24.87°N)",
        icon=folium.Icon(color="darkblue", icon="info-sign", prefix="glyphicon"),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# Streamlit UI

# Sidebar controls
st.sidebar.header("⚙️ Map Controls")

use_custom = st.sidebar.checkbox("Analyse a specific historical flood event", value=False)

if use_custom:
    event_date = st.sidebar.date_input(
        "Flood event end date",
        value=date(2024, 4, 20),
        min_value=date(2015, 1, 1),
        max_value=date.today(),
        help="Script loads S1 images from 12 days before this date as the 'at-flood' window.",
    )
    flood_end_str = str(event_date)
    st.sidebar.caption(
        "**Suggested historical events:**\n"
        "- 2024-04-20 (April 2024 flash flood)\n"
        "- 2022-04-25 (2022 catastrophic flood)\n"
        "- 2019-07-15 (2019 monsoon)\n"
        "- 2017-04-10 (2017 pre-harvest flood)"
    )
else:
    # Default: most recent 12-day window
    flood_end_str = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    st.sidebar.info(f"Using current window ending **{flood_end_str}**")

show_layers = st.sidebar.multiselect(
    "Map layers to display",
    options=["VV Reference (dry)", "VV At-flood", "Change image",
             "Flood mask (Otsu)", "Permanent water (JRC)"],
    default=["Flood mask (Otsu)", "Change image"],
)

if st.sidebar.button("🔄 Refresh / Recompute", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:11px;color:#888;text-align:center;line-height:1.7'>"
    "🌊 <b>HaorFloodAlert v2.0</b><br>"
    "© 2026 Salma Hoque Talukdar Koli<br>"
    "RTM Al-Kabir Technical University<br>"
    "CSE Thesis Project"
    "</div>",
    unsafe_allow_html=True,
)

# GEE computation
with st.spinner("Computing Otsu flood mask from Sentinel-1 SAR … (first load ~30s)"):
    result = compute_flood_layers(flood_end_str)

# Error state
if result.get("error"):
    st.error(f"GEE error: {result['error']}")
    st.info(
        "Showing base map only. Common fixes:\n"
        "- Choose a different date (Sentinel-1 has 6-day revisit over Bangladesh)\n"
        "- Click Refresh in the sidebar\n"
        "- Check GEE authentication: run `earthengine authenticate` in terminal"
    )
    result = {"urls": {}, "flood_area": 0, "threshold": "—",
              "ref_period": "—", "flood_period": "—",
              "n_ref": 0, "n_flood": 0, "upstream_vv": None,
              "hist_counts": [], "hist_means": []}

# Statistics bar
flood_area  = result.get("flood_area", 0)
threshold   = result.get("threshold", "—")
upstream_vv = result.get("upstream_vv")
n_flood     = result.get("n_flood", 0)
n_ref       = result.get("n_ref", 0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Flood Area (Otsu)", f"{flood_area} km²",
            help="SAR-detected inundation area after Otsu threshold + JRC permanent water removal")
col2.metric("Otsu Threshold", f"{threshold} dB",
            help="Auto-computed from Otsu bi-modal histogram. Typical haor floods: 3–6 dB change")
col3.metric("At-flood S1 images", str(n_flood),
            help=f"Sentinel-1 images in flood window ({result.get('flood_period','')})")
col4.metric("Reference S1 images", str(n_ref),
            help=f"Dry-season reference ({result.get('ref_period','')})")

if upstream_vv is not None:
    up_label = "⚠️ elevated" if upstream_vv < -16 else "✅ normal"
    col5.metric("Upstream Barak VV", f"{upstream_vv} dB", delta=up_label,
                delta_color="inverse" if upstream_vv < -16 else "off",
                help="Sentinel-1 over Silchar/Assam — barrage release proxy (~36h lead time)")
else:
    col5.metric("Upstream Barak VV", "unavailable")

# Upstream flood alert
if upstream_vv is not None and upstream_vv < -16:
    st.warning(
        "⚠️ **Upstream flood signal detected** — Barak river (Assam) VV is below −16 dB.  \n"
        "**বাংলা:** উজানের সংকেত — বরাক নদীতে পানি বেড়েছে।  \n"
        f"Current upstream VV: **{upstream_vv} dB** | Expected haor arrival: **~36 hours**"
    )

if isinstance(flood_area, (int, float)) and flood_area > 200:
    st.error(
        f"🚨 **Major inundation detected — {flood_area} km² flooded**  \n"
        "বন্যার বিস্তার স্বাভাবিকের চেয়ে অনেক বেশি। সতর্ক থাকুন।"
    )

st.divider()

# Map
m = build_map(result, show_layers)
st_folium(m, width="100%", height=660, returned_objects=[])

st.divider()

# Method explanation
with st.expander("🔬 Methodology — Otsu Change Detection (for thesis Chapter 4)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Algorithm: SAR Bi-modal Otsu Change Detection**

| Step | Operation | Detail |
|---|---|---|
| 1 | Reference composite | Jan–Feb {result.get('ref_period','')[:4]} S1 VV median |
| 2 | At-flood composite | {result.get('flood_period','')} S1 VV median |
| 3 | Speckle filter | Focal mean, 50 m radius, circle kernel |
| 4 | Change image | Δ VV = reference_VV − flood_VV (dB) |
| 5 | Otsu threshold | **{threshold} dB** (auto, maximises inter-class variance) |
| 6 | Flood mask | Pixels where Δ VV > {threshold} dB |
| 7 | Permanent water | JRC GSW ≥10 months/year → excluded |
| 8 | Flood area | {flood_area} km² (pixel area sum at 20 m) |

**Why Otsu (not fixed -15 dB):**
Haor dry-land VV baseline varies seasonally (−10 to −14 dB). A fixed threshold
would either miss real floods or generate false alarms depending on season. Otsu
finds the statistically optimal split from the actual histogram of each image pair.
        """)
    with c2:
        if result.get("hist_counts") and result.get("hist_means"):
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result["hist_means"],
                y=result["hist_counts"],
                marker_color=[
                    "#FF4B4B" if m > (threshold if isinstance(threshold, float) else 3.0)
                    else "#5b9bd5"
                    for m in result["hist_means"]
                ],
                name="Pixel count",
            ))
            if isinstance(threshold, float):
                fig.add_vline(
                    x=threshold, line_color="red", line_dash="dash",
                    annotation_text=f"Otsu = {threshold:.1f} dB",
                    annotation_position="top right",
                )
            fig.update_layout(
                title="SAR Change Image Histogram (pre − flood VV, dB)",
                xaxis_title="VV Change (dB) — positive = water onset",
                yaxis_title="Pixel count",
                height=300,
                showlegend=False,
            )
            fig.add_annotation(
                text="Blue = unchanged  |  Red = flooded (> Otsu threshold)",
                xref="paper", yref="paper", x=0.5, y=-0.20,
                showarrow=False, font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Histogram unavailable — GEE data not loaded.")

    st.markdown("""
**References (both cited in thesis proposal Section 4):**
- Uddin, K., Matin, M.A., Meyer, F.J. (2019). Multi-temporal Sentinel-1 SAR based operational
  flood mapping: A case study in Bangladesh. *Remote Sensing*, 11(13), 1581.
- Singha, M., et al. (2020). Flood and flood impacted paddy rice mapping using Sentinel-1 imagery
  on Google Earth Engine in Bangladesh. *ISPRS Journal*, 166, 278–293.

**GEE assets used:** `COPERNICUS/S1_GRD` · `JRC/GSW1_4/GlobalSurfaceWater`
    """)

# Legend
with st.expander("🗺️ Map Legend"):
    st.markdown("""
    | Symbol | Meaning |
    |---|---|
    | 🔵 **Blue overlay** | Flood extent — SAR change > Otsu threshold, permanent water excluded |
    | 🟦 **Cyan overlay** | Permanent water (JRC) — present ≥10 months/year, not counted as flood |
    | ⬛ **Grayscale** | Sentinel-1 VV backscatter (darker = lower dB = more water) |
    | 🟠 **Orange border** | Study area bounding box (91.35–91.55°E, 24.75–25.00°N) |
    | 🔵 **Blue polygon** | Tanguar Haor (northern basin) |
    | 🔴 **Red dots** | High-risk haor communities (6 upazilas) |
    | 🔵 **Blue marker** | Study centre — Sunamganj (91.45°E, 24.87°N) |

    **Change image colour scale:**
    Dark blue (−4 dB) → White (0 dB) → Dark red (+8 dB)
    Dark red = large VV decrease = flooded. White = no change. Blue = VV increased (unlikely flood).
    """)

# Data transparency
st.caption(
    "**Method:** Sentinel-1 SAR Change Detection (Otsu Bi-modal Thresholding) — "
    "Uddin et al. (2019) *Remote Sensing* 11(13):1581; Singha et al. (2020) *ISPRS J.* 166:278–293  |  "
    f"Data: Copernicus Sentinel-1 GRD via GEE ({GEE_PROJECT})  |  "
    f"Reference: {result.get('ref_period','—')}  |  "
    f"Flood window: {result.get('flood_period','—')}  |  "
    "JRC GSW v4 permanent water mask  |  20 m pixel  |  Cached 1 h"
)
