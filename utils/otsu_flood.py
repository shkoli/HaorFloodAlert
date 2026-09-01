"""
utils/otsu_flood.py
====================
Sentinel-1 SAR Otsu change-detection flood extent — the core GEE computation,
extracted from streamlit_app/pages/2_Map.py so it can be reused outside the
Streamlit app (e.g. by eval scripts computing flood extent for historical
events). 2_Map.py imports compute_flood_extent() from here and adds its own
tile-URL / caching layer on top; the Otsu algorithm and GEE pipeline live in
exactly one place.

Method: Uddin et al. (2019) Remote Sensing 11(13):1581;
        Singha et al. (2020) ISPRS Journal 166:278-293.
"""

import numpy as np
import ee

from config import GEE_PROJECT, HAOR_BBOX, UPSTREAM_BBOX
from datetime import datetime, timedelta


def otsu_threshold(counts: list, means: list) -> float:
    """
    Otsu's method: find the threshold that maximises inter-class variance.
    Applied to the SAR change histogram (pre_VV - flood_VV, in dB).
    Positive change > threshold -> flooded pixel.

    Reference: Otsu (1979) IEEE Trans. Systems, Man, Cybernetics 9(1):62-66.
    """
    counts = np.array(counts, dtype=np.float64)
    means = np.array(means, dtype=np.float64)

    total = counts.sum()
    if total == 0:
        return 3.0  # fallback: 3 dB change

    sum_total = (counts * means).sum()
    w_b, sum_b, best_var, best_thresh = 0.0, 0.0, 0.0, float(means[0])

    for i in range(len(counts)):
        w_b += counts[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += counts[i] * means[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * (m_b - m_f) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = means[i]

    return float(best_thresh)


def _s1_composite(haor, start: str, end: str):
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
    return raw.focal_mean(50, "circle", "meters")


def _count_s1(haor, start: str, end: str) -> int:
    return int(
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(haor).filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW")).size().getInfo()
    )


def compute_flood_extent(flood_end_str: str, lookback_days: int = 12) -> dict:
    """
    Full Otsu change-detection pipeline (no tile URLs, no Streamlit caching —
    just the numeric result). Same 8-step algorithm as
    streamlit_app/pages/2_Map.py::compute_flood_layers:

      1. Dry-season reference composite (Jan-Feb of the relevant year)
      2. At-flood composite (`lookback_days` days before flood_end_str)
      3. Speckle filter both (focal mean, 50 m)
      4. Change image: reference_VV - flood_VV (positive = VV dropped = water)
      5. Otsu threshold from the live histogram
      6. Flood mask = change > threshold
      7. Remove permanent water (JRC GSW >=10 months/year)
      8. Flood area (km^2)

    Returns a dict with keys: error, flood_area_km2, threshold_db, n_ref,
    n_flood, ref_period, flood_period, upstream_vv.
    """
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        pass  # already initialized

    haor = ee.Geometry.Rectangle(HAOR_BBOX)
    upstream = ee.Geometry.Rectangle(UPSTREAM_BBOX)

    flood_end = datetime.strptime(flood_end_str, "%Y-%m-%d").date()
    flood_start = flood_end - timedelta(days=lookback_days)
    flood_start_str = str(flood_start)

    ref_year = flood_end.year if flood_end.month >= 3 else flood_end.year - 1
    ref_start = f"{ref_year}-01-01"
    ref_end = f"{ref_year}-02-28"

    n_ref = _count_s1(haor, ref_start, ref_end)
    n_flood = _count_s1(haor, flood_start_str, flood_end_str)

    if n_flood == 0 or n_ref == 0:
        return {
            "error": f"No Sentinel-1 images: n_ref={n_ref}, n_flood={n_flood} "
                     f"for window {flood_start_str} -> {flood_end_str}",
            "flood_area_km2": None,
        }

    s1_ref = _s1_composite(haor, ref_start, ref_end)
    s1_flood = _s1_composite(haor, flood_start_str, flood_end_str)

    change = s1_ref.select("VV").subtract(s1_flood.select("VV")).rename("change")

    hist_info = change.reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=128, minBucketWidth=0.1),
        geometry=haor, scale=20, maxPixels=int(1e8),
    ).getInfo()
    hist = hist_info.get("change", {})
    counts = hist.get("histogram", [])
    means = hist.get("bucketMeans", [])

    if len(counts) > 5 and len(means) > 5:
        threshold = otsu_threshold(counts, means)
    else:
        threshold = 3.0
    threshold = max(threshold, 2.0)

    raw_flood = change.gt(threshold)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("seasonality")
    permanent_water = jrc.gte(10)
    flood_mask = raw_flood.where(permanent_water, 0).selfMask()

    area_km2_img = flood_mask.multiply(ee.Image.pixelArea()).divide(1e6)
    area_val = area_km2_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=haor, scale=20, maxPixels=int(1e8)
    ).getInfo().get("change", 0) or 0

    s1_up = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(upstream)
        .filterDate(flood_start_str, flood_end_str)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select(["VV"]).median().clip(upstream)
    )
    up_vv_val = s1_up.select("VV").reduceRegion(ee.Reducer.mean(), upstream, 20).getInfo().get("VV")
    upstream_vv = round(float(up_vv_val), 2) if up_vv_val else None

    return {
        "error": None,
        "flood_area_km2": round(float(area_val), 1),
        "threshold_db": round(threshold, 2),
        "n_ref": n_ref,
        "n_flood": n_flood,
        "ref_period": f"{ref_start} -> {ref_end}",
        "flood_period": f"{flood_start_str} -> {flood_end_str}",
        "upstream_vv": upstream_vv,
    }
