"""
utils/boro_damage.py
=====================
Boro rice flood-duration and damage-estimation model, extracted from
streamlit_app/pages/6_CropDamage.py so it can be reused outside the
Streamlit app (e.g. eval/run_journal_extension.py damage scenarios).
6_CropDamage.py imports these functions/constants from here; the formulas
live in exactly one place.

DATA HONESTY (unchanged from the original page docstring):
  - Yield constants (45 t/km^2, 28k BDT/t): from BRRI/BBS 2024 publications
  - Historical impact (2017/2019/2022/2024): ESTIMATED from BWDB reports (approximate)
  - Calculation model: empirical formula, not field-measured
  - Flood duration: calibrated from BWDB gauge records, not physics simulation
"""

import numpy as np
import pandas as pd

# Sunamganj Haor agricultural constants
# Source: Bangladesh Rice Research Institute (BRRI) 2024; BBS Agricultural Statistics
HAOR_AREA_KM2  = 8000.0
BORO_COVERAGE  = 0.72        # 72% of haor under boro rice (BRRI survey)
BORO_YIELD_KM2 = 45.0        # metric tons/km^2 = 4.5 t/ha (BRRI average)
BORO_PRICE_BDT = 28000.0     # BDT/ton (2024 market, BBS)
BORO_PRICE_USD = 255.0       # USD/ton (BDT/USD exchange rate ~110)

# Growth stage sensitivity table — BRRI 2022, Table 4.2
# month -> (max_sensitivity, stage_name, depth_threshold_cm, duration_threshold_days)
STAGE_TABLE = {
    10: (0.10, "Nursery / seedbed (নার্সারি / বীজতলা)",         8, 12),
    11: (0.15, "Transplanting (রোপণ)",                           6,  8),
    12: (0.20, "Vegetative growth (কায়িক বৃদ্ধি)",              7, 12),
     1: (0.30, "Booting / heading (বুটিং / শীষ বের হওয়া)",     6, 10),
     2: (0.45, "Panicle initiation (শীষ শুরু)",                  5,  7),
     3: (0.85, "Pre-harvest / grain fill (আগাম কাটার আগে)",     4,  5),
     4: (1.00, "Grain filling / harvest (দানা পূরণ / ফসল কাটা)", 3,  3),
     5: (0.70, "Harvest period (কাটার সময়)",                    4,  5),
}

# Upazila-level Boro rice acreage shares, Sunamganj district.
# Source: BBS Agricultural Statistics 2023; BRRI Upazila Boro Survey 2022;
# DAE Sunamganj district office (approximate, district-office-level figures,
# not a field census).
UPAZILA_BORO_AREA_KM2 = {
    "Tahirpur":         620,
    "Jamalganj":        580,
    "Derai":            400,
    "Dowarabazar":      450,
    "Shalla":           350,
    "Dharmapasha":      310,
    "Sunamganj Sadar":  280,
    "Bishwamvarpur":    240,
}


def estimate_duration(prob, soil, upvv, rain):
    """
    Empirical flood duration model calibrated from BWDB Sunamganj gauge data.
    Based on: peak probability, soil saturation, upstream signal, rainfall.
    """
    if prob < 0.10:
        return {"days": 0.0, "ci_low": 0.0, "ci_high": 0.0, "category": "No flood expected"}

    if prob >= 0.85:   base = 18 + prob * 20
    elif prob >= 0.65: base = 8  + prob * 15
    elif prob >= 0.40: base = 3  + prob * 8
    else:              base = 1  + prob * 4

    soil_mod     = 1 + (soil - 30) / 100
    upstream_mod = 1.3 if upvv < -16 else 1.0
    rain_mod     = 1 + rain / 500

    d = float(np.clip(base * soil_mod * upstream_mod * rain_mod, 0, 60))
    return {
        "days":     round(d, 1),
        "ci_low":   round(max(0, d * 0.7), 1),
        "ci_high":  round(d * 1.4, 1),
        "category": ("Prolonged (>21d)" if d > 21 else
                     "Extended (8–21d)" if d > 8  else "Short (<8d)"),
    }


def estimate_damage(area_km2, prob, dur_days, month, flood_depth_cm=50.0, sens_override=None):
    """
    BRRI-calibrated boro rice flood damage model (2022 methodology).
    Damage = f(growth stage sensitivity x depth factor x duration factor x flood probability).
    Source: Bangladesh Rice Research Institute (BRRI), Flood Impact Assessment Table 4.2 (2022).

    sens_override: if given, replaces the STAGE_TABLE sensitivity for `month`
    (stage_name/depth_thresh/days_thresh still come from the table). Used to
    propagate growth-stage-sensitivity uncertainty in scenario analyses
    without duplicating this function.
    """
    boro_in_flood = area_km2 * BORO_COVERAGE

    sens_max, stage_name, depth_thresh, days_thresh = STAGE_TABLE.get(
        month, (0.30, "Off-season / Aus-Aman", 7, 10)
    )
    if sens_override is not None:
        sens_max = sens_override

    if flood_depth_cm <= depth_thresh:
        depth_factor = 0.05
    elif flood_depth_cm <= 30:
        depth_factor = 0.05 + (flood_depth_cm - depth_thresh) / max(1, 30 - depth_thresh) * 0.35
    elif flood_depth_cm <= 80:
        depth_factor = 0.40 + (flood_depth_cm - 30) / 50 * 0.40
    else:
        depth_factor = min(1.0, 0.80 + (flood_depth_cm - 80) / 120 * 0.20)

    if dur_days <= days_thresh:
        duration_factor = 0.25
    elif dur_days <= 14:
        duration_factor = 0.25 + (dur_days - days_thresh) / max(1, 14 - days_thresh) * 0.55
    else:
        duration_factor = min(1.0, 0.80 + (dur_days - 14) / 20 * 0.20)

    base_rate = sens_max * depth_factor * duration_factor
    eff_rate  = min(1.0, base_rate * min(prob * 1.15, 1.0))

    # ±20% uncertainty range (BRRI field variability estimate across haor sub-basins)
    rate_low  = max(0.0, eff_rate * 0.80)
    rate_high = min(1.0, eff_rate * 1.20)

    damaged_km2 = boro_in_flood * eff_rate
    lost_tons   = damaged_km2 * BORO_YIELD_KM2
    loss_bdt    = lost_tons * BORO_PRICE_BDT
    loss_usd    = lost_tons * BORO_PRICE_USD
    farmers     = int(damaged_km2 * 100 * 2.3)   # 2.3 families/hectare (BRRI census)

    if eff_rate >= 0.70:
        msg   = f"🔴 CRITICAL — {stage_name}: Catastrophic crop loss ({eff_rate*100:.0f}%)"
        color = "#FF4B4B"
    elif eff_rate >= 0.40:
        msg   = f"🟠 SEVERE — {stage_name}: Major crop loss ({eff_rate*100:.0f}%)"
        color = "#FF8C00"
    elif eff_rate >= 0.15:
        msg   = f"🟡 MODERATE — {stage_name}: Significant crop loss ({eff_rate*100:.0f}%)"
        color = "#FFD700"
    else:
        msg   = f"🟢 LOW — {stage_name}: Minimal crop loss ({eff_rate*100:.0f}%)"
        color = "#00C49A"

    return {
        "boro_in":      round(boro_in_flood, 1),
        "damaged":      round(damaged_km2, 1),
        "rate_pct":     round(eff_rate * 100, 1),
        "rate_low":     round(rate_low * 100, 1),
        "rate_high":    round(rate_high * 100, 1),
        "lost_tons":    round(lost_tons, 0),
        "bdt_crore":    round(loss_bdt / 1e7, 2),
        "usd_million":  round(loss_usd / 1e6, 2),
        "farmers":      farmers,
        "msg":          msg,
        "color":        color,
        "stage_name":   stage_name,
        "depth_factor": round(depth_factor, 2),
        "dur_factor":   round(duration_factor, 2),
        "sens_max":     round(sens_max * 100, 0),
    }


def upazila_breakdown(damaged_km2: float) -> pd.DataFrame:
    """
    Splits a district-total damaged-area figure across Sunamganj's 8 upazilas
    in proportion to each upazila's share of total Boro acreage
    (UPAZILA_BORO_AREA_KM2). Same logic as the table in
    streamlit_app/pages/6_CropDamage.py, extracted here so both that page and
    eval scripts use one implementation.
    """
    df = pd.DataFrame({
        "upazila":        list(UPAZILA_BORO_AREA_KM2.keys()),
        "boro_area_km2":  list(UPAZILA_BORO_AREA_KM2.values()),
    })
    total_boro = df["boro_area_km2"].sum()
    df["share_pct"]        = (df["boro_area_km2"] / total_boro * 100).round(1)
    df["damaged_km2"]      = (df["share_pct"] / 100 * damaged_km2).round(1)
    df["rice_lost_tons"]   = (df["damaged_km2"] * BORO_YIELD_KM2).round(0).astype(int)
    df["loss_crore_bdt"]   = (df["rice_lost_tons"] * BORO_PRICE_BDT / 1e7).round(2)
    df["families_affected"] = (df["damaged_km2"] * 100 * 2.3).astype(int)
    return df
