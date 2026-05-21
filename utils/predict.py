"""
utils/predict.py — HaorFloodAlert ML prediction helpers.

3-Layer prediction architecture
--------------------------------
Layer 1  ML ensemble (RF + XGBoost)  — base probability from 13 satellite/met features
Layer 2  Current discharge level      — DANGER/HIGH Barak discharge adjustment
Layer 3  Discharge trend              — rising-trend adjustment (reliability-gated)

Safeguards
----------
- _MAX_DISCHARGE_ADJ = 0.30  : L2 + L3 combined cannot exceed 30 percentage points,
  preventing upstream discharge from overpowering the ML satellite/rainfall signal.
- _MAX_FINAL_PROB    = 0.95  : No window probability is ever reported as 100%, preserving
  uncertainty even under simultaneous DANGER discharge + very-fast trend.
- is_trend_reliable flag     : If analyze_discharge_trend() flags R² < 0.60 or
  |slope| < 100 m³/s/day, Layer 3 is suppressed (adj_trend = 0).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    FEATURES, RF_WEIGHT, XGB_WEIGHT,
    UPSTREAM_DISCHARGE_THRESHOLD_DANGER,
    UPSTREAM_DISCHARGE_THRESHOLD_HIGH,
)

# ── Safety constants ──────────────────────────────────────────────────────────
_MAX_DISCHARGE_ADJ = 0.30   # L2+L3 combined cap — prevents discharge from
                             # overpowering the ML rainfall/satellite signal
_MAX_FINAL_PROB    = 0.95   # Never output 100% certainty from discharge alone


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensemble_prob(feat_dict: dict, rf, xgb, active_feats: list) -> float:
    """
    Run the RF+XGB two-model ensemble on a single feature dict.
    Weights normalised to sum=1 (LSTM excluded in batch forecast mode).
    Returns float in [0, 1].
    """
    feat_row = [feat_dict.get(f, 0.0) for f in FEATURES]
    df       = pd.DataFrame([feat_row], columns=list(FEATURES))
    df_act   = df[active_feats]

    rf_p  = float(rf.predict_proba(df_act)[0][1])
    xgb_p = float(xgb.predict_proba(df_act)[0][1])

    w_sum = RF_WEIGHT + XGB_WEIGHT
    return float(RF_WEIGHT / w_sum * rf_p + XGB_WEIGHT / w_sum * xgb_p)


def _apply_3layer(base_prob: float,
                  discharge: float,
                  trend: float,
                  is_trend_reliable: bool = True) -> dict:
    """
    Compute the 3-layer probability adjustment for one forecast window.

    Layer 2 — Current discharge level:
        DANGER (≥7 500 m³/s) → +15 pp
        HIGH   (≥6 000 m³/s) → +10 pp
        otherwise             →  0 pp

    Layer 3 — Discharge trend (m³/s/day):
        > 500  (rising very fast) → +15 pp    ← suppressed if not reliable
        > 300  (rising fast)      → +10 pp
        > 100  (rising slowly)    → + 5 pp
        otherwise / unreliable    →  0 pp

    Safeguards applied inside this function:
        - L2 + L3 combined capped at _MAX_DISCHARGE_ADJ (30 pp).
          Priority: L2 (more certain) fills first; L3 gets remainder.
        - Final probability capped at _MAX_FINAL_PROB (95%).

    Parameters
    ----------
    is_trend_reliable : bool — if False, adj_trend is forced to 0.

    Returns
    -------
    {
        "base_prob":         float,   # Layer 1 ML-only probability
        "adj_current":       float,   # Layer 2 delta (after cap)
        "adj_trend":         float,   # Layer 3 delta (after cap; 0 if unreliable)
        "raw_adj_current":   float,   # Layer 2 delta before cap
        "raw_adj_trend":     float,   # Layer 3 delta before cap
        "final_prob":        float,   # base + L2 + L3, clamped ≤ 0.95
        "cap_applied":       bool,    # True if 30 pp or 95% cap was triggered
        "trend_suppressed":  bool,    # True if L3 was zeroed due to reliability
        "l2_reason":         str,
        "l3_reason":         str,
    }
    """
    # ── Layer 2: current discharge level ─────────────────────────────────────
    if discharge >= UPSTREAM_DISCHARGE_THRESHOLD_DANGER:
        raw_l2  = 0.15
        l2_reason = (f"DANGER discharge ({discharge:,.0f} m³/s ≥ "
                     f"{UPSTREAM_DISCHARGE_THRESHOLD_DANGER:,}) → +15 pp")
    elif discharge >= UPSTREAM_DISCHARGE_THRESHOLD_HIGH:
        raw_l2  = 0.10
        l2_reason = (f"HIGH discharge ({discharge:,.0f} m³/s ≥ "
                     f"{UPSTREAM_DISCHARGE_THRESHOLD_HIGH:,}) → +10 pp")
    else:
        raw_l2  = 0.0
        l2_reason = f"Normal/rising discharge ({discharge:,.0f} m³/s) → no L2 adjustment"

    # ── Layer 3: trend — suppressed if unreliable ────────────────────────────
    trend_suppressed = False
    if not is_trend_reliable:
        raw_l3    = 0.0
        l3_reason = "Layer 3 suppressed — trend unreliable (R² < 0.60 or |slope| < 100 m³/s/day)"
        trend_suppressed = True
    elif trend > 500:
        raw_l3    = 0.15
        l3_reason = f"Rising very fast ({trend:+.0f} m³/s/day) → +15 pp"
    elif trend > 300:
        raw_l3    = 0.10
        l3_reason = f"Rising fast ({trend:+.0f} m³/s/day) → +10 pp"
    elif trend > 100:
        raw_l3    = 0.05
        l3_reason = f"Rising slowly ({trend:+.0f} m³/s/day) → +5 pp"
    else:
        raw_l3    = 0.0
        l3_reason = f"Stable/falling ({trend:+.0f} m³/s/day) → no L3 adjustment"

    # ── Apply combined cap: L2 + L3 ≤ 30 pp ─────────────────────────────────
    # Priority: L2 fills first (discharge level is more certain than trend).
    adj_current = min(raw_l2, _MAX_DISCHARGE_ADJ)
    adj_trend   = min(raw_l3, max(0.0, _MAX_DISCHARGE_ADJ - adj_current))
    adj_cap_hit = (raw_l2 + raw_l3) > _MAX_DISCHARGE_ADJ

    # ── Apply final probability ceiling ──────────────────────────────────────
    raw_final  = base_prob + adj_current + adj_trend
    final      = round(min(_MAX_FINAL_PROB, max(0.0, raw_final)), 4)
    prob_capped = raw_final > _MAX_FINAL_PROB

    return {
        "base_prob":        round(base_prob,    4),
        "adj_current":      round(adj_current,  4),
        "adj_trend":        round(adj_trend,    4),
        "raw_adj_current":  round(raw_l2,       4),
        "raw_adj_trend":    round(raw_l3,       4),
        "final_prob":       final,
        "cap_applied":      adj_cap_hit or prob_capped,
        "trend_suppressed": trend_suppressed,
        "l2_reason":        l2_reason,
        "l3_reason":        l3_reason,
    }


def _generate_explanation(base_prob: float,
                          adj_current: float,
                          adj_trend: float,
                          final_prob: float,
                          discharge: float,
                          trend: float,
                          trend_suppressed: bool = False) -> str:
    """Plain-English explanation of the 3-layer prediction result."""
    low_rain = base_prob < 0.40

    if low_rain and adj_current >= 0.10 and adj_trend >= 0.05:
        return (
            f"Satellite/rainfall conditions alone show low risk "
            f"({base_prob*100:.0f}%), but upstream Barak discharge "
            f"({discharge:,.0f} m³/s) is high and rising fast "
            f"({trend:+.0f} m³/s/day). Flood likely in 24–36 h "
            f"from upstream water release."
        )
    if low_rain and adj_current >= 0.10:
        return (
            f"Satellite/rainfall conditions show low risk ({base_prob*100:.0f}%), "
            f"but upstream Barak discharge ({discharge:,.0f} m³/s) is elevated. "
            f"Monitor closely — barrage release may trigger haor flooding."
        )
    if not low_rain and adj_current == 0 and adj_trend == 0:
        note = " (Layer 3 suppressed — trend unreliable)" if trend_suppressed else ""
        return (
            f"Satellite and rainfall conditions indicate {base_prob*100:.0f}% probability. "
            f"Upstream discharge is normal — ML estimate is the primary signal{note}."
        )

    parts = [f"ML base: {base_prob*100:.0f}%"]
    if adj_current > 0:
        parts.append(f"discharge ({discharge:,.0f} m³/s) adds {adj_current*100:.0f} pp")
    if adj_trend > 0:
        parts.append(f"rising trend adds {adj_trend*100:.0f} pp")
    elif trend_suppressed:
        parts.append("trend suppressed (insufficient reliability)")
    parts.append(f"combined: {final_prob*100:.0f}%")
    return ". ".join(parts) + "."


# ── Public API ────────────────────────────────────────────────────────────────

def predict_flood_72h(
    current_features: dict,
    forecast_breakdown: dict,
    rf,
    xgb,
    active_feats: list,
    current_discharge: float = 0.0,
    discharge_trend: float = 0.0,
    discharge_projections: list = None,
    is_trend_reliable: bool = True,
) -> dict:
    """
    Estimate flood probability for three 24-hour forecast windows using the
    3-layer model: ML ensemble (L1) + discharge level (L2) + trend (L3).

    For each window the function:
      • Replaces 'rainfall' with the window-specific forecast rain
        (scaled ×3 to approximate the 7-day CHIRPS equivalent).
      • Slightly bumps 'soil_moisture' to reflect cumulative wetting.
      • Applies Layer 2 (discharge level) using the projected discharge
        for that window (current → 24h → 48h projection).
      • Applies Layer 3 (trend) uniformly — suppressed if is_trend_reliable=False.

    Safeguards: L2+L3 capped at 30 pp; final probability capped at 95%.

    Parameters
    ----------
    current_features       : dict  — output of fetch_live() in 1_Prediction.py
    forecast_breakdown     : dict  — output of get_forecast_rainfall_72h()
    rf, xgb                : trained sklearn / xgboost models
    active_feats           : list[str]  — features used by the models
    current_discharge      : float — Barak discharge now (m³/s); 0 = no L2/L3
    discharge_trend        : float — m³/s/day (effective — already zeroed if unreliable)
    discharge_projections  : list[float] — [24h, 48h, 72h] projected discharge;
                             if None, uses current_discharge for all windows
    is_trend_reliable      : bool — from analyze_discharge_trend()["is_reliable"]

    Returns
    -------
    {
        "now":              float,
        "0_24h":            float,   # final probability (all 3 layers)
        "24_48h":           float,
        "48_72h":           float,
        "peak":             float,
        "peak_window":      str,
        "windows":          list[dict],
        "breakdown":        dict,    # peak-window 3-layer breakdown
        "any_cap_applied":  bool,    # True if any window hit 30pp or 95% cap
    }
    """
    if discharge_projections is None or len(discharge_projections) < 3:
        discharge_projections = [current_discharge] * 3

    base     = dict(current_features)
    now_prob = _ensemble_prob(base, rf, xgb, active_feats)

    bd       = forecast_breakdown.get("breakdown", {})
    total_72 = forecast_breakdown.get("total", base.get("forecast_rain_72h", 0.0))
    next_12  = bd.get("0_6h", 0.0) + bd.get("6_12h", 0.0)

    window_cfg = [
        {
            "label":     "0–24h",
            "rain_mm":   bd.get("0_6h", 0.0) + bd.get("6_12h", 0.0) + bd.get("12_24h", 0.0),
            "soil_bump": 0.0,
            "discharge": current_discharge,
        },
        {
            "label":     "24–48h",
            "rain_mm":   bd.get("24_48h", 0.0),
            "soil_bump": 2.0,
            "discharge": discharge_projections[0],
        },
        {
            "label":     "48–72h",
            "rain_mm":   bd.get("48_72h", 0.0),
            "soil_bump": 4.0,
            "discharge": discharge_projections[1],
        },
    ]

    windows      = []
    any_cap      = False
    for cfg in window_cfg:
        feat = dict(base)
        feat["rainfall"]               = cfg["rain_mm"] * 3.0
        feat["forecast_rain_next_12h"] = next_12
        feat["forecast_rain_72h"]      = total_72
        feat["soil_moisture"]          = min(
            50.0, base.get("soil_moisture", 30.0) + cfg["soil_bump"]
        )
        ml_prob = _ensemble_prob(feat, rf, xgb, active_feats)
        lyr     = _apply_3layer(ml_prob, cfg["discharge"], discharge_trend, is_trend_reliable)
        if lyr["cap_applied"]:
            any_cap = True

        windows.append({
            "label":            cfg["label"],
            "prob":             lyr["final_prob"],
            "base_prob":        lyr["base_prob"],
            "adj_current":      lyr["adj_current"],
            "adj_trend":        lyr["adj_trend"],
            "raw_adj_current":  lyr["raw_adj_current"],
            "raw_adj_trend":    lyr["raw_adj_trend"],
            "discharge":        cfg["discharge"],
            "rain_mm":          cfg["rain_mm"],
            "cap_applied":      lyr["cap_applied"],
            "trend_suppressed": lyr["trend_suppressed"],
            "l2_reason":        lyr["l2_reason"],
            "l3_reason":        lyr["l3_reason"],
        })

    probs       = {w["label"]: w["prob"] for w in windows}
    peak_window = max(probs, key=probs.get)
    peak_w      = next(w for w in windows if w["label"] == peak_window)

    explanation = _generate_explanation(
        peak_w["base_prob"], peak_w["adj_current"], peak_w["adj_trend"],
        peak_w["prob"], peak_w["discharge"], discharge_trend,
        peak_w["trend_suppressed"],
    )

    breakdown = {
        "base_prob":        peak_w["base_prob"],
        "adj_current":      peak_w["adj_current"],
        "adj_trend":        peak_w["adj_trend"],
        "raw_adj_current":  peak_w["raw_adj_current"],
        "raw_adj_trend":    peak_w["raw_adj_trend"],
        "final_prob":       peak_w["prob"],
        "discharge":        peak_w["discharge"],
        "trend":            discharge_trend,
        "cap_applied":      peak_w["cap_applied"],
        "trend_suppressed": peak_w["trend_suppressed"],
        "l2_reason":        peak_w["l2_reason"],
        "l3_reason":        peak_w["l3_reason"],
        "explanation":      explanation,
    }

    return {
        "now":             round(now_prob, 4),
        "0_24h":           probs["0–24h"],
        "24_48h":          probs["24–48h"],
        "48_72h":          probs["48–72h"],
        "peak":            round(max(probs.values()), 4),
        "peak_window":     peak_window,
        "windows":         windows,
        "breakdown":       breakdown,
        "any_cap_applied": any_cap,
    }


def apply_discharge_adjustment(base_prob: float,
                                discharge: float,
                                trend: float,
                                is_trend_reliable: bool = True) -> dict:
    """
    Apply a heuristic post-model adjustment based on upstream Barak discharge
    for the *current-time* probability shown in the discharge monitoring section.
    Uses the same 3-layer logic (with same safeguards) as predict_flood_72h().

    This is NOT a learned ML parameter — physics-informed correction applied
    after the ensemble output.  Label the result clearly in the UI.

    Returns
    -------
    {
        "base_prob":        float,
        "adjusted_prob":    float,
        "adjustment":       float,
        "reason":           str,
        "cap_applied":      bool,
        "trend_suppressed": bool,
    }
    """
    lyr       = _apply_3layer(base_prob, discharge, trend, is_trend_reliable)
    total_adj = lyr["adj_current"] + lyr["adj_trend"]

    parts = []
    if lyr["adj_current"] > 0:
        parts.append(lyr["l2_reason"])
    if lyr["adj_trend"] > 0:
        parts.append(lyr["l3_reason"])
    if lyr["trend_suppressed"]:
        parts.append(lyr["l3_reason"])   # includes suppression note
    if lyr["cap_applied"]:
        parts.append(f"Adjustment capped at {_MAX_DISCHARGE_ADJ*100:.0f} pp / {_MAX_FINAL_PROB*100:.0f}%")
    reason = "; ".join(parts) if parts else "No discharge adjustment applied."

    return {
        "base_prob":        lyr["base_prob"],
        "adjusted_prob":    lyr["final_prob"],
        "adjustment":       round(total_adj, 4),
        "reason":           reason,
        "cap_applied":      lyr["cap_applied"],
        "trend_suppressed": lyr["trend_suppressed"],
    }
