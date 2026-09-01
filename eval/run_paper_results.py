"""
eval/run_paper_results.py
==========================
Regenerates every number the IEEE paper needs, from scratch, live, with no
hardcoded metrics anywhere. Whatever this script prints is the truth that
gets reported in the paper -- it is NOT tuned to reproduce any previously
published number (89.6% etc).

Outputs:
  results/paper_results.json       -- full machine-readable results
  results/paper_results.txt        -- human-readable mirror, with timestamps
  results/loocv_predictions.csv    -- per-event probabilities from the
                                       primary 77-event real-SAR LOOCV run
                                       (with augmentation, threshold 0.5)
  results/PAPER_NUMBERS_SUMMARY.md -- one table of every number the paper
                                       needs, each traceable to a JSON key
  gee_scripts/export_fig_sar_real.js -- GEE script to export REAL Sentinel-1
                                       VV composites for Fig 3.4 (run in the
                                       GEE code editor by hand -- this repo
                                       does not fabricate a substitute image)

Run:
  python eval/run_paper_results.py

This can take 15-30 minutes -- the ablation study alone runs 7 feature-arm
LOOCV loops of 131 folds each, with 8x augmentation and 500-tree/500-round
RF+XGB models per fold.
"""
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GEE_DIR = ROOT / "gee_scripts"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# ONE canonical model configuration -- single source of truth for this run.
# ─────────────────────────────────────────────────────────────────────────────
# RF/XGB hyperparameters: taken directly from the CURRENTLY SAVED models
# (models/rf_model.pkl / models/xgb_model.pkl), confirmed by loading them
# with joblib and reading .get_params(): RF max_depth=5, min_samples_split=2,
# class_weight='balanced'; XGB max_depth=4, learning_rate=0.05,
# n_estimators=500. Both models have n_estimators=500 and identical mtimes
# (2026-05-15 23:48:53), and these exact hyperparameters are hardcoded in
# train_honest.py (the script that produced them) -- NOT the values
# (max_depth=12/8, min_samples_split=5) previously claimed in the paper draft.
RF_PARAMS = dict(
    n_estimators=500, max_depth=5, min_samples_split=2,
    class_weight="balanced", random_state=SEED, n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=SEED, verbosity=0,
)
# scale_pos_weight is computed per-fold from the (post-augmentation) training
# labels, exactly as train_honest.py does -- it is data-derived, not a fixed
# hyperparameter, so it is not listed above.

# Ensemble combination: equal 50/50 RF+XGB average. This matches
# train_honest.py:85 (the script that actually produced the currently-saved
# models and the historically reported LOOCV numbers) -- NOT the 0.5625/0.4375
# renormalized weight used elsewhere for the dashboard's RF+XGB-only display.
# See PAPER_VERIFICATION.md Task 5 for the trace of this discrepancy.
ENSEMBLE_WEIGHTS = dict(rf=0.5, xgb=0.5)

# Augmentation: Gaussian noise added to training folds only (never to the
# held-out fold), factor and sigma below. CHOSEN BECAUSE: these are the exact
# values hardcoded in train_honest.py (AUG=8, NOISE=0.04), which is the
# script that produced the currently-saved models/rf_model.pkl and
# models/xgb_model.pkl (confirmed by identical mtimes and matching
# hyperparameters -- see PAPER_VERIFICATION.md). A different script,
# train_augmented.py, uses 10x/0.05 noise, but that script trains on a
# different, smaller, superseded dataset (real_training_data_v2.csv, 40
# rows) and did NOT produce the currently deployed models. We therefore use
# 8x/0.04 here, not the 10x/0.05 suggested as an example -- because 10x/0.05
# does not actually correspond to the saved-model-producing script once
# checked against file mtimes and pickled hyperparameters.
AUG_FACTOR = 8
NOISE_SIGMA = 0.04

# Decision thresholds to report side-by-side so the paper can choose one.
THRESHOLD_PRIMARY = 0.50
THRESHOLD_ALT = 0.40

# Feature list: EXACTLY models/active_features.pkl (10 features, no
# temp_anomaly/slope/twi -- those are excluded because they were found to be
# zero-variance or a confound source in the currently deployed model).
ALL_FEATS = [
    "VV", "VH", "vv_vh_ratio",
    "rainfall", "soil_moisture", "wind",
    "forecast_rain_next_12h", "ndwi", "upstream_vv", "forecast_rain_72h",
]
SAR_FEATS = ["VV", "VH", "vv_vh_ratio"]
RAIN_FEATS = ["rainfall", "forecast_rain_next_12h", "forecast_rain_72h"]
WEATHER_FEATS = ["rainfall", "soil_moisture", "wind",
                 "forecast_rain_next_12h", "forecast_rain_72h"]

HOLDOUT_SEEDS = [1, 2, 3, 4, 5]
HOLDOUT_TEST_SIZE = 0.40

LOG = []  # (timestamp, message) for the human-readable report


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────
def build_rf():
    return RandomForestClassifier(**RF_PARAMS)


def build_xgb(scale_pos_weight):
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)


def augment(Xtr, ytr, stds, rng):
    """Gaussian-noise augmentation applied to TRAINING data only."""
    rows, labs = [Xtr], [ytr]
    for _ in range(AUG_FACTOR - 1):
        rows.append(Xtr + rng.standard_normal(Xtr.shape) * stds * NOISE_SIGMA)
        labs.append(ytr)
    return np.vstack(rows), np.concatenate(labs)


def compute_metrics(trues, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(trues, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(trues, preds)),
        "precision": float(precision_score(trues, preds, zero_division=0)),
        "recall": float(recall_score(trues, preds, zero_division=0)),
        "f1": float(f1_score(trues, preds, zero_division=0)),
        "specificity": float(specificity),
        "n_correct": int((preds == trues).sum()),
        "n_total": int(len(trues)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def loocv_ensemble(df, feats, use_augmentation, seed=SEED):
    """RF+XGB 50/50 LOOCV. Returns per-event probs/trues plus metrics at
    both thresholds and AUC. Augmentation (if enabled) is applied to the
    training fold ONLY, never to the held-out row -- verified by construction
    since `augment()` is called on X[tr]/y[tr], and Xval/yval always come
    straight from the untouched original arrays."""
    X = df[feats].values.astype(float)
    y = df["flood_label"].values.astype(int)
    rng = np.random.default_rng(seed)
    stds = X.std(axis=0)
    stds = np.where(stds < 1e-6, 1e-6, stds)

    loo = LeaveOneOut()
    probs, trues = [], []
    for tr_idx, val_idx in loo.split(X):
        if use_augmentation:
            Xtr, ytr = augment(X[tr_idx], y[tr_idx], stds, rng)
        else:
            Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        n_pos = float((ytr == 1).sum())
        n_neg = float((ytr == 0).sum())
        spw = n_neg / max(n_pos, 1.0)

        rf = build_rf()
        xgb = build_xgb(spw)
        rf.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)

        p = (ENSEMBLE_WEIGHTS["rf"] * rf.predict_proba(Xval)[:, 1]
             + ENSEMBLE_WEIGHTS["xgb"] * xgb.predict_proba(Xval)[:, 1])
        probs.append(float(p[0]))
        trues.append(int(yval[0]))

    probs = np.array(probs)
    trues = np.array(trues)
    auc = float(roc_auc_score(trues, probs)) if len(set(trues)) > 1 else float("nan")

    return {
        "n_events": int(len(df)),
        "n_flood": int(y.sum()),
        "n_dry": int((y == 0).sum()),
        "augmentation": {"applied": use_augmentation, "factor": AUG_FACTOR if use_augmentation else None,
                          "sigma": NOISE_SIGMA if use_augmentation else None},
        "auc": auc,
        "metrics_threshold_0.50": compute_metrics(trues, probs, THRESHOLD_PRIMARY),
        "metrics_threshold_0.40": compute_metrics(trues, probs, THRESHOLD_ALT),
        "_probs": probs,
        "_trues": trues,
    }


def loocv_ablation_arm(df, feats, seed=SEED):
    """Same protocol as loocv_ensemble, but only returns accuracy/AUC/F1
    (used across many feature-arms in the ablation study)."""
    res = loocv_ensemble(df, feats, use_augmentation=True, seed=seed)
    m = res["metrics_threshold_0.50"]
    return {
        "n_features": len(feats),
        "features": feats,
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "specificity": m["specificity"],
        "auc": res["auc"],
        "confusion_matrix": m["confusion_matrix"],
    }


def five_fold_cv(df, feats, seed=SEED):
    X = df[feats].values.astype(float)
    y = df["flood_label"].values.astype(int)
    rng = np.random.default_rng(seed)
    stds = X.std(axis=0)
    stds = np.where(stds < 1e-6, 1e-6, stds)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accs, fold_aucs = [], []
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        Xtr, ytr = augment(X[tr_idx], y[tr_idx], stds, rng)
        Xval, yval = X[val_idx], y[val_idx]
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        rf = build_rf()
        xgb = build_xgb(spw)
        rf.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)
        p = 0.5 * rf.predict_proba(Xval)[:, 1] + 0.5 * xgb.predict_proba(Xval)[:, 1]
        preds = (p >= THRESHOLD_PRIMARY).astype(int)
        fold_accs.append(float(accuracy_score(yval, preds)))
        if len(set(yval)) > 1:
            fold_aucs.append(float(roc_auc_score(yval, p)))
        else:
            fold_aucs.append(float("nan"))

    return {
        "n_events": int(len(df)),
        "n_splits": 5,
        "fold_accuracy": fold_accs,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_auc": fold_aucs,
        "mean_auc": float(np.nanmean(fold_aucs)),
        "std_auc": float(np.nanstd(fold_aucs)),
    }


def stratified_holdout(df, feats, seeds, test_size):
    X = df[feats].values.astype(float)
    y = df["flood_label"].values.astype(int)
    per_seed = []
    for seed in seeds:
        Xtr0, Xte, ytr0, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)
        rng = np.random.default_rng(seed)
        stds = Xtr0.std(axis=0)
        stds = np.where(stds < 1e-6, 1e-6, stds)
        Xtr, ytr = augment(Xtr0, ytr0, stds, rng)
        spw = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)
        rf = build_rf()
        xgb = build_xgb(spw)
        rf.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)
        p = 0.5 * rf.predict_proba(Xte)[:, 1] + 0.5 * xgb.predict_proba(Xte)[:, 1]
        m = compute_metrics(yte, p, THRESHOLD_PRIMARY)
        auc = float(roc_auc_score(yte, p)) if len(set(yte)) > 1 else float("nan")
        per_seed.append({"seed": seed, "accuracy": m["accuracy"], "auc": auc,
                          "precision": m["precision"], "recall": m["recall"], "f1": m["f1"]})

    accs = [s["accuracy"] for s in per_seed]
    aucs = [s["auc"] for s in per_seed]
    return {
        "n_seeds": len(seeds),
        "test_size": test_size,
        "per_seed": per_seed,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_auc": float(np.nanmean(aucs)),
        "std_auc": float(np.nanstd(aucs)),
    }


def baseline_majority(df, feats):
    """Majority-class LOOCV baseline: predict the majority label of the
    training fold (always predicts 0 or always 1, whichever the training
    fold favors). AUC is intentionally NOT computed from the training-fold
    positive-class proportion: because removing one row from a small n=77
    fold slightly shifts that proportion depending on which class was
    removed, using it as a ranking score produces a spurious near-perfect
    ANTI-correlation with the true label under LOOCV (an artifact of the
    leave-one-out mechanics, not a real classifier property). A constant
    predictor has no meaningful ranking, so AUC is reported as null with an
    explanation rather than a misleading number."""
    y = df["flood_label"].values.astype(int)
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr_idx, val_idx in loo.split(y):
        majority_label = int(round(y[tr_idx].mean()))
        preds.append(majority_label)
        trues.append(int(y[val_idx][0]))
    preds = np.array(preds, dtype=float)
    trues = np.array(trues)
    m = compute_metrics(trues, preds, THRESHOLD_PRIMARY)
    return {
        "auc": None,
        "auc_note": ("Not computed: a constant majority-class predictor has no "
                     "meaningful ranking score; using the training-fold class "
                     "proportion as a pseudo-probability produces a spurious "
                     "near-perfect anti-correlation under LOOCV on small n, which "
                     "would be misleading to report as an AUC value."),
        **m,
    }


def baseline_rainfall_rule(df):
    """Rainfall-threshold LOOCV baseline: for each fold, pick the rainfall
    cutoff (from candidate values in the training fold only) that maximizes
    training-fold accuracy, then apply it to the held-out event. AUC uses the
    raw rainfall value as the ranking score (valid for a monotonic rule)."""
    x = df["rainfall"].values.astype(float)
    y = df["flood_label"].values.astype(int)
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr_idx, val_idx in loo.split(x):
        xtr, ytr = x[tr_idx], y[tr_idx]
        candidates = np.unique(xtr)
        best_thr, best_acc = candidates[0], -1.0
        for thr in candidates:
            acc = accuracy_score(ytr, (xtr > thr).astype(int))
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        preds.append(int(x[val_idx][0] > best_thr))
        trues.append(int(y[val_idx][0]))
    preds = np.array(preds)
    trues = np.array(trues)
    m = compute_metrics(trues, preds.astype(float), 0.5)
    auc = float(roc_auc_score(trues, x)) if len(set(trues)) > 1 else float("nan")
    return {"auc_using_raw_rainfall_as_score": auc, **m}


def baseline_logreg(df, feats, seed=SEED):
    X = df[feats].values.astype(float)
    y = df["flood_label"].values.astype(int)
    rng = np.random.default_rng(seed)
    stds = X.std(axis=0)
    stds = np.where(stds < 1e-6, 1e-6, stds)
    loo = LeaveOneOut()
    probs, trues = [], []
    for tr_idx, val_idx in loo.split(X):
        Xtr, ytr = augment(X[tr_idx], y[tr_idx], stds, rng)
        Xval, yval = X[val_idx], y[val_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xval_s = scaler.transform(Xtr), scaler.transform(Xval)
        clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
        clf.fit(Xtr_s, ytr)
        p = clf.predict_proba(Xval_s)[:, 1]
        probs.append(float(p[0]))
        trues.append(int(yval[0]))
    probs = np.array(probs)
    trues = np.array(trues)
    m = compute_metrics(trues, probs, THRESHOLD_PRIMARY)
    auc = float(roc_auc_score(trues, probs)) if len(set(trues)) > 1 else float("nan")
    return {"auc": auc, **m}


def compute_deconfounding(df):
    r_temp_label, p1 = pearsonr(df["temp"].values.astype(float), df["flood_label"].values.astype(float))
    month = pd.to_datetime(df["date"]).dt.month.values.astype(float)
    r_temp_month, p2 = pearsonr(df["temp"].values.astype(float), month)
    r_anom_label, p3 = pearsonr(df["temp_anomaly"].values.astype(float), df["flood_label"].values.astype(float))
    return {
        "n_events": int(len(df)),
        "raw_temp_vs_label": {"r": float(r_temp_label), "p_value": float(p1)},
        "raw_temp_vs_month": {"r": float(r_temp_month), "p_value": float(p2)},
        "temp_anomaly_vs_label": {"r": float(r_anom_label), "p_value": float(p3)},
    }


def compute_ffwc_stats():
    xlsx_path = DATA_DIR / "Sunamganj (SW269) WL.xlsx"
    if not xlsx_path.exists():
        return {
            "status": "MISSING",
            "message": ("The SW269 gauge file was not found at "
                        f"{xlsx_path}. Re-download it before these claims "
                        "can stay in the paper -- no statistics were computed."),
        }

    wl = pd.read_excel(xlsx_path)
    wl["date_time"] = pd.to_datetime(wl["date_time"])
    n_readings = int(len(wl))

    # The raw gauge file has no flood/dry label of its own. To compute a
    # flood-vs-dry split we join its daily mean water level to the
    # flood_label already assigned (from FFWC/BWDB reports, independent of
    # SAR features) in the training dataset, matched by calendar date. This
    # is a live, documented join -- not a hardcoded number.
    wl["date_only"] = wl["date_time"].dt.date
    daily = wl.groupby("date_only")["water_level"].mean().reset_index()
    daily["date_only"] = pd.to_datetime(daily["date_only"])

    train = pd.read_csv(DATA_DIR / "honest_training_data_v2.csv")
    train["date"] = pd.to_datetime(train["date"])
    merged = train.merge(daily, left_on="date", right_on="date_only", how="inner")
    n_matched = int(len(merged))

    if n_matched < 10:
        return {
            "status": "INSUFFICIENT_OVERLAP",
            "n_readings": n_readings,
            "n_matched_to_labels": n_matched,
            "message": "Fewer than 10 dates overlap between the gauge file and the labeled event dataset; flood/dry split not computed.",
        }

    flood_wl = merged.loc[merged["flood_label"] == 1, "water_level"].values.astype(float)
    dry_wl = merged.loc[merged["flood_label"] == 0, "water_level"].values.astype(float)

    t_stat, t_p = ttest_ind(flood_wl, dry_wl, equal_var=False)
    r_pb, r_p = pointbiserialr(merged["flood_label"].values.astype(int), merged["water_level"].values.astype(float))

    # Naive best-threshold search (in-sample, descriptive statistic over the
    # matched rows -- NOT cross-validated; reported as such).
    candidates = np.unique(merged["water_level"].values.astype(float))
    best_thr, best_acc = candidates[0], -1.0
    for thr in candidates:
        acc = accuracy_score(merged["flood_label"].values.astype(int),
                              (merged["water_level"].values.astype(float) > thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    return {
        "status": "OK",
        "n_readings_total_in_gauge_file": n_readings,
        "n_dates_matched_to_labeled_events": n_matched,
        "note": ("Flood/dry split is derived by joining the gauge file's daily-mean "
                 "water level to flood_label (from FFWC/BWDB reports) on calendar "
                 "date. Only dates present in both files can be labeled -- this is "
                 f"{n_matched} of {len(train)} labeled events, not the full "
                 f"{n_readings}-reading series."),
        "flood_mean_water_level_m": float(np.mean(flood_wl)),
        "flood_n": int(len(flood_wl)),
        "dry_mean_water_level_m": float(np.mean(dry_wl)),
        "dry_n": int(len(dry_wl)),
        "t_test": {"t_statistic": float(t_stat), "p_value": float(t_p)},
        "point_biserial_r": {"r": float(r_pb), "p_value": float(r_p)},
        "naive_best_threshold_m": float(best_thr),
        "naive_threshold_accuracy": float(best_acc),
        "naive_threshold_is_in_sample_not_cross_validated": True,
    }


def write_sar_gee_script():
    """Writes a GEE script the user must run by hand in the Code Editor to
    export REAL Sentinel-1 VV composites for the pre-flood (Jan 2017) and
    flood (Apr 2017) windows. No synthetic substitute is generated."""
    content = '''/**
 * export_fig_sar_real.js
 * =======================
 * Exports REAL Sentinel-1 VV backscatter composites for the Haor AOI for a
 * pre-flood reference window (Jan 2017) and a flood window (Apr 2017), for
 * use as the two panels of thesis Figure 3.4. Adapted from
 * gee_scripts/01_sentinel1_flood_detection.js, generalized to export the
 * raw VV composites themselves (not just a change-detection mask), because
 * the figure needs the actual backscatter panels, in dB, with a shared
 * colorbar -- not a flood mask.
 *
 * How to use:
 *   1. Paste this script into https://code.earthengine.google.com/
 *   2. Click Run
 *   3. Two Export.image.toDrive tasks appear in the "Tasks" tab on the
 *      right -- click "Run" on each to start the export to your Drive
 *      folder "HaorFloodAlert_Fig3_4"
 *   4. Download both GeoTIFFs from Drive once the exports finish
 *   5. Build the two-panel figure locally (e.g. with matplotlib/rasterio):
 *        - plot both rasters with the SAME vmin/vmax (suggest -25 to 0 dB,
 *          matching the visualization params below) so the colorbar is
 *          shared and comparable between panels
 *        - add a single shared colorbar labeled "VV backscatter (dB)"
 *        - target 300+ DPI, sized for an 8.9 cm (3.5 in) IEEE column width
 *          e.g. figsize=(3.5, 1.75) at dpi=300 for a two-panel side-by-side
 *          layout
 *   6. Save the final composite as sar_before_during.png
 *
 * Do NOT substitute simulated/synthetic data for this figure -- if the
 * exports are not available yet, leave the figure as a TODO in the draft
 * rather than fabricating backscatter values.
 */

// ── Study area: Sunamganj Haor, Bangladesh (same AOI as 01_sentinel1_flood_detection.js) ──
var HAOR = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00]);

// ── Date windows ──────────────────────────────────────────────────────────
// Pre-flood / dry-season reference: January 2017
var PRE_START = '2017-01-01';
var PRE_END   = '2017-01-31';
// Flood window: the 2017 Sunamganj pre-monsoon flash flood
var FLOOD_START = '2017-04-01';
var FLOOD_END   = '2017-04-20';

// ── Load Sentinel-1 GRD, VV, IW mode (same filters as 01_sentinel1_flood_detection.js) ──
function loadVV(start, end) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(HAOR)
    .filterDate(start, end)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .select(['VV']);
}

var pre_coll   = loadVV(PRE_START, PRE_END);
var flood_coll = loadVV(FLOOD_START, FLOOD_END);

print('Pre-flood (Jan 2017) S1 image count:', pre_coll.size());
print('Flood (Apr 2017) S1 image count:', flood_coll.size());

var s1_pre   = pre_coll.median();
var s1_flood = flood_coll.median();

// ── Speckle filtering: same Lee-filter approximation as 01_sentinel1_flood_detection.js ──
var s1_pre_smooth   = s1_pre.focal_mean({radius: 50, kernelType: 'circle', units: 'meters'});
var s1_flood_smooth = s1_flood.focal_mean({radius: 50, kernelType: 'circle', units: 'meters'});

// ── Visualize with a SHARED dB range for a comparable colorbar ─────────────
var visParams = {min: -25, max: 0, palette: ['000000', '2b6cb0', 'ffffff']};
Map.centerObject(HAOR, 11);
Map.addLayer(s1_pre_smooth, visParams, 'VV -- Pre-Flood (Jan 2017)');
Map.addLayer(s1_flood_smooth, visParams, 'VV -- Flood (Apr 2017)');

// ── Export BOTH composites as GeoTIFF (real dB values, no colormap baked in) ──
Export.image.toDrive({
  image: s1_pre_smooth,
  description: 'Haor_VV_preflood_Jan2017',
  folder: 'HaorFloodAlert_Fig3_4',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: s1_flood_smooth,
  description: 'Haor_VV_flood_Apr2017',
  folder: 'HaorFloodAlert_Fig3_4',
  region: HAOR,
  scale: 10,
  maxPixels: 1e9,
  fileFormat: 'GeoTIFF'
});

// ── Notes ───────────────────────────────────────────────────────────────────
// 1. If either image count above is 0, Sentinel-1 did not image this AOI in
//    that window -- widen PRE_START/PRE_END or FLOOD_START/FLOOD_END and
//    re-run before exporting.
// 2. This script exports the raw VV composites, in dB, not a flood mask --
//    use gee_scripts/01_sentinel1_flood_detection.js for the Otsu-threshold
//    flood-extent mask instead.
'''
    out_path = GEE_DIR / "export_fig_sar_real.js"
    out_path.write_text(content, encoding="utf-8")
    return {
        "status": "SCRIPT_GENERATED_NOT_RUN",
        "script_path": str(out_path.relative_to(ROOT)),
        "instructions": (
            "Paste gee_scripts/export_fig_sar_real.js into "
            "https://code.earthengine.google.com/, click Run, then run both "
            "Export.image.toDrive tasks from the Tasks tab. Download the two "
            "GeoTIFFs from Google Drive folder 'HaorFloodAlert_Fig3_4' and "
            "build the two-panel dB figure locally with a shared colorbar "
            "(vmin=-25, vmax=0 dB), sized for an 8.9cm/3.5in IEEE column "
            "width at 300+ DPI, saved as sar_before_during.png. No synthetic "
            "substitute was generated by this script."
        ),
    }


def main():
    t0 = time.time()
    log("=" * 70)
    log("HaorFloodAlert -- eval/run_paper_results.py -- FULL REGENERATION RUN")
    log("=" * 70)

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "config": {
            "rf_params": RF_PARAMS,
            "xgb_params_fixed": XGB_PARAMS,
            "xgb_scale_pos_weight": "computed per-fold from training-fold class balance (n_neg/n_pos)",
            "ensemble_weights": ENSEMBLE_WEIGHTS,
            "augmentation_factor": AUG_FACTOR,
            "augmentation_noise_sigma": NOISE_SIGMA,
            "augmentation_note": (
                "8x/0.04 chosen because these are the exact values in "
                "train_honest.py, the script confirmed (via matching "
                "hyperparameters and identical file mtimes) to have produced "
                "the currently-saved models/rf_model.pkl and "
                "models/xgb_model.pkl. train_augmented.py uses a different "
                "10x/0.05 combination on a different, superseded 40-row "
                "dataset and did not produce the deployed models."
            ),
            "decision_thresholds_reported": [THRESHOLD_PRIMARY, THRESHOLD_ALT],
            "feature_list_all_10": ALL_FEATS,
            "feature_list_source": "models/active_features.pkl (verified identical)",
        },
    }

    # ── Load data ────────────────────────────────────────────────────────
    log("Loading data/honest_training_data_v2.csv ...")
    df_full = pd.read_csv(DATA_DIR / "honest_training_data_v2.csv")
    df_full["date"] = pd.to_datetime(df_full["date"])
    df_real_sar = df_full[df_full["data_quality"] == "real_sar"].reset_index(drop=True)
    df_full_sorted = df_full.reset_index(drop=True)

    results["dataset"] = {
        "source_file": "data/honest_training_data_v2.csv",
        "n_total_events": int(len(df_full)),
        "n_real_sar_events": int(len(df_real_sar)),
        "n_proxy_pre_sentinel1_events": int((df_full["data_quality"] == "pre_sentinel1").sum()),
        "primary_set": "77-event real_sar subset (data_quality == 'real_sar')",
    }
    log(f"  Total events: {len(df_full)} | real_sar: {len(df_real_sar)} | "
        f"pre_sentinel1 (proxy): {(df_full['data_quality'] == 'pre_sentinel1').sum()}")

    # ── 3+4. Primary 77-event LOOCV, with and without augmentation ────────
    log("Running primary 77-event real-SAR LOOCV WITH augmentation "
        f"({AUG_FACTOR}x, sigma={NOISE_SIGMA}) ...")
    t = time.time()
    loocv_with_aug = loocv_ensemble(df_real_sar, ALL_FEATS, use_augmentation=True)
    log(f"  Done in {time.time()-t:.1f}s -- Acc@0.5="
        f"{loocv_with_aug['metrics_threshold_0.50']['accuracy']*100:.1f}% "
        f"AUC={loocv_with_aug['auc']*100:.1f}%")

    log("Running primary 77-event real-SAR LOOCV WITHOUT augmentation ...")
    t = time.time()
    loocv_no_aug = loocv_ensemble(df_real_sar, ALL_FEATS, use_augmentation=False)
    log(f"  Done in {time.time()-t:.1f}s -- Acc@0.5="
        f"{loocv_no_aug['metrics_threshold_0.50']['accuracy']*100:.1f}% "
        f"AUC={loocv_no_aug['auc']*100:.1f}%")

    # Save per-event predictions (primary, with-augmentation run) for the ROC curve
    pred_df = pd.DataFrame({
        "date": df_real_sar["date"].dt.strftime("%Y-%m-%d").values,
        "flood_label": loocv_with_aug["_trues"],
        "predicted_probability": loocv_with_aug["_probs"],
        "predicted_class_threshold_0.50": (loocv_with_aug["_probs"] >= THRESHOLD_PRIMARY).astype(int),
        "predicted_class_threshold_0.40": (loocv_with_aug["_probs"] >= THRESHOLD_ALT).astype(int),
    })
    pred_df.to_csv(RESULTS_DIR / "loocv_predictions.csv", index=False)
    log(f"  Wrote {len(pred_df)} per-event predictions -> results/loocv_predictions.csv")

    results["loocv_77_real_sar"] = {
        "with_augmentation": {k: v for k, v in loocv_with_aug.items() if not k.startswith("_")},
        "without_augmentation": {k: v for k, v in loocv_no_aug.items() if not k.startswith("_")},
    }

    # ── 5. 5-fold stratified CV on full 131 and on 77-event real-SAR ───────
    log("Running 5-fold stratified CV on full 131-event set ...")
    t = time.time()
    cv_full = five_fold_cv(df_full_sorted, ALL_FEATS)
    log(f"  Done in {time.time()-t:.1f}s -- mean acc "
        f"{cv_full['mean_accuracy']*100:.1f}% +/- {cv_full['std_accuracy']*100:.1f}%")

    log("Running 5-fold stratified CV on 77-event real-SAR set ...")
    t = time.time()
    cv_real_sar = five_fold_cv(df_real_sar, ALL_FEATS)
    log(f"  Done in {time.time()-t:.1f}s -- mean acc "
        f"{cv_real_sar['mean_accuracy']*100:.1f}% +/- {cv_real_sar['std_accuracy']*100:.1f}%")

    results["five_fold_cv"] = {
        "full_131_events": cv_full,
        "real_sar_77_events": cv_real_sar,
    }

    # ── 6. Ablation on the 131-event set ───────────────────────────────────
    log("Running ablation study on 131-event set (7 feature-arms, LOOCV each) ...")
    ablation_arms = {
        "all_10_features_baseline": ALL_FEATS,
        "no_sar_upstream_vv_kept": [f for f in ALL_FEATS if f not in SAR_FEATS],
        "no_rain_forecasts": [f for f in ALL_FEATS if f not in RAIN_FEATS],
        "no_soil_moisture": [f for f in ALL_FEATS if f != "soil_moisture"],
        "sar_only": SAR_FEATS,
        "weather_only": WEATHER_FEATS,
        "all_minus_upstream_vv_only": [f for f in ALL_FEATS if f != "upstream_vv"],
    }
    ablation_results = {}
    for i, (name, feats) in enumerate(ablation_arms.items()):
        t = time.time()
        log(f"  [{i+1}/{len(ablation_arms)}] {name} ({len(feats)} feats) ...")
        ablation_results[name] = loocv_ablation_arm(df_full_sorted, feats)
        log(f"      -> Acc={ablation_results[name]['accuracy']*100:.1f}% "
            f"AUC={ablation_results[name]['auc']*100:.1f}% "
            f"({time.time()-t:.1f}s)")

    results["ablation_131_events"] = {
        "note": ("'no_sar' removes VV/VH/vv_vh_ratio only -- upstream_vv is "
                 "explicitly KEPT in that arm. 'all_minus_upstream_vv_only' "
                 "is the new arm isolating just the upstream discharge proxy."),
        "arms": ablation_results,
    }

    # ── 7. Stratified holdout, 5 seeds, 60/40, on the 77-event real-SAR set ──
    log(f"Running stratified holdout ({len(HOLDOUT_SEEDS)} seeds, "
        f"{int((1-HOLDOUT_TEST_SIZE)*100)}/{int(HOLDOUT_TEST_SIZE*100)} split) "
        "on 77-event real-SAR set ...")
    t = time.time()
    holdout = stratified_holdout(df_real_sar, ALL_FEATS, HOLDOUT_SEEDS, HOLDOUT_TEST_SIZE)
    log(f"  Done in {time.time()-t:.1f}s -- mean acc "
        f"{holdout['mean_accuracy']*100:.1f}% +/- {holdout['std_accuracy']*100:.1f}%, "
        f"mean AUC {holdout['mean_auc']:.3f}")
    results["holdout_77_real_sar"] = {"dataset": "77-event real_sar subset", **holdout}

    # ── 8. Deconfounding ───────────────────────────────────────────────────
    log("Computing live deconfounding correlations on full 131-event set ...")
    deconf = compute_deconfounding(df_full_sorted)
    log(f"  raw temp vs label r={deconf['raw_temp_vs_label']['r']:.4f} | "
        f"raw temp vs month r={deconf['raw_temp_vs_month']['r']:.4f} | "
        f"temp_anomaly vs label r={deconf['temp_anomaly_vs_label']['r']:.4f}")
    results["deconfounding"] = deconf

    # ── 9. FFWC statistics ──────────────────────────────────────────────────
    log("Computing FFWC/SW269 gauge statistics ...")
    ffwc = compute_ffwc_stats()
    if ffwc["status"] == "OK":
        log(f"  n_readings={ffwc['n_readings_total_in_gauge_file']} "
            f"matched={ffwc['n_dates_matched_to_labeled_events']} "
            f"flood_mean={ffwc['flood_mean_water_level_m']:.3f}m "
            f"dry_mean={ffwc['dry_mean_water_level_m']:.3f}m "
            f"t={ffwc['t_test']['t_statistic']:.2f} r={ffwc['point_biserial_r']['r']:.3f}")
    else:
        log(f"  FFWC stats status: {ffwc['status']} -- {ffwc.get('message', '')}")
    results["ffwc_sw269_gauge"] = ffwc

    # ── 10. Baselines on 77-event real-SAR set ─────────────────────────────
    log("Running baselines on 77-event real-SAR set ...")
    t = time.time()
    b_majority = baseline_majority(df_real_sar, ALL_FEATS)
    log(f"  Majority-class: Acc={b_majority['accuracy']*100:.1f}%")
    b_rainfall = baseline_rainfall_rule(df_real_sar)
    log(f"  Rainfall-threshold rule: Acc={b_rainfall['accuracy']*100:.1f}%")
    b_logreg = baseline_logreg(df_real_sar, ALL_FEATS)
    log(f"  Logistic Regression (same LOOCV+augmentation protocol): "
        f"Acc={b_logreg['accuracy']*100:.1f}% AUC={b_logreg['auc']*100:.1f}% "
        f"({time.time()-t:.1f}s)")

    results["baselines_77_real_sar"] = {
        "majority_class": b_majority,
        "rainfall_threshold_rule": b_rainfall,
        "logistic_regression": b_logreg,
    }

    # ── 11. Real SAR figure export script ──────────────────────────────────
    log("Writing gee_scripts/export_fig_sar_real.js (to be run by hand in GEE) ...")
    sar_fig = write_sar_gee_script()
    log(f"  {sar_fig['status']}: {sar_fig['script_path']}")
    results["sar_figure_fig3_4"] = sar_fig

    total_s = time.time() - t0
    results["total_runtime_seconds"] = round(total_s, 1)
    log(f"Total runtime: {total_s/60:.1f} minutes")

    # ── Write JSON ──────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "paper_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    log(f"Wrote {json_path.relative_to(ROOT)}")

    # ── Write human-readable TXT ────────────────────────────────────────────
    write_txt_report(results)
    log(f"Wrote {(RESULTS_DIR / 'paper_results.txt').relative_to(ROOT)}")

    # ── Write PAPER_NUMBERS_SUMMARY.md ──────────────────────────────────────
    write_summary_md(results)
    log(f"Wrote {(RESULTS_DIR / 'PAPER_NUMBERS_SUMMARY.md').relative_to(ROOT)}")

    log("=" * 70)
    log("DONE. Every number above was computed live in this run -- nothing")
    log("was tuned to reproduce a previously reported figure.")
    log("=" * 70)


def write_txt_report(r):
    lines = []
    lines.append("HaorFloodAlert -- Paper Results (regenerated from scratch)")
    lines.append("=" * 65)
    lines.append(f"Generated at (UTC): {r['generated_at_utc']}")
    lines.append(f"Seed: {r['seed']}")
    lines.append("")
    lines.append("This file mirrors results/paper_results.json in human-readable")
    lines.append("form. See that file for full machine-readable detail.")
    lines.append("")
    lines.extend(LOG)
    lines.append("")
    lines.append("=" * 65)
    lines.append("Full JSON: results/paper_results.json")
    with open(RESULTS_DIR / "paper_results.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_summary_md(r):
    d = r["dataset"]
    la = r["loocv_77_real_sar"]["with_augmentation"]
    lna = r["loocv_77_real_sar"]["without_augmentation"]
    m50 = la["metrics_threshold_0.50"]
    m40 = la["metrics_threshold_0.40"]
    mna50 = lna["metrics_threshold_0.50"]
    cv_full = r["five_fold_cv"]["full_131_events"]
    cv_rs = r["five_fold_cv"]["real_sar_77_events"]
    holdout = r["holdout_77_real_sar"]
    deconf = r["deconfounding"]
    ffwc = r["ffwc_sw269_gauge"]
    base = r["baselines_77_real_sar"]
    ablation = r["ablation_131_events"]["arms"]

    md = []
    md.append("# Paper Numbers Summary")
    md.append("")
    md.append(f"Generated live by `eval/run_paper_results.py` at {r['generated_at_utc']}. "
               "Every value below is traceable to a key in `results/paper_results.json` "
               "(path shown in parentheses). Nothing here was tuned to match a previously "
               "reported number.")
    md.append("")
    md.append("| # | Number | Value | JSON path |")
    md.append("|---|---|---|---|")
    md.append(f"| 1 | Total events | {d['n_total_events']} | `dataset.n_total_events` |")
    md.append(f"| 2 | Real-SAR events (primary set) | {d['n_real_sar_events']} | `dataset.n_real_sar_events` |")
    md.append(f"| 3 | Proxy (pre-Sentinel-1) events | {d['n_proxy_pre_sentinel1_events']} | `dataset.n_proxy_pre_sentinel1_events` |")
    md.append(f"| 4 | LOOCV accuracy @0.50, with augmentation | {m50['accuracy']*100:.1f}% ({m50['n_correct']}/{m50['n_total']}) | `loocv_77_real_sar.with_augmentation.metrics_threshold_0.50` |")
    md.append(f"| 5 | LOOCV precision @0.50, with augmentation | {m50['precision']*100:.1f}% | same |")
    md.append(f"| 6 | LOOCV recall @0.50, with augmentation | {m50['recall']*100:.1f}% | same |")
    md.append(f"| 7 | LOOCV F1 @0.50, with augmentation | {m50['f1']*100:.1f}% | same |")
    md.append(f"| 8 | LOOCV specificity @0.50, with augmentation | {m50['specificity']*100:.1f}% | same |")
    md.append(f"| 9 | LOOCV AUC, with augmentation | {la['auc']:.3f} | `loocv_77_real_sar.with_augmentation.auc` |")
    cm = m50["confusion_matrix"]
    md.append(f"| 10 | LOOCV confusion matrix @0.50 | TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']} | same |")
    md.append(f"| 11 | LOOCV accuracy @0.40, with augmentation | {m40['accuracy']*100:.1f}% | `loocv_77_real_sar.with_augmentation.metrics_threshold_0.40` |")
    md.append(f"| 12 | LOOCV accuracy @0.50, WITHOUT augmentation | {mna50['accuracy']*100:.1f}% | `loocv_77_real_sar.without_augmentation.metrics_threshold_0.50` |")
    md.append(f"| 13 | LOOCV F1 @0.50, WITHOUT augmentation | {mna50['f1']*100:.1f}% | same |")
    md.append(f"| 14 | LOOCV AUC, WITHOUT augmentation | {lna['auc']:.3f} | `loocv_77_real_sar.without_augmentation.auc` |")
    md.append(f"| 15 | 5-fold CV, 131 events: mean accuracy | {cv_full['mean_accuracy']*100:.1f}% +/- {cv_full['std_accuracy']*100:.1f}% | `five_fold_cv.full_131_events` |")
    md.append(f"| 16 | 5-fold CV, 77 real-SAR events: mean accuracy | {cv_rs['mean_accuracy']*100:.1f}% +/- {cv_rs['std_accuracy']*100:.1f}% | `five_fold_cv.real_sar_77_events` |")
    md.append(f"| 17 | Holdout (5 seeds, 60/40): mean accuracy | {holdout['mean_accuracy']*100:.1f}% +/- {holdout['std_accuracy']*100:.1f}% | `holdout_77_real_sar` |")
    md.append(f"| 18 | Holdout (5 seeds, 60/40): mean AUC | {holdout['mean_auc']:.3f} +/- {holdout['std_auc']:.3f} | `holdout_77_real_sar` |")
    md.append(f"| 19 | Deconfounding: raw temp vs label r | {deconf['raw_temp_vs_label']['r']:.4f} | `deconfounding.raw_temp_vs_label` |")
    md.append(f"| 20 | Deconfounding: raw temp vs month r | {deconf['raw_temp_vs_month']['r']:.4f} | `deconfounding.raw_temp_vs_month` |")
    md.append(f"| 21 | Deconfounding: temp_anomaly vs label r | {deconf['temp_anomaly_vs_label']['r']:.4f} | `deconfounding.temp_anomaly_vs_label` |")

    for i, (name, arm) in enumerate(ablation.items()):
        md.append(f"| {22+i} | Ablation -- {name} ({arm['n_features']} feats) | Acc={arm['accuracy']*100:.1f}%, AUC={arm['auc']:.3f} | `ablation_131_events.arms.{name}` |")

    n = 22 + len(ablation)
    if ffwc["status"] == "OK":
        md.append(f"| {n} | FFWC: readings in gauge file | {ffwc['n_readings_total_in_gauge_file']} | `ffwc_sw269_gauge.n_readings_total_in_gauge_file` |")
        md.append(f"| {n+1} | FFWC: dates matched to labeled events | {ffwc['n_dates_matched_to_labeled_events']} | `ffwc_sw269_gauge.n_dates_matched_to_labeled_events` |")
        md.append(f"| {n+2} | FFWC: flood mean water level | {ffwc['flood_mean_water_level_m']:.3f} m (n={ffwc['flood_n']}) | `ffwc_sw269_gauge.flood_mean_water_level_m` |")
        md.append(f"| {n+3} | FFWC: dry mean water level | {ffwc['dry_mean_water_level_m']:.3f} m (n={ffwc['dry_n']}) | `ffwc_sw269_gauge.dry_mean_water_level_m` |")
        md.append(f"| {n+4} | FFWC: t-statistic | {ffwc['t_test']['t_statistic']:.2f} (p={ffwc['t_test']['p_value']:.2e}) | `ffwc_sw269_gauge.t_test` |")
        md.append(f"| {n+5} | FFWC: point-biserial r | {ffwc['point_biserial_r']['r']:.3f} | `ffwc_sw269_gauge.point_biserial_r` |")
        md.append(f"| {n+6} | FFWC: naive best threshold | {ffwc['naive_best_threshold_m']:.2f} m | `ffwc_sw269_gauge.naive_best_threshold_m` |")
        md.append(f"| {n+7} | FFWC: naive threshold accuracy (in-sample) | {ffwc['naive_threshold_accuracy']*100:.1f}% | `ffwc_sw269_gauge.naive_threshold_accuracy` |")
        n = n + 8
    else:
        md.append(f"| {n} | FFWC statistics | UNAVAILABLE -- {ffwc['status']}: {ffwc.get('message','')} | `ffwc_sw269_gauge` |")
        n += 1

    md.append(f"| {n} | Baseline: majority-class accuracy | {base['majority_class']['accuracy']*100:.1f}% | `baselines_77_real_sar.majority_class` |")
    md.append(f"| {n+1} | Baseline: rainfall-threshold-rule accuracy | {base['rainfall_threshold_rule']['accuracy']*100:.1f}% | `baselines_77_real_sar.rainfall_threshold_rule` |")
    md.append(f"| {n+2} | Baseline: logistic regression accuracy | {base['logistic_regression']['accuracy']*100:.1f}% (AUC={base['logistic_regression']['auc']:.3f}) | `baselines_77_real_sar.logistic_regression` |")
    n += 3
    md.append(f"| {n} | SAR Figure 3.4 (real Sentinel-1) | {r['sar_figure_fig3_4']['status']} -- see `{r['sar_figure_fig3_4']['script_path']}` | `sar_figure_fig3_4` |")

    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- Model config, augmentation factor/sigma, ensemble weights, and feature list "
               "are documented in `paper_results.json.config` and were chosen to match the "
               "**currently saved** models (`models/rf_model.pkl`, `models/xgb_model.pkl`, "
               "`models/active_features.pkl`), not any previously reported paper number.")
    md.append("- The ablation study runs on the full 131-event set, per instruction; the "
               "primary LOOCV/holdout numbers above run on the 77-event real-SAR subset.")
    md.append("- FFWC naive-threshold accuracy is an in-sample descriptive statistic over "
               "dates that could be matched between the gauge file and the labeled dataset "
               "(not a cross-validated claim) -- see the note field in the JSON.")
    md.append("- Figure 3.4 requires manually running `gee_scripts/export_fig_sar_real.js` "
               "in the GEE Code Editor; no synthetic substitute was generated.")

    with open(RESULTS_DIR / "PAPER_NUMBERS_SUMMARY.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


if __name__ == "__main__":
    main()
