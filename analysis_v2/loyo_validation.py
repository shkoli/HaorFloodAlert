"""
analysis_v2/loyo_validation.py
===============================
Leave-one-year-out (LOYO) validation on the 77 real-SAR events, as a
complement to the primary LOOCV in eval/run_paper_results.py.

This module does not copy any logic from eval/run_paper_results.py -- it
imports the shared pieces directly (constants + model/augmentation/metric
functions) and only adds the year-based fold splitter, since
`loocv_ensemble` there is hardwired to sklearn's LeaveOneOut and has no hook
for a custom fold definition.

Imported, not reimplemented, from eval.run_paper_results:
  DATA_DIR, ALL_FEATS, SEED, AUG_FACTOR, NOISE_SIGMA, ENSEMBLE_WEIGHTS,
  THRESHOLD_PRIMARY, RF_PARAMS, XGB_PARAMS, build_rf, build_xgb, augment,
  compute_metrics

Fold definition: a held-out fold is every real-SAR event dated in one
calendar year; the model trains on all other real-SAR events. Folds are run
for 2017-2024 only. 2014 (1 event), 2015 (4 events) and 2016 (2 events)
contain zero flood events each (verified against
data/honest_training_data_v2.csv), so a fold holding one of them out would
have a single-class test set -- they are skipped as held-out folds per
instruction, but because the train mask for every evaluated year is simply
"not this year", events from 2014/2015/2016 are automatically included in
every one of the 8 training folds without any special-cased inclusion logic.

Deliberate deviation from the primary LOOCV -- flagged, not silently applied:
  eval.run_paper_results.loocv_ensemble computes the per-feature std used to
  scale augmentation noise ONCE, over the full 77-event set, before its fold
  loop (run_paper_results.py:182, outside the `for tr_idx, val_idx in
  loo.split(X)` loop at line 187) -- it is not refit per fold there, and nowhere
  in that function is there a StandardScaler applied to the model inputs
  (RF/XGB use the raw feature values; only the unrelated `baseline_logreg`
  function elsewhere in that file uses StandardScaler, per fold). The
  protocol for this LOYO analysis explicitly requires refitting the
  standardization inside each fold, so that is what this script does for the
  augmentation noise scale -- computed fresh from each fold's own training
  rows. This is intentionally NOT what the primary LOOCV does; see the
  chat report for this discrepancy.

Nothing here tunes anything against these results. Run with:
  python analysis_v2/loyo_validation.py
"""
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval.run_paper_results import (
    ALL_FEATS, AUG_FACTOR, DATA_DIR, ENSEMBLE_WEIGHTS, NOISE_SIGMA, SEED,
    THRESHOLD_PRIMARY, build_rf, build_xgb, augment, compute_metrics,
)

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

LOYO_YEARS = list(range(2017, 2025))  # 2017-2024 inclusive
SKIPPED_YEARS_NOTE = {2014: 1, 2015: 4, 2016: 2}  # year -> n events, all dry (0 floods)


def load_real_sar():
    """Same source file and same data_quality=='real_sar' filter as
    eval.run_paper_results.main() (run_paper_results.py:640-644)."""
    df_full = pd.read_csv(DATA_DIR / "honest_training_data_v2.csv")
    df_full["date"] = pd.to_datetime(df_full["date"])
    df_real_sar = df_full[df_full["data_quality"] == "real_sar"].reset_index(drop=True)
    df_real_sar["year"] = df_real_sar["date"].dt.year
    return df_real_sar


def run_loyo():
    df = load_real_sar()
    X_all = df[ALL_FEATS].values.astype(float)
    y_all = df["flood_label"].values.astype(int)
    years = df["year"].values
    dates = df["date"]

    # One generator created once and advanced across folds -- mirrors how
    # loocv_ensemble creates its rng once before the fold loop, rather than
    # re-seeding identically on every fold.
    rng = np.random.default_rng(SEED)

    per_event_rows = []
    per_year_stats = {}

    for year in LOYO_YEARS:
        val_mask = years == year
        train_mask = ~val_mask

        Xtr_raw, ytr_raw = X_all[train_mask], y_all[train_mask]
        Xval, yval = X_all[val_mask], y_all[val_mask]

        # Per-protocol: standardization (the augmentation noise scale) is
        # refit here from this fold's training rows only -- see module
        # docstring for how this differs from the primary LOOCV.
        stds = Xtr_raw.std(axis=0)
        stds = np.where(stds < 1e-6, 1e-6, stds)

        Xtr, ytr = augment(Xtr_raw, ytr_raw, stds, rng)

        n_pos = float((ytr == 1).sum())
        n_neg = float((ytr == 0).sum())
        spw = n_neg / max(n_pos, 1.0)

        rf = build_rf()
        xgb = build_xgb(spw)
        rf.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)

        p = (ENSEMBLE_WEIGHTS["rf"] * rf.predict_proba(Xval)[:, 1]
             + ENSEMBLE_WEIGHTS["xgb"] * xgb.predict_proba(Xval)[:, 1])

        val_dates = dates[val_mask].dt.strftime("%Y-%m-%d").values
        for d, yr_val, lbl, prob in zip(val_dates, years[val_mask], yval, p):
            per_event_rows.append({
                "date": d,
                "year": int(yr_val),
                "flood_label": int(lbl),
                "predicted_probability": float(prob),
            })

        m = compute_metrics(yval, p, THRESHOLD_PRIMARY)
        auc = float(roc_auc_score(yval, p)) if len(set(yval)) > 1 else float("nan")
        per_year_stats[year] = {
            "n": int(val_mask.sum()),
            "n_flood": int(yval.sum()),
            "n_dry": int((yval == 0).sum()),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "specificity": m["specificity"],
            "auc": auc,
            "confusion_matrix": m["confusion_matrix"],
        }

    pred_df = pd.DataFrame(per_event_rows).sort_values("date").reset_index(drop=True)
    pred_df.to_csv(OUT_DIR / "loyo_predictions.csv", index=False)

    all_trues = pred_df["flood_label"].values
    all_probs = pred_df["predicted_probability"].values
    pooled_metrics = compute_metrics(all_trues, all_probs, THRESHOLD_PRIMARY)
    pooled_auc = float(roc_auc_score(all_trues, all_probs))

    return {
        "per_year": per_year_stats,
        "pooled": pooled_metrics,
        "pooled_auc": pooled_auc,
        "n_folds": len(LOYO_YEARS),
        "n_events_evaluated": len(pred_df),
    }, pred_df


def format_report(summary):
    lines = []
    lines.append("HaorFloodAlert -- Leave-One-Year-Out (LOYO) validation")
    lines.append("=" * 60)
    lines.append(f"Held-out folds: {summary['n_folds']} (years {LOYO_YEARS[0]}-{LOYO_YEARS[-1]})")
    lines.append(f"Events evaluated (pooled across folds): {summary['n_events_evaluated']}")
    lines.append("Years skipped as held-out folds (0 flood events each, kept in every "
                  "training fold): " +
                  ", ".join(f"{y} (n={n})" for y, n in SKIPPED_YEARS_NOTE.items()))
    lines.append("")
    lines.append("-- Pooled metrics (threshold 0.50) --")
    p = summary["pooled"]
    lines.append(f"  Accuracy    : {p['accuracy']*100:.1f}%")
    lines.append(f"  Precision   : {p['precision']*100:.1f}%")
    lines.append(f"  Recall      : {p['recall']*100:.1f}%")
    lines.append(f"  F1          : {p['f1']*100:.1f}%")
    lines.append(f"  Specificity : {p['specificity']*100:.1f}%")
    lines.append(f"  AUC         : {summary['pooled_auc']:.3f}")
    cm = p["confusion_matrix"]
    lines.append(f"  Confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    lines.append("")
    lines.append("-- Per-year metrics --")
    header = (f"  {'Year':<6}{'n':>4}{'flood':>7}{'dry':>5}"
              f"{'Acc%':>8}{'Prec%':>8}{'Rec%':>8}{'F1%':>8}{'Spec%':>8}{'AUC':>7}"
              f"   TN FP FN TP")
    lines.append(header)
    for year in LOYO_YEARS:
        s = summary["per_year"][year]
        cm = s["confusion_matrix"]
        auc_str = f"{s['auc']:.3f}" if not np.isnan(s["auc"]) else "  n/a"
        lines.append(
            f"  {year:<6}{s['n']:>4}{s['n_flood']:>7}{s['n_dry']:>5}"
            f"{s['accuracy']*100:>8.1f}{s['precision']*100:>8.1f}"
            f"{s['recall']*100:>8.1f}{s['f1']*100:>8.1f}{s['specificity']*100:>8.1f}"
            f"{auc_str:>7}   {cm['tn']:>2} {cm['fp']:>2} {cm['fn']:>2} {cm['tp']:>2}"
        )
    lines.append("")
    lines.append("Note: 2017's held-out fold has only 1 dry event (6 flood, 1 dry); its "
                  "specificity is therefore either 0% or 100% by construction and should "
                  "not be read as a stable estimate.")
    return "\n".join(lines)


if __name__ == "__main__":
    summary, pred_df = run_loyo()
    report = format_report(summary)
    print(report)
    with open(OUT_DIR / "loyo_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nWrote {len(pred_df)} per-event predictions -> analysis_v2/results/loyo_predictions.csv")
    print("Wrote summary -> analysis_v2/results/loyo_summary.txt")
