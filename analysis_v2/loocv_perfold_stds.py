"""
analysis_v2/loocv_perfold_stds.py
===================================
Diagnostic: rerun the primary 77-event real-SAR LOOCV from
eval/run_paper_results.py with exactly ONE change -- the per-feature `stds`
used to scale augmentation noise is computed from each fold's 76 training
rows only, instead of once over all 77 rows before the fold loop the way
`loocv_ensemble` does it (eval/run_paper_results.py:182, outside the
`for tr_idx, val_idx in loo.split(X)` loop at line 187).

Everything else is reused via import, not reimplemented:
  ALL_FEATS, RF_PARAMS (via build_rf), XGB_PARAMS (via build_xgb),
  AUG_FACTOR, NOISE_SIGMA (both baked into augment()), ENSEMBLE_WEIGHTS,
  THRESHOLD_PRIMARY, SEED, compute_metrics, DATA_DIR, RESULTS_DIR.

For a traceable, apples-to-apples comparison, this script also calls the
UNMODIFIED `loocv_ensemble` directly (imported, not copied) in this same
process, and cross-checks its output against the already-published
results/paper_results.json (read-only) before reporting the diagnostic
delta -- so the "baseline" side of the comparison is not just typed in from
the task description, it is recomputed fresh and verified to match what is
on disk.

This script never calls eval.run_paper_results.main() and does not open any
file outside analysis_v2/ for writing. results/paper_results.json,
results/loocv_predictions.csv, and every other existing file are read-only
inputs here, never touched.

Run with:
  python analysis_v2/loocv_perfold_stds.py
"""
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

from eval.run_paper_results import (
    ALL_FEATS, DATA_DIR, ENSEMBLE_WEIGHTS, RESULTS_DIR, SEED,
    THRESHOLD_PRIMARY, build_rf, build_xgb, augment, compute_metrics,
    loocv_ensemble,
)

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def load_real_sar():
    """Identical source file and filter to eval.run_paper_results.main()
    (run_paper_results.py:640-644)."""
    df_full = pd.read_csv(DATA_DIR / "honest_training_data_v2.csv")
    df_full["date"] = pd.to_datetime(df_full["date"])
    return df_full[df_full["data_quality"] == "real_sar"].reset_index(drop=True)


def run_perfold_stds_loocv(df, feats):
    """Same protocol as eval.run_paper_results.loocv_ensemble, with the one
    change under test: `stds` is recomputed inside the fold loop from
    X[tr_idx] only, instead of once over the full X before the loop."""
    X = df[feats].values.astype(float)
    y = df["flood_label"].values.astype(int)
    rng = np.random.default_rng(SEED)

    loo = LeaveOneOut()
    probs, trues = [], []
    for tr_idx, val_idx in loo.split(X):
        # The one change vs loocv_ensemble: per-fold stds instead of global.
        stds = X[tr_idx].std(axis=0)
        stds = np.where(stds < 1e-6, 1e-6, stds)

        Xtr, ytr = augment(X[tr_idx], y[tr_idx], stds, rng)
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
    metrics = compute_metrics(trues, probs, THRESHOLD_PRIMARY)
    return trues, probs, metrics, auc


def summarize(metrics, auc):
    cm = metrics["confusion_matrix"]
    return {
        "accuracy": metrics["accuracy"], "precision": metrics["precision"],
        "recall": metrics["recall"], "f1": metrics["f1"],
        "specificity": metrics["specificity"], "auc": auc,
        "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
    }


if __name__ == "__main__":
    df = load_real_sar()

    # 1. Fresh, traceable baseline -- the UNMODIFIED function, called directly.
    baseline_res = loocv_ensemble(df, ALL_FEATS, use_augmentation=True, seed=SEED)
    baseline = summarize(baseline_res["metrics_threshold_0.50"], baseline_res["auc"])

    # 2. Cross-check against the already-published, on-disk results (read-only).
    published_path = RESULTS_DIR / "paper_results.json"
    with open(published_path, encoding="utf-8") as fh:
        published_json = json.load(fh)
    pub_m = published_json["loocv_77_real_sar"]["with_augmentation"]["metrics_threshold_0.50"]
    published = {
        "accuracy": pub_m["accuracy"], "precision": pub_m["precision"],
        "recall": pub_m["recall"], "f1": pub_m["f1"],
        "specificity": pub_m["specificity"],
        "auc": published_json["loocv_77_real_sar"]["with_augmentation"]["auc"],
        **pub_m["confusion_matrix"],
    }
    mismatches = [(k, baseline[k], published[k]) for k in published
                  if abs(baseline[k] - published[k]) > 1e-9]

    # 3. The diagnostic: per-fold stds.
    trues, probs, diag_metrics, diag_auc = run_perfold_stds_loocv(df, ALL_FEATS)
    diagnostic = summarize(diag_metrics, diag_auc)

    # 4. Write per-event predictions for the diagnostic run.
    pred_df = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d").values,
        "flood_label": trues,
        "predicted_probability": probs,
        "predicted_class_threshold_0.50": (probs >= THRESHOLD_PRIMARY).astype(int),
    })
    pred_df.to_csv(OUT_DIR / "loocv_perfold_stds.csv", index=False)

    # 5. Report.
    lines = []
    lines.append("=" * 72)
    lines.append("Cross-check: freshly-run loocv_ensemble() vs results/paper_results.json")
    if mismatches:
        lines.append("  MISMATCH found (fresh run does not reproduce the published file):")
        for k, v, pv in mismatches:
            lines.append(f"    {k}: fresh={v!r}  published_json={pv!r}")
    else:
        lines.append("  MATCH -- the unmodified loocv_ensemble(), called here directly, "
                      "reproduces results/paper_results.json exactly (deterministic, seed=42).")
    lines.append("=" * 72)
    lines.append("")

    def pct(x):
        return f"{x*100:.1f}%"

    lines.append(f"{'Metric':<14}{'Baseline (global stds)':>24}{'Diagnostic (per-fold stds)':>28}{'Delta (pp)':>13}")
    for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                      ("f1", "F1"), ("specificity", "Specificity")]:
        b, d = baseline[k], diagnostic[k]
        lines.append(f"{label:<14}{pct(b):>24}{pct(d):>28}{(d - b) * 100:>+12.1f}")
    lines.append(f"{'AUC':<14}{baseline['auc']:>24.3f}{diagnostic['auc']:>28.3f}{diagnostic['auc'] - baseline['auc']:>+13.3f}")
    lines.append("")
    lines.append("Confusion matrix:")
    lines.append(f"  Baseline (global stds)    : TN={baseline['tn']} FP={baseline['fp']} FN={baseline['fn']} TP={baseline['tp']}")
    lines.append(f"  Diagnostic (per-fold stds): TN={diagnostic['tn']} FP={diagnostic['fp']} FN={diagnostic['fn']} TP={diagnostic['tp']}")

    baseline_preds = (baseline_res["_probs"] >= THRESHOLD_PRIMARY).astype(int)
    diag_preds = (probs >= THRESHOLD_PRIMARY).astype(int)
    n_flipped = int((baseline_preds != diag_preds).sum())
    lines.append("")
    lines.append(f"Events whose predicted class (@0.50) flipped between the two runs: {n_flipped} / 77")

    report = "\n".join(lines)
    print(report)
    with open(OUT_DIR / "loocv_perfold_stds_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print(f"\nWrote {len(pred_df)} per-event predictions -> analysis_v2/results/loocv_perfold_stds.csv")
    print("Wrote summary -> analysis_v2/results/loocv_perfold_stds_summary.txt")
