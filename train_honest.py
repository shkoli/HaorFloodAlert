"""
train_honest.py  —  HaorFloodAlert primary model trainer.

Trains RF + XGBoost on honest_training_data_v2.csv using Leave-One-Out
Cross-Validation (LOOCV). Labels are sourced from FFWC Annual Reports and
peer-reviewed literature, fully independent of the GEE satellite features.

Usage
-----
    python train_honest.py
    python train_honest.py --real-only          # use only real Sentinel-1 events
    python train_honest.py --data custom.csv    # use a different CSV

Validated performance (LOOCV, n=77 real-SAR events):
    Accuracy 89.6%  |  Recall 87.5%  |  AUC-ROC 93.4%  |  F1 85.3%
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train HaorFloodAlert RF+XGB ensemble")
    parser.add_argument(
        "--data", default="honest_training_data_v2.csv",
        help="CSV filename inside the data/ directory",
    )
    parser.add_argument(
        "--real-only", action="store_true",
        help="Exclude pre-Sentinel-1 proxy rows (use only real SAR events)",
    )
    return parser.parse_known_args()[0]


def load_dataset(csv_path, real_only=False):
    """
    Load training CSV and optionally filter to real Sentinel-1 events.

    Parameters
    ----------
    csv_path : Path
    real_only : bool
        If True, drops rows flagged as pre-Sentinel-1 proxy data.

    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    if real_only:
        before = len(df)
        if "data_quality" in df.columns:
            df = df[df["data_quality"] != "pre_sentinel1"].reset_index(drop=True)
        else:
            # Sentinel-1 launched April 2014; rows with VV == -15 before 2014
            # are default-filled (no real SAR available).
            mask = (df["VV"] == -15.0) & (df["date"].str[:4].astype(int) < 2014)
            df = df[~mask].reset_index(drop=True)
        print(f"  --real-only: dropped {before - len(df)} proxy rows, {len(df)} remain")
    return df


def augment(X_train, y_train, feature_stds, n_copies=8, noise_scale=0.04):
    """
    Apply Gaussian noise augmentation inside a LOOCV fold.

    Adds n_copies - 1 perturbed copies of the training set to increase
    effective sample size without leaking validation data.

    Parameters
    ----------
    X_train, y_train : np.ndarray
    feature_stds : np.ndarray  — per-feature standard deviations
    n_copies : int             — total copies including original
    noise_scale : float        — noise as fraction of feature std

    Returns
    -------
    X_aug, y_aug : np.ndarray
    """
    rows, labels = [X_train], [y_train]
    for _ in range(n_copies - 1):
        noise = np.random.randn(*X_train.shape) * feature_stds * noise_scale
        rows.append(X_train + noise)
        labels.append(y_train)
    return np.vstack(rows), np.concatenate(labels)


def run_loocv(X, y, feature_stds, n_aug=8):
    """
    Run Leave-One-Out Cross-Validation with per-fold Gaussian augmentation.

    Parameters
    ----------
    X, y : np.ndarray
    feature_stds : np.ndarray
    n_aug : int — augmentation copies per fold

    Returns
    -------
    trues, preds, probs : np.ndarray
    """
    loo = LeaveOneOut()
    preds, probs, trues = [], [], []

    for i, (tr_idx, val_idx) in enumerate(loo.split(X)):
        X_tr, y_tr = augment(X[tr_idx], y[tr_idx], feature_stds, n_aug)
        X_val, y_val = X[val_idx], y[val_idx]

        class_ratio = float((y_tr == 0).sum()) / max(float(y_tr.sum()), 1)

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=5,
            class_weight="balanced", random_state=42,
        )
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=class_ratio, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        prob = (
            0.5 * rf.predict_proba(X_val)[:, 1]
            + 0.5 * xgb.predict_proba(X_val)[:, 1]
        )
        probs.append(float(prob[0]))
        preds.append(int(prob[0] >= 0.5))
        trues.append(int(y_val[0]))

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(y)}] folds complete", flush=True)

    return np.array(trues), np.array(preds), np.array(probs)


def run_cv5(X, y, feature_stds, n_aug=8):
    """
    Run 5-fold stratified cross-validation for a quick stability check.

    Returns
    -------
    cv_accuracies : list[float]
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = augment(X[tr_idx], y[tr_idx], feature_stds, n_aug)
        class_ratio = float((y_tr == 0).sum()) / max(float(y_tr.sum()), 1)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=5,
            class_weight="balanced", random_state=42,
        )
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4,
            scale_pos_weight=class_ratio,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        prob = (
            0.5 * rf.predict_proba(X[val_idx])[:, 1]
            + 0.5 * xgb.predict_proba(X[val_idx])[:, 1]
        )
        acc = accuracy_score(y[val_idx], (prob >= 0.5).astype(int)) * 100
        cv_acc.append(acc)
        print(f"  Fold {fold + 1}: {acc:.1f}%")
    return cv_acc


def train_final_models(X, y, feature_stds, n_aug=8):
    """
    Train final RF and XGBoost models on the full dataset.

    Parameters
    ----------
    X, y : np.ndarray — full dataset
    feature_stds : np.ndarray

    Returns
    -------
    rf, xgb : fitted models
    """
    X_full, y_full = augment(X, y, feature_stds, n_aug)
    class_ratio = float((y_full == 0).sum()) / max(float(y_full.sum()), 1)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        class_weight="balanced", random_state=42,
    )
    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        scale_pos_weight=class_ratio, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    rf.fit(X_full, y_full)
    xgb.fit(X_full, y_full)
    return rf, xgb


def main():
    """Main training pipeline."""
    args = parse_args()
    np.random.seed(42)

    csv_path = DATA_DIR / args.data
    print("=" * 65)
    print("  HaorFloodAlert — Honest Training (FFWC/BWDB Labels)")
    print("  Labels independent of features — zero circularity")
    print("=" * 65)

    df = load_dataset(csv_path, real_only=args.real_only)

    label_col = "flood_label"
    avail = [f for f in FEATURES if f in df.columns]
    X_full = df[avail].values.astype(float)
    y = df[label_col].values.astype(int)

    # Drop zero-variance features
    stds = X_full.std(axis=0)
    keep_mask = stds > 1e-6
    active = [f for f, k in zip(avail, keep_mask) if k]
    dropped = [f for f, k in zip(avail, keep_mask) if not k]
    X = X_full[:, keep_mask]
    feature_stds = stds[keep_mask]

    print(f"\nDataset : {len(df)} rows | Flood={int(y.sum())} | Dry={int((y == 0).sum())}")
    print(f"Label source: FFWC Annual Reports + peer-reviewed papers")
    if dropped:
        print(f"Dropped zero-variance features: {dropped}")
    print(f"Active features ({len(active)}): {active}")

    # Save active feature list
    joblib.dump(active, MODELS_DIR / "active_features.pkl")

    # LOOCV
    print(f"\nRunning LOOCV ({len(df)} folds, 8x augmentation) ...")
    trues, preds, probs = run_loocv(X, y, feature_stds)

    acc  = accuracy_score(trues, preds) * 100
    prec = precision_score(trues, preds, zero_division=0) * 100
    rec  = recall_score(trues, preds, zero_division=0) * 100
    f1   = f1_score(trues, preds, zero_division=0) * 100
    auc  = roc_auc_score(trues, probs) * 100
    cm   = confusion_matrix(trues, preds)
    base = int(y.sum()) / len(y) * 100

    print(f"\n  LOOCV Results (n={len(df)})")
    print(f"  Accuracy  : {acc:.1f}%  (baseline {base:.0f}%, +{acc - base:.1f}%)")
    print(f"  Precision : {prec:.1f}%")
    print(f"  Recall    : {rec:.1f}%")
    print(f"  F1 Score  : {f1:.1f}%")
    print(f"  AUC-ROC   : {auc:.1f}%")
    print(f"  Confusion  : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # 5-fold CV
    print("\n  5-Fold Stratified CV")
    cv_acc = run_cv5(X, y, feature_stds)
    print(f"  Mean: {np.mean(cv_acc):.1f}%  Std: {np.std(cv_acc):.1f}%")

    # Train and save final models
    print("\nTraining final models on full dataset ...")
    rf_final, xgb_final = train_final_models(X, y, feature_stds)
    joblib.dump(rf_final,  MODELS_DIR / "rf_model.pkl")
    joblib.dump(xgb_final, MODELS_DIR / "xgb_model.pkl")
    print(f"Models saved to {MODELS_DIR}")

    # Save LOOCV report
    report_path = RESULTS_DIR / "loocv_report.txt"
    with open(report_path, "w") as fh:
        fh.write(f"LOOCV n={len(df)}\n")
        fh.write(f"Accuracy  : {acc:.1f}%\n")
        fh.write(f"Precision : {prec:.1f}%\n")
        fh.write(f"Recall    : {rec:.1f}%\n")
        fh.write(f"F1        : {f1:.1f}%\n")
        fh.write(f"AUC-ROC   : {auc:.1f}%\n")
        fh.write(f"CM        : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}\n")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
