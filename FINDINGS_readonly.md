# Read-Only Findings — HaorFloodAlert Methodology Audit

Scope: answer 5 methodology questions strictly from source code (no edits made
to this repository). Every claim below is cited to a file path and line
number that can be independently re-checked. Line numbers refer to the files
as they exist in the working tree at the time of this audit.

---

## Q1. Are the 54 pre-2014 proxy events included in each training fold of the primary 77-event LOOCV?

**Answer: No. Training in each fold is restricted to the remaining 76 real-SAR events. The 54 proxy events are structurally absent from this LOOCV — they are filtered out before the loop ever runs.**

Evidence, [eval/run_paper_results.py](eval/run_paper_results.py):

- Line 644: `df_real_sar = df_full[df_full["data_quality"] == "real_sar"].reset_index(drop=True)` — this builds a dataframe containing **only** the 77 rows where `data_quality == "real_sar"`. The 54 `pre_sentinel1` rows are excluded from `df_real_sar` entirely.
- Lines 661 and 668: `loocv_with_aug = loocv_ensemble(df_real_sar, ALL_FEATS, use_augmentation=True)` and `loocv_no_aug = loocv_ensemble(df_real_sar, ALL_FEATS, use_augmentation=False)` — the primary LOOCV function is called with `df_real_sar` (77 rows), not `df_full`/`df_full_sorted` (131 rows).
- Inside `def loocv_ensemble(df, feats, use_augmentation, seed=SEED):` (line 173): line 179 `X = df[feats].values.astype(float)` and line 180 `y = df["flood_label"].values.astype(int)` build `X`/`y` **only** from whatever `df` was passed in. Line 185 `loo = LeaveOneOut()` and line 187 `for tr_idx, val_idx in loo.split(X):` then split indices of *that same* `X`.

Because `X` (line 179) is constructed exclusively from the 77-row `df_real_sar`, `tr_idx` at line 187 can only ever index into those same 77 rows. With one row held out (`val_idx`), each training fold (`X[tr_idx]`) necessarily contains the other 76 real-SAR rows and nothing else — the 54 proxy rows were never loaded into `X` in the first place, so there is no code path by which they could enter a training fold of this LOOCV.

Dataset composition independently verified by reading `data/honest_training_data_v2.csv` (the file loaded at line 640, `pd.read_csv(DATA_DIR / "honest_training_data_v2.csv")`): 131 rows total, `data_quality` value counts = `real_sar: 77`, `pre_sentinel1: 54`.

---

## Q2. Is the monthly climatological mean (T_clim) computed once over the full dataset, or recomputed per fold?

**Answer: Neither. T_clim is not computed from the training dataset at all, in any scope. It is a hardcoded dictionary literal in `config.py`, applied identically and unconditionally to every event at feature-generation time, entirely outside of and prior to the LOOCV loop.**

Evidence, [config.py:76-89](config.py:76):
```
TEMP_CLIMATOLOGY = {
    1: 17.03, 2: 19.35, 3: 22.04, 4: 24.83, 5: 25.35, 6: 27.14,
    7: 27.26, 8: 27.48, 9: 28.26, 10: 26.95, 11: 25.40, 12: 20.09,
}
```
This is a plain Python dict literal — 12 fixed float constants, one per calendar month. It is not derived by any `.groupby()`, `.resample()`, `.mean()`, or similar pandas/numpy call on the training CSV anywhere in the repository (searched for `groupby.*month`, `resample`, `.mean()` across all `.py` files; the only `.mean()` hits are unrelated ERA5 `reduceRegion` spatial means in `collect_honest_data.py:111` and `collect_more_data.py:143`, computed per-event from Earth Engine, not from the label/feature CSV).

`TEMP_CLIMATOLOGY` is consumed identically in three places, all at per-event feature-construction time (i.e., when a row is built for the dataset), never inside `eval/run_paper_results.py`'s LOOCV loop itself:
- [utils/gee_features.py:441-442](utils/gee_features.py:441): `end_month = int(end[5:7]); temp_anomaly = round(temp - TEMP_CLIMATOLOGY.get(end_month, temp), 2)`
- [expand_training_data.py:156-157](expand_training_data.py:156): `month = int(date_str[5:7]); temp_anomaly = round(wx["temp"] - TEMP_CLIMATOLOGY.get(month, wx["temp"]), 2)`
- [dib_export/refetch_15_events.py:180-182](dib_export/refetch_15_events.py:180): `def temp_anomaly_for(date_str, temp): ... return round(temp - TEMP_CLIMATOLOGY.get(month, temp), 2)`

Since the same fixed constant is applied to every row regardless of which fold it later lands in, there is no "once over the full dataset" computation step to point to and no "recomputed per fold" step either — the dictionary is simply imported (e.g. `config.py` import in each of the three files above) and indexed by month. (Separately, per Q4 below, `temp_anomaly` itself never enters the primary LOOCV's feature matrix regardless.)

---

## Q3. Is the |r| > 0.3 feature-correlation screening computed on the full dataset or per fold?

**Answer: Cannot determine — no code implementing an |r| > 0.3 feature-correlation screening step was found anywhere in this repository.**

Searches performed (all read-only, all returned no matching implementation):
- `grep -rn "> 0\.3\|>0\.3\|< -0\.3\|abs(r\|abs(corr" --include=*.py .` — no correlation-screening threshold at 0.3 anywhere.
- `grep -rn "\.corr(\|corrcoef\|corr_matrix\|correlation_matrix\|feature_corr\|feature_select\|select_features\|CORR_THRESH" --include=*.py .` — no feature-selection-by-correlation routine anywhere.
- Checked `docs/PAPER_VERIFICATION.md` and `docs/conference_paper.md` for the same terms — no mention of a 0.3 correlation-screening threshold.

The only correlation-based feature exclusions that exist in the codebase use a **different** threshold and are unrelated to a systematic |r| > 0.3 screen:
- [add_barak_discharge.py:110](add_barak_discharge.py:110): `if abs(r_upvv) > 0.70:` — a one-off multicollinearity check between `barak_discharge_cumecs` and `upstream_vv`; threshold is 0.70, and `barak_discharge_cumecs` is not in `FEATURES`/`ALL_FEATS` regardless (dashboard-only, per `config.py:69`).
- [eval/run_paper_results.py:399-408](eval/run_paper_results.py:399), `def compute_deconfounding(df):` — computes three Pearson correlations (`temp` vs `flood_label`, `temp` vs `month`, `temp_anomaly` vs `flood_label`) and returns them for reporting only. Called once, at line 747: `deconf = compute_deconfounding(df_full_sorted)`, on the full 131-event set. Nothing is dropped or screened as a result of these numbers in this function — it produces report values, not a feature mask.

Per the instructions, since no such screening code exists to characterize as full-dataset-scope or per-fold-scope, this is reported as **cannot determine** rather than inferred from the paper text or comments.

---

## Q4. Is temp_anomaly dropped entirely before training, so no information derived from it reaches the model?

**Answer, for the primary 77-event LOOCV (`eval/run_paper_results.py`): Yes, unambiguously — `temp_anomaly` is never loaded into the feature matrix at all.**

Evidence:
- [eval/run_paper_results.py:112-116](eval/run_paper_results.py:112):
  ```
  ALL_FEATS = [
      "VV", "VH", "vv_vh_ratio",
      "rainfall", "soil_moisture", "wind",
      "forecast_rain_next_12h", "ndwi", "upstream_vv", "forecast_rain_72h",
  ]
  ```
  This 10-item list literal does not contain `"temp_anomaly"` (nor `"slope"` nor `"twi"`).
- Line 179: `X = df[feats].values.astype(float)` — inside `loocv_ensemble`, called with `feats=ALL_FEATS` (lines 661, 668). Since `ALL_FEATS` never contains `"temp_anomaly"`, the column is never selected into `X`, so it cannot reach `rf.fit(Xtr, ytr)` / `xgb.fit(Xtr, ytr)` (lines 200-201) in this script.
- None of the 10 features in `ALL_FEATS` are computed as a function of `temp_anomaly` elsewhere (`vv_vh_ratio` is a SAR ratio; the others are independent rainfall/wind/SAR/NDWI/discharge signals), so there is no indirect/derived leakage path into this feature set either.

**Important caveat found during verification — reported for completeness, since it bears directly on whether this holds for the currently-deployed model artifact, not just this one script:**

[train_honest.py](train_honest.py) — referenced elsewhere in the repo as the script that produced the currently-saved `models/rf_model.pkl` / `models/xgb_model.pkl` — uses a **different** feature-selection mechanism that does **not** explicitly exclude `temp_anomaly`:
- Line 17: `from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES` — imports `config.FEATURES`, which **does** include `"temp_anomaly"` ([config.py:91-116](config.py:91), `"temp_anomaly"` at line 102).
- Line 45: `avail = [f for f in FEATURES if f in df.columns]`.
- Lines 49-52 (the only feature-dropping logic in this script): `stds = X_full.std(axis=0); keep = stds > 1e-6; kept = [f for f, k in zip(avail, keep) if k]` — a **zero-variance filter only**.

Empirically re-running this exact filtering logic (read-only, no files written) against the two candidate CSVs present in `data/` shows `temp_anomaly` has substantial, non-zero variance in both:
- `data/honest_training_data.csv`: `temp_anomaly` std ≈ 1.007 (101 rows)
- `data/honest_training_data_v2.csv`: `temp_anomaly` std ≈ 1.038 (77 real_sar rows) / ≈1.084 (all 131 rows)

Reproducing `train_honest.py`'s exact zero-variance-filter logic on either CSV yields `kept` = 11 features, **including** `temp_anomaly` (only `slope` and `twi` are dropped, since both have std = 0.0 in this single-location dataset). This means: as currently written, if `train_honest.py` were re-run today on either CSV in `data/`, its own zero-variance filter would **not** remove `temp_anomaly`.

Yet the actual saved artifact `models/active_features.pkl` (loaded directly via `joblib.load`) contains exactly 10 features, matching `ALL_FEATS` and excluding `temp_anomaly`/`slope`/`twi`; and `rf_model.pkl`/`xgb_model.pkl` (also loaded directly) report `n_features_in_ == 10`. So **the currently-saved model artifact does not contain temp_anomaly**, empirically — but the mechanism by which `train_honest.py`'s current code would arrive at that same 10-feature result is not evident from the code as it stands (its zero-variance filter alone would keep 11). I cannot determine from the current repository state which script/run/version actually produced the saved 10-feature model; I can only confirm that (a) the saved model itself has no `temp_anomaly` input, and (b) the present-day `train_honest.py` source, run on the present-day CSVs, would not reproduce that exclusion through the mechanism it currently implements.

---

## Q5. Is there evidence of how sigma=0.04 and the 8x augmentation factor were chosen? Was a search run against the same 77 events?

**Answer: No. No search, sweep, or comparison of these values against the 77-event (or 131-event) dataset was found anywhere — in code, in any config file, in any notebook, or in git history.**

**Git history:** `git log --all --oneline` returns a single commit:
```
f796371 security: load alert credentials from .env instead of hardcoding
```
`git show --stat HEAD` confirms this commit touched only `.env.example`, `.gitignore`, `config.py`, and `requirements.txt` — it added `python-dotenv` support, nothing related to training/augmentation. `git status --short` shows every training script, `eval/`, `data/`, and `docs/` file as untracked (`??`). **There is no commit history for any training or evaluation script in this repository**, so there is no earlier-commit trail of alternate sigma/augmentation values to inspect.

**Notebooks:** `find . -iname "*.ipynb"` returns no results — there are no notebooks in this repository at all.

**Hardcoded values found, by script** (none accompanied by a comparison against alternatives):

| File:line | AUG | sigma/noise | Dataset used |
|---|---|---|---|
| [eval/run_paper_results.py:102-103](eval/run_paper_results.py:102) | 8 | 0.04 | `honest_training_data_v2.csv`, 77-event real_sar subset (line 644) |
| [train_honest.py:63](train_honest.py:63) | 8 | 0.04 | `--data` arg, default `honest_training_data.csv` (101 rows, line 19) |
| [paper_figures.py:545](paper_figures.py:545) | 8 | 0.04 | `DATA_CSV` = `honest_training_data.csv` (line 37), 101 rows |
| [train_with_shap.py:49](train_with_shap.py:49) | 8 | 0.04 | `honest_training_data.csv` (line 27), 101 rows |
| [train_augmented.py:19-20](train_augmented.py:19) | 10 | 0.05 (`NOISE_FRAC`) | `real_training_data_v2.csv` (line 18), 40 rows |
| [relabel_and_train.py:68-69](relabel_and_train.py:68) | 10 | 0.05 | `real_training_data_v2.csv` (line 21), 40 rows |
| [train_lstm_real.py:75](train_lstm_real.py:75) | 6 | 0.03 | `honest_training_data.csv` (line 23), 101 rows, LSTM model (not RF/XGB) |

No script anywhere loops over multiple values of an augmentation factor or noise sigma and compares outcomes — searched for `GridSearchCV`, `param_grid`, `sweep`, and `for <var> in [...]`-style loops over noise/sigma/augmentation parameters (`grep -rn "for sigma\|for noise\|for aug\|GridSearchCV\|param_grid" --include=*.py .`); the only "sweep" hit in the repository is an unrelated decision-threshold sweep for the dashboard UI (`streamlit_app/pages/5_Validation.py:808-858`), not an augmentation-parameter sweep, and not run against the 77-event set.

The 8/0.04 pair is identical across four scripts, but three of those four (`train_honest.py` default, `paper_figures.py`, `train_with_shap.py`) point at the older 101-row `honest_training_data.csv`, not the 131-row/77-real-SAR file — only `eval/run_paper_results.py` demonstrably runs this value against the 77-event real-SAR set (per Q1). The 10/0.05 and 6/0.03 pairs are used on entirely different, smaller datasets (40 rows) or a different model architecture (LSTM), not as alternatives tested against the same 77 events.

**Conclusion: a search over sigma/augmentation-factor values against the same 77 events was not run** — there is no code, config, notebook, or git history evidence of one. The value pair (8, 0.04) appears as a literal constant, matched across scripts, with no accompanying ablation.

*(Auxiliary, not relied upon as proof for the above — a pre-existing document already in this repository, [docs/PAPER_VERIFICATION.md:148](docs/PAPER_VERIFICATION.md:148), independently reaches the same conclusion from an earlier verification pass: "Augmentation factor/sigma differs by script... no before/after accuracy pair (84.4%→89.6%) was found anywhere.")*

---

## File locations requested

### Per-event LOOCV predictions (roc_y.npy / roc_p.npy)

**No file named `make_figures.py` exists in this repository, no `.npy` file of any kind exists in this repository, and no code anywhere calls `np.save`/`np.load`.** Verified by:
- `find . -iname "*make_figures*"` — no results.
- `find . -iname "*.npy"` — no results.
- `grep -rn "np\.save\|\.npy\|np\.load" --include=*.py .` — no results.
- `grep -rln "roc_y\|roc_p" .` (all text file types) — no results.

The closest existing artifact is **[results/loocv_predictions.csv](results/loocv_predictions.csv)**, written by `eval/run_paper_results.py`:
- Line 673 (comment): `# Save per-event predictions (primary, with-augmentation run) for the ROC curve`
- Lines 674-680: builds `pred_df` with columns `date`, `flood_label` (the true labels — functional analogue of `roc_y`), `predicted_probability` (the functional analogue of `roc_p`), and predicted classes at both thresholds, from `df_real_sar["date"]` and `loocv_with_aug["_trues"]`/`loocv_with_aug["_probs"]`.
- Line 681: `pred_df.to_csv(RESULTS_DIR / "loocv_predictions.csv", index=False)`.

This is a single 77-row CSV (verified: 77 rows, columns `date, flood_label, predicted_probability, predicted_class_threshold_0.50, predicted_class_threshold_0.40`), not two separate `.npy` arrays. `grep -rln "loocv_predictions" --include=*.py .` shows this file is written only by `eval/run_paper_results.py` and is never read back by any script in the repository — it is a terminal output artifact, not an input to a figure generator. The actual ROC-curve figure generator, [paper_figures.py:382-449](paper_figures.py:382) `def fig6_roc_curve():`, independently recomputes its own out-of-fold probabilities in memory (`oof_probs`, built across lines 421-443, fed to `roc_curve(y, oof_probs, pos_label=1)` at line 449) rather than reading `results/loocv_predictions.csv` or any `.npy` file.

### 131-event listing (dates + labels)

**[data/honest_training_data_v2.csv](data/honest_training_data_v2.csv)** — confirmed by direct read: 131 rows, columns `date, flood_label, source, VV, VH, vv_vh_ratio, rainfall, soil_moisture, temp, wind, slope, twi, forecast_rain_next_12h, ndwi, upstream_vv, forecast_rain_72h, data_quality, temp_anomaly`; `data_quality` value counts `real_sar: 77`, `pre_sentinel1: 54`. This is the file loaded at `eval/run_paper_results.py:640`. Cross-confirmed by a pre-existing repository document, `docs/PAPER_VERIFICATION.md:46`: "`data/honest_training_data_v2.csv` = 131 rows (132 lines incl. header), 18 columns, with `data_quality` = `real_sar` (77 rows) / `pre_sentinel1` (54 rows)."

No other file in the repository lists 131 events: the `dib_export/*.csv` exports and `results/loocv_predictions.csv` all contain 77 rows (real-SAR subset only); `data/honest_training_data.csv` and `data/real_training_data_v3.csv` contain 101 rows; `data/real_training_data_v2.csv` contains 40 rows.
