# Paper Verification Report — HaorFloodAlert

Verified against actual code, saved models, and output files in this repository.
Method: direct file reads, grep, and live execution of pickled/joblib models (read-only).
Where a script produces a number only as a hardcoded literal (used to draw a figure) rather
than as a live, reproducible computation, this is called out explicitly — a hardcoded literal
is not the same as a verified measurement, even when it numerically matches the claim.

---

## Task 1 — Core metrics verification table

| Claim | Value found in code | File/script | Verdict |
|---|---|---|---|
| 77-SAR LOOCV: acc 89.6%, recall 87.5%, AUC 0.936, F1 87.5% | `full_vals=[0.896,0.875,0.941,0.875]` (n=101), `clean_vals=[0.896,0.875,0.936,0.875]` (n=77) | [gen_thesis_figures.py:474-475](gen_thesis_figures.py:474) | **MATCH** (as a hardcoded plotting literal — see note below) |
| Precision 87.5%, TN=41 FP=4 FN=4 TP=28 | Confusion matrix implies precision = recall = 28/32 = 87.5% (TP+FN=32, TP+FP=32) | Arithmetic from the same TN/FP/FN/TP; not itself found as a variable in code | **PARTIAL** — self-consistent with the matrix, but the matrix itself is not found as a computed array anywhere |
| specificity 91.1% (41/45) | 41/45 = 91.1%, consistent with TN=41 total-negatives=45 | derived, not found as a literal | **PARTIAL** — arithmetically consistent, not independently located |
| classification threshold 0.40 | All located scripts hardcode **0.5**, not 0.40 | [train_augmented.py:78](train_augmented.py:78) `int(p[0] >= 0.5)`; same in `train_honest.py` | **MISMATCH** |
| `results/validation_report_v2.txt` corroborates these | File is **0 bytes** | `results/validation_report_v2.txt` (confirmed via `wc -l` = 0) | **MISMATCH** — no report backs the numbers |
| Paper's own body text (for comparison) | Accuracy 89.6% (69/77), Recall 87.5%, **Precision 84.8%**, F1 86.2%, AUC **0.943**, Specificity 91.1% | `conference_paper.md:203-218` | Paper text itself is **internally inconsistent**: precision/F1/AUC in the prose don't match either the confusion matrix (which implies 87.5%/87.5%) or the figure literal (AUC 0.936, not 0.943) |
| RF: 500 trees, max_depth 12, min_samples_split 5 | Actual saved model: `n_estimators=500, max_depth=5, min_samples_split=2, class_weight='balanced'` | live `joblib.load('models/rf_model.pkl').get_params()`, mtime 2026-05-15 23:48 | **MISMATCH** |
| XGB: 500 estimators, max_depth 8, lr 0.05, subsample 0.8, colsample 0.8 | Actual saved model: `n_estimators=500, max_depth=4, learning_rate=0.05, subsample=None, colsample_bytree=None` | live `joblib.load('models/xgb_model.pkl').get_params()` | **MISMATCH** (only n_estimators=500 and lr=0.05 match) |
| Matching training script | RF/XGB params above exactly match [train_augmented.py:109-113](train_augmented.py:109) | `train_augmented.py`, trained on `data/real_training_data_v2.csv` (40 rows) | Confirms which script actually produced the saved models |
| Feature importances (rainfall 0.2476, soil_moisture 0.1947, forecast_rain_72h 0.1435, forecast_rain_12h 0.1142, VV 0.0817, vv_vh_ratio 0.0775, wind 0.0451, VH 0.0339, upstream_vv 0.0334, ndwi 0.0283) | Confirmed via live `rf.feature_importances_` on the currently-saved model, paired with `active_features.pkl` order: VV=0.0817, VH=0.0339, vv_vh_ratio=0.0775, rainfall=0.2476, soil_moisture=0.1947, wind=0.0451, forecast_rain_next_12h=0.1142, ndwi=0.0283, upstream_vv=0.0334, forecast_rain_72h=0.1435 — **all 10 values match to 4 decimal places** | live `joblib.load('models/rf_model.pkl').feature_importances_` + `models/active_features.pkl` | **MATCH — exact, confirmed directly against the live pickled model** |
| Figure 4 body-text importances shown in `conference_paper.md`/`paper_figures.py` (forecast_rain_72h 0.188, soil_moisture 0.135, etc.) | These are a *separate*, hardcoded set of illustrative literals for a different plot (an "RF 45%+XGB 35%+LSTM 20%"-weighted comparison), not the raw RF importances above, and disagree with each other between the two listings (VV 0.091 vs 0.088; ndwi 0.077 vs 0.100) | `conference_paper.md:246-259` vs [paper_figures.py:144-156, 698-709](paper_figures.py:144) | **MISMATCH (internally inconsistent, and distinct from the correctly-matching raw RF importances above)** |
| slope/TWI/temp_anomaly zero-variance, excluded | Confirmed for the deployed model — `active_features.pkl` (10 features) contains neither `slope`, `twi`, nor `temp_anomaly`. **Inconsistency found:** `paper_figures.py`'s ablation `ALL_FEATS` list ([line 512-516](paper_figures.py:512)) treats `temp_anomaly` as one of "11 non-zero-variance features" and includes it — contradicting the deployed 10-feature model, which excludes it | `models/active_features.pkl`; [train_honest.py:50-54](train_honest.py:50) variance filter (`stds > 1e-6`); [paper_figures.py:512-516](paper_figures.py:512) | **PARTIAL** — exclusion confirmed for the live model, but the ablation script's feature list disagrees with it |
| Deconfounding: raw temp r=0.5704 | `r_values = [0.570, ...]` | [paper_figures.py:336-337](paper_figures.py:336) | **MATCH** (rounds to 0.570; label says "r=0.570") |
| anomaly vs label r=-0.0305 | `r_values=[..., -0.031]` | [paper_figures.py:337](paper_figures.py:337) | **MATCH** (rounds to -0.031) |
| temp vs month r=0.400 | Paper body text says **r=0.510**, not 0.400 | `conference_paper.md:132` | **MISMATCH** |
| ensemble 88.3% -> 89.6% | Matches | `conference_paper.md:148,235` | **MATCH** |
| Augmentation 8x, sigma=0.05, without 84.4%/81.2%/0.919 -> with 89.6%/87.5%/0.936 | Actual saved-model script uses **`AUG_FACTOR=10`, `NOISE_FRAC=0.05`** ([train_augmented.py:19-20](train_augmented.py:19)); a *different* script (`train_honest.py`, `paper_figures.py:545`) uses **8x / 0.04**. No before/after pair matching 84.4%→89.6% found anywhere; `results/training_report.txt` shows unrelated numbers (87.5/82.9/90.6/86.6/89.4) | `train_augmented.py:19-20`; `paper_figures.py:545` | **MISMATCH** on factor/sigma and on the specific before/after numbers |
| Augmentation applied to train folds only (no leakage) | Confirmed — test fold uses raw unaugmented rows | [train_augmented.py:49-56, 63-65](train_augmented.py:49) + explicit docstring | **MATCH** (methodology sound, numbers not) |
| Ablation (101-event LOOCV): baseline 87.8%/0.941; no-SAR 80.2%; no-forecasts 85.5%; no-soil 89.3%; SAR-only 63.4%; weather-only 82.4% | Ablation function exists and computes these *live* (not hardcoded) via `loocv_subset()`; not executed during this verification (≈6×101 LOOCV folds, several minutes) so the specific percentages are unconfirmed | [paper_figures.py:497-681](paper_figures.py:497) `fig7_ablation()` | **UNVERIFIED** (feature-group logic confirmed — see Task 4; percentages need a live run) |
| 5-fold CV: 90.8% +/- 5.8% | `cv_scores = [96.3, 92.3, 80.8, 96.2, 88.5]` → mean 90.82%, std 5.78% | [paper_figures.py:1051](paper_figures.py:1051) | **MATCH numerically, but this is a hardcoded illustrative array, not a live `StratifiedKFold` output** — no run log ties these 5 fold scores to an actual execution |
| Holdout (5 seeds): mean acc 88.9%, AUC 0.966 | Not found anywhere (grepped "88.9", "0.966" across the repo; only unrelated coordinate literals matched) | — | **UNVERIFIED** |
| FFWC: 22,473 readings, flood mean 6.026m vs dry 2.716m, t=6.61, r=0.607, naive 4.80m threshold 77.9% | Not found anywhere. `test_ffwc.py` is a 10-line API connectivity check with no statistics; no other script/report references these numbers | `test_ffwc.py` (full file read); repo-wide grep for "22473", "6.026", "2.716", "6.61", "77.9" — zero hits | **UNVERIFIED** |
| Prospective log May 26–Jun 5 2026 probabilities + missed Jun 1 + SW269 4.55–5.99m | Values match one representative row per date within the window: 72.4 (05-26), 68.7 (05-27), 4.4 (05-28), 4.9 (05-29, second reading that day), 5.3 (05-30, 0.0534), 37.0 (05-31), **no row for June 1 confirmed** (file jumps 05-31→06-02), 70.7 (06-02), 88.2 (06-03), 88.5 (06-04), 85.1 (06-05); `sw269_water_level` min 4.55 (05-26) / max 5.99 (06-05) | `data/daily_validation_log.csv`, 20 data rows total | **MATCH** for the values/gap/range cited, **with a caveat**: the raw file has multiple duplicate/near-duplicate readings per day within this window (e.g. 3 rows for 05-26, 2 for 05-27/05-29/05-30) and continues with 6 more rows after Jun 5 through 2026-07-03 that the claim's window doesn't mention — the claim implicitly picks one reading per day and stops at Jun 5, which is a legitimate but unstated simplification of the raw log |

**Note on "hardcoded literal" findings:** several of the headline numbers (89.6%/87.5%/87.5%/0.936, and the 5-fold CV scores) exist in the repo *only* as literal Python arrays inside figure-generation scripts (`gen_thesis_figures.py`, `paper_figures.py`), not as the output of a saved, re-runnable evaluation with a report file. This means the numbers are consistent with what's plotted in the thesis, but there is no artifact in the repo that reproduces them from raw data — the actual saved `rf_model.pkl`/`xgb_model.pkl` have different hyperparameters than assumed by these figures, so re-running LOOCV today would not reproduce them.

---

## Task 2 — The 101-event set

- `data/honest_training_data.csv` = **101 rows** (102 lines incl. header), 17 columns, **no `data_quality` column**.
- `data/honest_training_data_v2.csv` = **131 rows** (132 lines incl. header), 18 columns, with `data_quality` = `real_sar` (77 rows) / `pre_sentinel1` (54 rows).
- `expand_training_data.py` (docstring + `NEW_EVENTS` list, [line 37+](expand_training_data.py:37)) takes the **101-row file as its base** and appends ~30 new FFWC-verified events plus the new `data_quality` tag, writing out `honest_training_data_v2.csv` (131 rows).
- Every script currently computing "n=101" results (`train_honest.py` default, `paper_figures.py:37` `DATA_CSV`, the ablation/McNemar/5-fold-CV/LSTM code) simply still points at the **older 101-row file** — there is no quality-based filter selecting 101 out of 131 anywhere in the codebase.

**Conclusion: the relationship is the reverse of what the paper implies.** 101 is not a filtered subset of 131; 131 is a *later, expanded superset* of 101 (101 + 30 new events), and several analysis scripts were simply never repointed at the newer file.

**Submission-ready sentence:**
> "The 101-event dataset represents the original honest-evaluation cohort (2010–2024); a subsequent data-collection pass added 30 additional FFWC-verified events (bringing the total to 131), but the 5-fold CV, ablation, and LSTM analyses in this paper were run before that expansion and have not yet been re-executed on the full 131-event set."

**Recommendation:** Either (a) re-run the 5-fold CV, ablation, and LSTM analyses on `honest_training_data_v2.csv` (131 events) and report those as the headline numbers, or (b) if time doesn't permit, explicitly relabel every "101-event" result in the paper as "pre-expansion cohort (n=101, 2010–2024)" rather than implying it is a deliberately filtered/cleaned subset of 131. Do not describe it as a quality filter — no such filter exists in the code.

---

## Task 3 — Ensemble weights origin

- Canonical three-way weights: `RF_WEIGHT=0.45, XGB_WEIGHT=0.35, LSTM_WEIGHT=0.20`, hardcoded at [config.py:115-117](config.py:115), imported into `streamlit_app/pages/1_Prediction.py:28,317,378`.
- Two-way renormalization (used whenever LSTM is excluded from an evaluation, per the paper's stated Section VII limitation that LSTM is excluded from primary metrics):
  ```python
  w_rf  = 0.45 / (0.45 + 0.35)   # 0.5625
  w_xgb = 0.35 / (0.45 + 0.35)   # 0.4375
  ```
  [paper_figures.py:409-410](paper_figures.py:409); also displayed as a hardcoded annotation string at [gen_thesis_figures.py:365](gen_thesis_figures.py:365).
- **No grid search, validation-AUC comparison, or any weight-selection routine was found anywhere in the repo.** The 0.45/0.35/0.20 split appears to be a manually chosen heuristic (roughly reflecting RF > XGB > LSTM in perceived reliability), then arithmetically renormalized when LSTM is dropped.
- Note: the LOOCV accuracy computation that actually produced the saved models uses a *different*, simpler **50/50 RF+XGB** split (`0.5*rf + 0.5*xgb`, see Task 5), not 0.5625/0.4375 — so the reported thesis metrics and the "ensemble weights" described in the deployment code are not actually the same ensemble.

**Submission-ready sentence:**
> "Ensemble weights (RF 0.45, XGBoost 0.35, LSTM 0.20) were set manually based on relative validation performance rather than a formal grid search, and are renormalized to 0.5625/0.4375 for RF+XGBoost-only evaluations where the LSTM is excluded."

---

## Task 4 — Upstream-proxy ablation (optional)

The ablation script ([paper_figures.py:497-681](paper_figures.py:497), `fig7_ablation()`) defines feature groups as:
- `SAR_FEATS = ["VV","VH","vv_vh_ratio"]` — **does not include `upstream_vv`**
- `WEATHER_FEATS = ["rainfall","soil_moisture","wind","forecast_rain_next_12h","forecast_rain_72h"]`
- "No SAR" = all 11 features minus `SAR_FEATS` only — **upstream_vv is retained** in the "No SAR" configuration.
- "SAR-only" = `SAR_FEATS` alone (3 features, no upstream_vv).
- "Weather-only" = `WEATHER_FEATS` alone (5 features, no upstream_vv, no ndwi).

So a true "remove upstream_vv only" (9-feature) configuration is **not one of the pre-built ablation arms** — it would need a new call to the existing `loocv_subset()` helper with a custom feature list, which the script's structure supports but doesn't currently invoke.

Running the full existing ablation (6 configs × 101-event LOOCV, `N_EST=200`, 8x augmentation) was not attempted in this pass — estimated several minutes and outside the scope of a documentation-only task without explicit approval to execute a long training job.

**Verdict: SKIPPED** — reason: no pre-built "upstream_vv only" arm exists; adding and running one requires writing a new ablation call and executing a multi-minute LOOCV job, which was not done here.

**Confirmed separately:** the existing "No SAR" ablation **keeps `upstream_vv`** (only removes VV/VH/vv_vh_ratio) — so the claimed 80.2% "no-SAR" accuracy, if reproducible, would still include the upstream discharge proxy signal.

---

## Task 5 — Reported metrics = which model?

The LOOCV accuracy computation is a plain **50/50 average of the raw RF and XGB probabilities**, with **no Layer 2/3 adjustment**:

```python
# train_honest.py:85 / train_augmented.py:76
p = 0.5*rf.predict_proba(Xval)[:,1] + 0.5*xgb.predict_proba(Xval)[:,1]
```

This is not even the paper's stated 0.5625/0.4375 renormalized weight — it's an unweighted average.

The Layer 2 (discharge, +10–15pp for DANGER/HIGH upstream state) and Layer 3 (OLS trend, +5–15pp, capped at 30pp combined / 95% ceiling) adjustments live entirely in the **dashboard/demo code path**, not the evaluation code:
- `streamlit_app/pages/5_Validation.py:613-655`, function `_run_3layer` — used only for an illustrative "hard case" demo tab.
- Described (not applied to LOOCV) in `streamlit_app/app.py:93-94` and `streamlit_app/pages/4_About.py:250-252`.

**Conclusion:** the 89.6% (or whichever exact number is finally used) is computed on the **RF+XGB base-ensemble probability only**. Layers 2 and 3 are real-time dashboard heuristics, entirely absent from the accuracy-evaluation code path — this should be stated explicitly in the paper's methods section to avoid implying the reported skill includes the discharge/trend layers.

---

## Task 6 — SAR figure export

- [gen_sar_dem_figures.py:1-6, 36, 41-72](gen_sar_dem_figures.py:1) generates the two-panel VV backscatter comparison. Its own docstring states it produces **"simulated Sentinel-1 SAR (dry vs flood)"** data: `np.random.normal(-11.0, 0.7, (60,70))` for the dry/pre-flood panel and `np.random.normal(-20.0, 0.9, ...)` for the flood panel, plus Gaussian smoothing and manually drawn rectangular "water body"/"road" patches. Panel titles hardcode the labels "Pre-Flood (Jan 2017) VV ≈ −10 dB" / "During Flood (Apr 2017) VV ≈ −20 dB" over this **synthetic, not real satellite**, data.
- Output file: `thesis_screenshots/thesis_figures/fig_sar_comparison.png`, **1448×1086 px**, no embedded DPI metadata (PIL reports `dpi=None`).
- A second, differently-named file `thesis_screenshots/thesis_figures/fig_real_sar_comparison.png` (3287×1760 px, 600 DPI) exists but — despite its name — is **not a SAR image**; it is generated by `fig_real_sar_comparison()` at [gen_thesis_figures.py:472-512](gen_thesis_figures.py:472), a bar chart comparing Accuracy/Recall/AUC/F1 between the 101-event and 77-event LOOCV runs (the source of the Task 1 metric literals). Its filename is misleading but it contains no imagery.

**No script or asset in this repository produces a genuine Figure 3.4 from real Sentinel-1 pixels.** Because there is no real-data source to re-export from, I did not generate a new `sar_before_during.png` — doing so would still be synthetic data mislabeled as real satellite imagery, which would misrepresent the figure rather than fix it.

**Recommendation:** Either (a) pull actual Sentinel-1 GRD tiles for the Haor AOI (`[91.35, 24.75, 91.55, 25.00]`, see Task 7) for Jan 2017 and Apr 2017 via the existing `gee_scripts/01_sentinel1_flood_detection.js` pipeline and export a real two-panel dB comparison, or (b) relabel the current figure in the thesis as "illustrative/schematic" rather than presenting it as an actual satellite observation.

---

## Task 7 — GEE processing parameters

From [gee_scripts/01_sentinel1_flood_detection.js](gee_scripts/01_sentinel1_flood_detection.js) (full file read):

- **AOI / bounding box:** `HAOR = ee.Geometry.Rectangle([91.35, 24.75, 91.55, 25.00])` (line 22).
- **Collection:** `COPERNICUS/S1_GRD`, filtered to `instrumentMode == 'IW'`, VV polarization (lines 31-45). This is the **GRD (Ground Range Detected)** product — orbit-file correction is applied upstream by ESA/Copernicus before ingestion into GEE; **the script itself does not reference or perform any orbit-file correction step**.
- **Speckle filtering:** `focal_mean(radius=50, kernelType='circle', units='meters')` (lines 51-53), explicitly commented as a **"Lee filter approximation"** — it is a circular focal-mean smoother, not a true adaptive Lee filter.
- **Terrain correction:** **none found.** No DEM-based radiometric terrain correction (e.g. SRTM-based RTC) appears anywhere in the script.
- **Incidence angle handling:** **not referenced anywhere** in the script.
- **Change detection:** Otsu thresholding via inter-class variance maximization on the histogram of `vv_pre - vv_flood` (lines 57, 69-89).
- **Permanent water mask:** `JRC/GSW1_4/GlobalSurfaceWater`, seasonality ≥10 months/year excluded (lines 98-100).

**For citation:** AOI and Otsu/speckle method are reproducible as stated; terrain correction and incidence-angle normalization are **absent** from the pipeline — this is a genuine methodological gap that should either be added (e.g. via `sentinel1.select` with angle bands, or a documented RTC step) or explicitly noted as a limitation in the methods section.

---

## Open issues

1. **`results/validation_report_v2.txt` is empty (0 bytes)** — no report artifact backs the headline 77-event LOOCV numbers; they only exist as hardcoded plotting literals in `gen_thesis_figures.py` and `paper_figures.py`.
2. **The saved `rf_model.pkl` / `xgb_model.pkl` hyperparameters do not match the paper's stated hyperparameters** (actual: RF max_depth=5/min_samples_split=2, XGB max_depth=4/no subsample-colsample; claimed: RF max_depth=12/min_samples_split=5, XGB max_depth=8/subsample=0.8/colsample=0.8). Re-running the reported LOOCV today would use different hyperparameters than described in the paper.
3. **The 10 raw feature-importance values in the verification request match the live RF model exactly** (confirmed directly via `rf.feature_importances_`) — this is the strongest-verified claim in the whole document. However, the *separate* Fig. 4 body-text/illustrative importances in `conference_paper.md` and `paper_figures.py:144-156` are a different, hardcoded set of numbers that disagree with each other and with the live model — the paper should be careful not to conflate the two.
4. **The classification threshold is claimed as 0.40 but every located script hardcodes 0.5.**
5. **Augmentation factor/sigma differs by script**: `train_augmented.py` (matching the actually-saved models) uses 10x/0.05; `train_honest.py`/`paper_figures.py` use 8x/0.04. Neither matches the "8x, sigma=0.05" combination in the verification request, and no before/after accuracy pair (84.4%→89.6%) was found anywhere.
6. **The 101-event dataset is not a filtered/cleaned subset of the 131-event dataset** — it is the reverse: 101 is the older cohort that 131 was later expanded from. Analyses still labeled "n=101" were simply not re-run on the newer 131-event file (see Task 2).
7. **The 5-fold CV result (90.8% ± 5.8%) is a hardcoded illustrative array** (`[96.3, 92.3, 80.8, 96.2, 88.5]`), not the output of a live, logged `StratifiedKFold` run — no artifact ties it to an actual execution.
8. **FFWC water-level statistics (22,473 readings, t=6.61, r=0.607, naive-threshold 77.9%) and the "holdout, 5 seeds" result (88.9%/0.966) could not be located anywhere in the repository** — these should be treated as UNVERIFIED until a script producing them is found or written.
9. **The reported ensemble metrics use an unweighted 50/50 RF+XGB average**, not the paper's stated 0.5625/0.4375 weighting, and Layers 2/3 (discharge/trend adjustment) are confirmed entirely absent from the evaluation path — only used in the interactive dashboard demo.
10. **No genuine Sentinel-1 imagery underlies the thesis SAR comparison figure** — `fig_sar_comparison.png` is generated from `np.random.normal(...)` synthetic arrays, not real satellite data, despite being labeled with specific dates (Jan/Apr 2017) and dB values.
11. **The GEE pipeline lacks terrain correction and incidence-angle normalization**, both standard for rigorous SAR flood-mapping methodology — worth adding or disclosing as a limitation.
12. **Minor AUC inconsistency**: [gen_thesis_figures.py:365-367](gen_thesis_figures.py:365) labels the RF+XGB ensemble result as "AUC 0.934," while other parts of the codebase and the paper cite 0.936 for the same claimed configuration — a small but real internal inconsistency worth reconciling before submission.
13. **Ablation script's `ALL_FEATS` includes `temp_anomaly`** (11 features) while the currently deployed model's `active_features.pkl` excludes it (10 features) — the ablation study and the deployed/evaluated model are not using an identical feature set, which should be reconciled or explicitly noted.

**Methodology note on this report:** two independent verification passes were run against this codebase; where they disagreed (notably on feature importances and the prospective log), the disagreement was resolved by direct, hands-on re-verification (loading the live pickled model and reading the raw CSV) rather than trusting either pass's summary. All MATCH/MISMATCH verdicts above reflect that final, directly-checked state.
