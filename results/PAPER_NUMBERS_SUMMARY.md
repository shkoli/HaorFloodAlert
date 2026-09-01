# Paper Numbers Summary

Generated live by `eval/run_paper_results.py` at 2026-07-03T12:30:03.136939+00:00. Every value below is traceable to a key in `results/paper_results.json` (path shown in parentheses). Nothing here was tuned to match a previously reported number.

| # | Number | Value | JSON path |
|---|---|---|---|
| 1 | Total events | 131 | `dataset.n_total_events` |
| 2 | Real-SAR events (primary set) | 77 | `dataset.n_real_sar_events` |
| 3 | Proxy (pre-Sentinel-1) events | 54 | `dataset.n_proxy_pre_sentinel1_events` |
| 4 | LOOCV accuracy @0.50, with augmentation | 90.9% (70/77) | `loocv_77_real_sar.with_augmentation.metrics_threshold_0.50` |
| 5 | LOOCV precision @0.50, with augmentation | 87.9% | same |
| 6 | LOOCV recall @0.50, with augmentation | 90.6% | same |
| 7 | LOOCV F1 @0.50, with augmentation | 89.2% | same |
| 8 | LOOCV specificity @0.50, with augmentation | 91.1% | same |
| 9 | LOOCV AUC, with augmentation | 0.939 | `loocv_77_real_sar.with_augmentation.auc` |
| 10 | LOOCV confusion matrix @0.50 | TN=41 FP=4 FN=3 TP=29 | same |
| 11 | LOOCV accuracy @0.40, with augmentation | 89.6% | `loocv_77_real_sar.with_augmentation.metrics_threshold_0.40` |
| 12 | LOOCV accuracy @0.50, WITHOUT augmentation | 85.7% | `loocv_77_real_sar.without_augmentation.metrics_threshold_0.50` |
| 13 | LOOCV F1 @0.50, WITHOUT augmentation | 82.0% | same |
| 14 | LOOCV AUC, WITHOUT augmentation | 0.922 | `loocv_77_real_sar.without_augmentation.auc` |
| 15 | 5-fold CV, 131 events: mean accuracy | 88.5% +/- 4.3% | `five_fold_cv.full_131_events` |
| 16 | 5-fold CV, 77 real-SAR events: mean accuracy | 88.2% +/- 6.5% | `five_fold_cv.real_sar_77_events` |
| 17 | Holdout (5 seeds, 60/40): mean accuracy | 81.3% +/- 6.6% | `holdout_77_real_sar` |
| 18 | Holdout (5 seeds, 60/40): mean AUC | 0.918 +/- 0.049 | `holdout_77_real_sar` |
| 19 | Deconfounding: raw temp vs label r | 0.5704 | `deconfounding.raw_temp_vs_label` |
| 20 | Deconfounding: raw temp vs month r | 0.4000 | `deconfounding.raw_temp_vs_month` |
| 21 | Deconfounding: temp_anomaly vs label r | -0.0305 | `deconfounding.temp_anomaly_vs_label` |
| 22 | Ablation -- all_10_features_baseline (10 feats) | Acc=89.3%, AUC=0.942 | `ablation_131_events.arms.all_10_features_baseline` |
| 23 | Ablation -- no_sar_upstream_vv_kept (7 feats) | Acc=87.8%, AUC=0.939 | `ablation_131_events.arms.no_sar_upstream_vv_kept` |
| 24 | Ablation -- no_rain_forecasts (7 feats) | Acc=87.0%, AUC=0.946 | `ablation_131_events.arms.no_rain_forecasts` |
| 25 | Ablation -- no_soil_moisture (9 feats) | Acc=88.5%, AUC=0.949 | `ablation_131_events.arms.no_soil_moisture` |
| 26 | Ablation -- sar_only (3 feats) | Acc=71.0%, AUC=0.795 | `ablation_131_events.arms.sar_only` |
| 27 | Ablation -- weather_only (5 feats) | Acc=79.4%, AUC=0.871 | `ablation_131_events.arms.weather_only` |
| 28 | Ablation -- all_minus_upstream_vv_only (9 feats) | Acc=88.5%, AUC=0.942 | `ablation_131_events.arms.all_minus_upstream_vv_only` |
| 29 | FFWC: readings in gauge file | 22473 | `ffwc_sw269_gauge.n_readings_total_in_gauge_file` |
| 30 | FFWC: dates matched to labeled events | 97 | `ffwc_sw269_gauge.n_dates_matched_to_labeled_events` |
| 31 | FFWC: flood mean water level | 5.773 m (n=43) | `ffwc_sw269_gauge.flood_mean_water_level_m` |
| 32 | FFWC: dry mean water level | 2.762 m (n=54) | `ffwc_sw269_gauge.dry_mean_water_level_m` |
| 33 | FFWC: t-statistic | 6.77 (p=1.09e-09) | `ffwc_sw269_gauge.t_test` |
| 34 | FFWC: point-biserial r | 0.562 | `ffwc_sw269_gauge.point_biserial_r` |
| 35 | FFWC: naive best threshold | 2.65 m | `ffwc_sw269_gauge.naive_best_threshold_m` |
| 36 | FFWC: naive threshold accuracy (in-sample) | 76.3% | `ffwc_sw269_gauge.naive_threshold_accuracy` |
| 37 | Baseline: majority-class accuracy | 58.4% | `baselines_77_real_sar.majority_class` |
| 38 | Baseline: rainfall-threshold-rule accuracy | 75.3% | `baselines_77_real_sar.rainfall_threshold_rule` |
| 39 | Baseline: logistic regression accuracy | 80.5% (AUC=0.912) | `baselines_77_real_sar.logistic_regression` |
| 40 | SAR Figure 3.4 (real Sentinel-1) | SCRIPT_GENERATED_NOT_RUN -- see `gee_scripts\export_fig_sar_real.js` | `sar_figure_fig3_4` |

## Notes

- Model config, augmentation factor/sigma, ensemble weights, and feature list are documented in `paper_results.json.config` and were chosen to match the **currently saved** models (`models/rf_model.pkl`, `models/xgb_model.pkl`, `models/active_features.pkl`), not any previously reported paper number.
- The ablation study runs on the full 131-event set, per instruction; the primary LOOCV/holdout numbers above run on the 77-event real-SAR subset.
- FFWC naive-threshold accuracy is an in-sample descriptive statistic over dates that could be matched between the gauge file and the labeled dataset (not a cross-validated claim) -- see the note field in the JSON.
- Figure 3.4 requires manually running `gee_scripts/export_fig_sar_real.js` in the GEE Code Editor; no synthetic substitute was generated.