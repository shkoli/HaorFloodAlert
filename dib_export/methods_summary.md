# Methods Summary

Compiled from reading pipeline source code only (`config.py`, `collect_honest_data.py`, `expand_training_data.py`, `gee_scripts/`, `eval/run_paper_results.py`, `utils/upstream_discharge.py`, `add_discharge.py`). No raw data folders or notebook outputs were inspected.

## 1. How flood events were identified

Event dates and binary flood/non-flood labels are hardcoded lists in `collect_honest_data.py` (25 events, 2017–2024) and `expand_training_data.py` (30 additional events, 2009–2024), each tied to a citation string stored in the `source` column (FFWC annual reports, BWDB bulletins, and peer-reviewed literature such as Mondal et al. 2021). Labels are assigned independently of the satellite-derived features in the same row, to avoid circularity between features and labels.

The subset delivered here (`dataset.csv`, 77 rows) is filtered to `data_quality == 'real_sar'`, i.e. events from 2014 onward where Sentinel-1 had usable coverage and the satellite/weather features were fetched from real imagery rather than synthesized. The remaining 54 rows in the source file (`pre_sentinel1`) use calibrated synthetic SAR profiles for pre-2014 events and are excluded from this export.

## 2. SAR verification

- **Satellite:** Sentinel-1 GRD, C-band, IW mode (`gee_scripts/01_sentinel1_flood_detection.js`).
- **Method:** Otsu automatic bi-modal thresholding applied to a VV change-detection image (pre-flood minus post-flood composite, in dB). A pixel is classified as flooded if VV drops by more than ~3 dB from the dry-season baseline (script line 56); the script notes a typical operating range of 3–6 dB for haor flash floods (line 144).
- **Cross-validation:** `gee_scripts/02_sentinel2_ndwi_mapping.js` computes Sentinel-2 NDWI and flags NDWI > 0.3 as permanent/flood open water (line 136), used to sanity-check the SAR-derived flood extent.
- **Feature extraction:** For each labeled event date, `collect_honest_data.py::get_s1()` pulls the nearest available Sentinel-1 VV/VH mean over the haor AOI at 30 m scale (`config.py::SAR_SCALE`).

## 3. Feature sources

| Feature(s) | Source API/dataset | Script | Processing |
|---|---|---|---|
| VV, VH, vv_vh_ratio | Sentinel-1 GRD (Google Earth Engine) | `collect_honest_data.py::get_s1()` | Mean backscatter over haor AOI, 30 m scale; ratio computed as VV/VH. |
| upstream_vv | Sentinel-1 GRD, Barak River AOI | `collect_honest_data.py::get_upstream()` | Same as above but centered on the upstream Barak/Silchar bounding box. |
| ndwi | Sentinel-2 (GEE) | `gee_scripts/02_sentinel2_ndwi_mapping.js`, config defaults | (Green−NIR)/(Green+NIR), 20 m scale (`config.py::S2_SCALE`). |
| rainfall, forecast_rain_next_12h, forecast_rain_72h | ERA5-Land archive / Open-Meteo | `collect_honest_data.py::get_era5()`, `get_forecast()` | 7-day cumulative precip (past), 12h/72h cumulative precip (forecast). |
| soil_moisture, temp, wind | ERA5-Land | `collect_honest_data.py::get_era5()` | Layer-1 volumetric soil moisture, 2 m temp, 10 m wind, at ERA5 scale (`config.py::ERA5_SCALE` = 11132 m). |
| slope, twi | USGS SRTM 30 m DEM | `config.py`, `collect_honest_data.py` | Static per-AOI values (not recomputed per event): slope = 1.91°, TWI = 17.185, at `config.py::DEM_SCALE` = 30 m. |
| temp_anomaly | Derived from `temp` | `expand_training_data.py`, `config.py::TEMP_CLIMATOLOGY` | temp minus the monthly climatological mean for that calendar month. |
| data_quality | Provenance flag, not a physical feature | `expand_training_data.py` | `"real_sar"` if event year ≥ 2014 and VV ≠ default placeholder, else `"pre_sentinel1"`. |

## 4. All hardcoded parameters found

**Coordinates / bounding boxes** (`config.py`):
- Haor AOI bounding box: `[91.35, 24.75, 91.55, 25.00]`; center point 24.87°N, 91.45°E.
- Upstream (Barak/Silchar) bounding box: `[92.70, 24.60, 93.20, 25.00]`; center point 24.80°N, 92.95°E.
- `utils/upstream_discharge.py` uses 24.82°N, 92.79°E for Barak at Silchar.
- `add_discharge.py` uses 24.87°N, 91.40°E (Surma gauge point).

**Timing:**
- Upstream-to-haor flood travel time: 36 hours (`config.py`).

**Discharge thresholds** (`config.py`, Barak/Silchar): DANGER = 7500 m³/s, HIGH = 6000 m³/s, WARNING = 4000 m³/s, DEFAULT = 450.0 m³/s. `add_discharge.py` default Surma discharge = 20.0 m³/s.

**Risk classification thresholds** (`config.py`): EXTREME = 0.85, HIGH = 0.65, MEDIUM = 0.40 (model output probability cutoffs).

**NDWI / TWI calibration means** (`config.py`): NDWI_FLOOD_MEAN = 0.25, NDWI_DRY_MEAN = -0.18, TWI_FLOOD_MEAN = 15.0, TWI_DRY_MEAN = 8.5.

**Static terrain constants:** slope = 1.91°, TWI = 17.185 (`config.py`, `expand_training_data.py`).

**Synthetic SAR event-type profiles** (`expand_training_data.py`, used only for the `pre_sentinel1` rows, not in this 77-row export): flash-flood VV=-14.5/VH=-20.5/ratio=0.707/ndwi=-0.35/upstream_vv=-20.5; monsoon VV=-18.5/VH=-23.5/ratio=0.787/ndwi=0.10/upstream_vv=-17.0; dry VV=-11.0/VH=-18.0/ratio=0.611/ndwi=-0.42/upstream_vv=-10.5, plus per-field jitter standard deviations.

**Model hyperparameters** (`eval/run_paper_results.py`): Random Forest — n_estimators=500, max_depth=5, min_samples_split=2; XGBoost — n_estimators=500, max_depth=4, learning_rate=0.05; data-augmentation factor=8, noise sigma=0.04; probability-threshold constants 0.50 and 0.40.

**Pixel scales** (`config.py`): SAR_SCALE = 30 m, S2_SCALE = 20 m, CHIRPS_SCALE = 5560 m, ERA5_SCALE = 11132 m, DEM_SCALE = 30 m.

**Event dates:** 25 hardcoded dates in `collect_honest_data.py` (2017–2024) and 30 hardcoded dates in `expand_training_data.py` (2009–2024), each an explicit `"YYYY-MM-DD"` string tied to a labeled event.

**SAR flood-detection threshold:** Otsu-derived automatically per scene, but the script documents an expected range of 3–6 dB VV drop for haor flash floods (`gee_scripts/01_sentinel1_flood_detection.js`); NDWI cross-check threshold > 0.3 (`gee_scripts/02_sentinel2_ndwi_mapping.js`).

**Station IDs:** No numeric BWDB/FFWC gauge station codes were found anywhere in the codebase; stations are referenced by name/location only (e.g. "Surma at Sunamganj", "Barak at Silchar").
