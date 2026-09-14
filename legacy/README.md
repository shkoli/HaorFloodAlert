# legacy/

This directory holds historical development scripts from earlier stages of the project.
They are **not used by the paper's evaluation pipeline** (`eval/run_paper_results.py`) or by
the live app (`app.py`, `daily_update.py`, `streamlit_app/`). They are kept for transparency
and to preserve the development history, not as maintained or supported code.

Several of these scripts (in particular the `train_*`, `collect_*`, and `relabel_and_train.py`
files) were written against the superseded 101-row `honest_training_data.csv` dataset and
predate the 131-row v2 dataset used for the published results. Running them as-is will not
reproduce the paper's numbers.

`paper_figures.py` is stale for the same reason: its `DATA_CSV` points at the old 101-row
file, not the 131-row v2 dataset.

Note: `retrain_all.bat` (at the repo root) still calls `retrain_model.py` and `train_lstm.py`
by their old root-level paths. That batch file has not been updated for this move and will
need its paths fixed (or removed) if it's still in use.

## lstm/

`lstm_model.h5`, `lstm_model.pth`, and `lstm_scaler.pkl` are **not** in this folder — they are
still actively loaded by `streamlit_app/pages/1_Prediction.py` and were left in `models/`.

This folder contains only `lstm_window.pkl`, which has no remaining consumer in the repo.

The LSTM component was excluded from the published ensemble because its walk-forward
evaluation showed 100% accuracy (AUC 1.000), which was diagnosed as a memorisation artefact
rather than genuine predictive skill (the dataset is small enough that the LSTM could
effectively memorise sequences). These artefacts are retained here for transparency only —
they are not part of the ensemble that produced the published results.
