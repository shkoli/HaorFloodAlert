@echo off
echo ================================================
echo   HaorFloodAlert - Retraining All Models
echo ================================================
echo.
cd /d C:\Users\Lenovo\HaorFloodAlert

echo [1/1] Training RF + XGBoost...
python legacy\retrain_model.py
echo.

REM LSTM retraining is intentionally not run here. The LSTM is excluded from
REM the published ensemble (its 100% walk-forward accuracy was a memorisation
REM artefact, not genuine skill). To retrain it anyway, run
REM legacy\train_lstm.py manually.

echo ================================================
echo   Retraining complete!
echo   Models saved to: models\
echo   Report saved to: results\training_report.txt
echo ================================================
pause
