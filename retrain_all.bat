@echo off
echo ================================================
echo   HaorFloodAlert - Retraining All Models
echo ================================================
echo.
cd /d C:\Users\Lenovo\HaorFloodAlert

echo [1/2] Training RF + XGBoost...
python retrain_model.py
echo.

echo [2/2] Training LSTM...
python train_lstm.py
echo.

echo ================================================
echo   Retraining complete!
echo   Models saved to: models\
echo   Report saved to: results\training_report.txt
echo ================================================
pause
