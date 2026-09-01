@echo off
echo ================================================
echo   HaorFloodAlert - Installing Dependencies
echo ================================================
echo.

pip install streamlit earthengine-api pandas numpy scikit-learn xgboost joblib requests folium streamlit-folium plotly torch torchvision

echo.
echo ================================================
echo   Done! Now run: start_app.bat
echo ================================================
pause
