@echo off
echo ================================================
echo   HaorFloodAlert - Starting Dashboard
echo ================================================
echo.
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop the app.
echo.
cd /d C:\Users\Lenovo\HaorFloodAlert
streamlit run streamlit_app\app.py
pause
