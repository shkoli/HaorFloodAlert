@echo off
chcp 65001 >nul
cd /d %~dp0
set PYTHONIOENCODING=utf-8
python daily_validation.py >> logger_output.txt 2>&1

echo.
if %ERRORLEVEL%==0 (
    echo === Daily update finished OK ===
) else (
    echo === Daily update FAILED (exit code %ERRORLEVEL%) ===
)
echo.
echo Last result:
echo.
powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Content -Path 'logger_output.txt' -Tail 15 -Encoding UTF8"
echo.
pause
