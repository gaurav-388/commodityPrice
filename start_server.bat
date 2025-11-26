@echo off
echo ========================================
echo Commodity Price Prediction - Server
echo ========================================
echo.
echo Stopping any existing Flask servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *flask*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting Flask server...
echo.
cd /d "%~dp0"
call conda activate tf_env
python app.py

echo.
echo Server stopped.
pause
