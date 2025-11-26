@echo off
echo ========================================
echo Starting Commodity Price Server
echo ========================================
echo.

cd /d "%~dp0"

echo Checking for existing server on port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    echo Killing existing process: %%a
    taskkill /PID %%a /F >nul 2>&1
    timeout /t 2 >nul
)

echo.
echo Activating conda environment...
call C:\Users\acer\anaconda3\Scripts\activate.bat tf_env

echo.
echo ========================================
echo Starting server...
echo ========================================
echo.

python run_server.py

echo.
echo Server stopped.
pause
