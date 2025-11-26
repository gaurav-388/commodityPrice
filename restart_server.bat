@echo off
echo ========================================
echo RESTARTING FLASK SERVER
echo ========================================
echo.

echo Killing any running Python/Flask processes...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting Flask server with updated code...
echo.
cd /d "c:\Users\acer\Desktop\btp1"
start cmd /k "conda activate tf_env && python app.py"

echo.
echo ========================================
echo Server restarted in new window!
echo ========================================
echo.
echo Now do the following:
echo 1. Go to your browser
echo 2. Press Ctrl+Shift+R (hard refresh)
echo 3. Try selecting district and commodity
echo.
echo If still not working, close browser completely and reopen
echo.
pause
