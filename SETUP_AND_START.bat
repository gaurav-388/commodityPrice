@echo off
echo ========================================
echo SETUP COMMODITY PREDICTION SYSTEM
echo ========================================
echo.

REM Check if database exists
if exist commodity_prices.db (
    echo Database found: commodity_prices.db
    echo.
) else (
    echo Database not found. Creating database...
    echo.
    python init_database.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create database!
        pause
        exit /b 1
    )
    echo.
    echo [OK] Database created successfully!
    echo.
)

echo ========================================
echo STARTING FLASK SERVER
echo ========================================
echo.
echo Server will start at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
