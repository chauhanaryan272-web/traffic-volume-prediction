@echo off
title Traffic Volume Prediction
echo ============================================
echo   Starting Traffic Volume Prediction System...
echo ============================================
echo.

REM -- Locate a working Python interpreter --
set PYTHON=
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=python
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON=py
    ) else (
        echo [ERROR] Python is not installed or not on PATH.
        echo         Please install Python 3.8+ and try again.
        pause
        exit /b 1
    )
)
echo Found Python: %PYTHON%

REM -- Create virtual environment if it doesn't exist --
if not exist ".venv" (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
)

REM -- Activate virtual environment --
call .venv\Scripts\activate.bat

REM -- Install dependencies --
echo Installing requirements...
pip install -r requirements.txt --quiet

REM -- Train model if not already present --
if not exist "models\traffic_volume_model.joblib" (
    echo.
    echo Generating dataset and training model...
    %PYTHON% scripts/prepare_data.py
    %PYTHON% scripts/train_model.py
)

echo.
echo ============================================
echo   Starting Flask application...
echo ============================================
start http://localhost:5000
%PYTHON% app.py
pause
