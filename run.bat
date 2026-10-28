@echo off
setlocal enabledelayedexpansion
title Traffic Volume Prediction System
cd /d "%~dp0"

echo ============================================================
echo   Traffic Volume Prediction System — Windows Setup & Launch
echo ============================================================
echo.

REM -- Step 1: Detect Python Interpreter --
set "PYTHON="

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=python"
    goto :python_found
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=py -3"
    goto :python_found
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=py"
    goto :python_found
)

echo [ERROR] Python is not installed or not found in system PATH.
echo         Please install Python 3.8 or higher from https://python.org/
echo.
pause
exit /b 1

:python_found
echo [INFO] Detected system Python: %PYTHON%

REM -- Step 2: Initialize Virtual Environment (.venv) --
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment (.venv)...
    %PYTHON% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
)

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "VENV_PIP=%~dp0.venv\Scripts\pip.exe"

REM -- Step 3: Install Required Dependencies --
echo [INFO] Verifying and installing dependencies from requirements.txt...
"%VENV_PIP%" install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Pip encountered a warning or non-critical error during installation.
)

REM -- Step 4: Ensure Trained Model & Datasets Exist --
if not exist "models\traffic_volume_model.joblib" (
    echo.
    echo [INFO] Model not found. Preparing dataset and training baseline pipeline...
    "%VENV_PYTHON%" scripts\prepare_data.py
    "%VENV_PYTHON%" scripts\train_model.py
)

REM -- Step 5: Start Application & Launch Web Browser --
echo.
echo ============================================================
echo   Starting Traffic Volume Prediction Web Application...
echo   URL: http://localhost:5000
echo ============================================================
echo.

start "" http://localhost:5000
"%VENV_PYTHON%" app.py

pause
