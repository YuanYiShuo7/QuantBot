@echo off
setlocal enabledelayedexpansion

set "PYTHON_CMD=python"
if not "%~1"=="" (
    set "PYTHON_CMD=%~1"
)

echo Setting up virtual environment in .venv using "%PYTHON_CMD%"...

where "%PYTHON_CMD%" >nul 2>nul
if errorlevel 1 (
    echo Python command "%PYTHON_CMD%" not found. Please install Python 3.9+ and ensure it is on PATH.
    exit /b 1
)

set "VENV_DIR=.venv"
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo .venv already exists. Skipping creation.
) else (
    "%PYTHON_CMD%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b %errorlevel%
    )
    echo Virtual environment created at .venv.
)

set "PIP_PATH="
if exist "%VENV_DIR%\Scripts\pip.exe" (
    set "PIP_PATH=%VENV_DIR%\Scripts\pip.exe"
)

if defined PIP_PATH (
    echo Upgrading pip in the virtual environment...
    "%PIP_PATH%" install --upgrade pip

    if exist "requirements.txt" (
        echo Installing dependencies from requirements.txt...
        "%PIP_PATH%" install -r requirements.txt
    ) else (
        echo requirements.txt not found in the current directory; skipping dependency installation.
    )
) else (
    echo pip executable not found in .venv. Please check the environment.
)

echo Done. Activate the environment with:
echo     call .\.venv\Scripts\activate.bat

