@echo off
:: Auto create and activate Python venv + install requirements.txt
:: Usage: double click or run in CMD: save_as_venv.bat

set VENV_NAME=venv

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    pause
    exit /b
)

:: Create venv if it does not exist
if not exist %VENV_NAME% (
    echo [INFO] Creating virtual environment "%VENV_NAME%"...
    python -m venv %VENV_NAME%
) else (
    echo [INFO] Virtual environment "%VENV_NAME%" already exists.
)

:: Activate venv (CMD)
call %VENV_NAME%\Scripts\activate.bat

:: Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo [INFO] Found requirements.txt, installing dependencies...
    pip install -r requirements.txt
) else (
    echo [INFO] No requirements.txt found, skipping dependencies installation.
)

echo.
echo [DONE] Virtual environment is now active.
echo [HINT] Type "deactivate" to exit the venv.
echo.

:: Open interactive CMD with venv active
cmd
