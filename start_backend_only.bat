@echo off
echo ========================================
echo GARUD Backend - Manual Start
echo ========================================
echo.
echo This script will start ONLY the backend server
echo Use this for debugging backend issues
echo.

cd /d "%~dp0"

:: Navigate to API directory
echo [1/3] Navigating to API directory...
cd "full model\src\api"
if errorlevel 1 (
    echo ERROR: Could not navigate to API directory
    pause
    exit /b 1
)
echo Current directory: %CD%
echo.

:: Activate virtual environment
echo [2/3] Activating Python virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_backend.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

:: Start FastAPI server
echo [3/3] Starting FastAPI server...
echo.
echo ========================================
echo Backend server starting...
echo API will be available at: http://localhost:8000
echo API Docs at: http://localhost:8000/docs
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python main.py

:: If we get here, the server stopped
echo.
echo ========================================
echo Backend server stopped
echo ========================================
pause
