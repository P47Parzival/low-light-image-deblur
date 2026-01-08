@echo off
echo ========================================
echo GARUD Backend Setup Script
echo ========================================
echo.

cd /d "%~dp0full model\src\api"

echo [1/5] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [4/5] Installing dependencies (this may take 10-15 minutes)...
echo This will download PyTorch, TensorFlow, and other ML libraries
pip install -r ..\..\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo [5/5] Initializing database...
python -c "import sys; sys.path.append('../core'); import database; database.init_db(); print('Database initialized successfully!')"
if errorlevel 1 (
    echo ERROR: Failed to initialize database
    pause
    exit /b 1
)
echo.

echo ========================================
echo Backend setup completed successfully!
echo ========================================
echo.
echo To start the backend server, run:
echo   cd "full model\src\api"
echo   venv\Scripts\activate
echo   python main.py
echo.
pause
