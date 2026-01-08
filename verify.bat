@echo off
echo ========================================
echo GARUD - System Verification Script
echo ========================================
echo.
echo This script will verify both frontend and backend are working correctly
echo.

cd /d "%~dp0"

:: Check if backend is set up
if not exist "full model\src\api\venv" (
    echo [ERROR] Backend virtual environment not found!
    echo Please run setup_backend.bat first
    pause
    exit /b 1
)

:: Check if frontend is set up
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend node_modules not found!
    echo Please run setup_frontend.bat first
    pause
    exit /b 1
)

echo [1/4] Testing Frontend Build...
echo ========================================
cd frontend
call npm run build
if errorlevel 1 (
    echo.
    echo [ERROR] Frontend build failed!
    echo Check the errors above
    cd ..
    pause
    exit /b 1
)
echo [SUCCESS] Frontend builds successfully!
echo.

cd ..

echo [2/4] Testing Backend Imports...
echo ========================================
cd "full model\src\api"
call venv\Scripts\activate
python -c "import fastapi, uvicorn, torch, cv2, ultralytics; print('[SUCCESS] All backend imports working!')"
if errorlevel 1 (
    echo.
    echo [ERROR] Backend imports failed!
    echo Some dependencies may not be installed correctly
    cd ..\..\..
    pause
    exit /b 1
)
echo.

echo [3/4] Testing Database Initialization...
echo ========================================
python -c "import sys; sys.path.append('../core'); import database; database.init_db(); print('[SUCCESS] Database initialized!')"
if errorlevel 1 (
    echo.
    echo [ERROR] Database initialization failed!
    cd ..\..\..
    pause
    exit /b 1
)
echo.

cd ..\..\..

echo [4/4] Checking YOLOv8 Model...
echo ========================================
if exist "full model\yolov8n.pt" (
    echo [SUCCESS] YOLOv8 base model found!
) else (
    echo [WARNING] YOLOv8 base model not found
    echo The system will attempt to download it on first run
)
echo.

echo ========================================
echo VERIFICATION COMPLETE!
echo ========================================
echo.
echo ✓ Frontend builds successfully
echo ✓ Backend dependencies installed
echo ✓ Database initialized
echo ✓ System ready to start
echo.
echo Next steps:
echo 1. Run start.bat to launch the application
echo 2. Backend will be on http://localhost:8000
echo 3. Frontend will be on http://localhost:5173
echo.
pause
