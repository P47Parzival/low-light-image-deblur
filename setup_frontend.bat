@echo off
echo ========================================
echo GARUD Frontend Setup Script
echo ========================================
echo.

cd /d "%~dp0frontend"

echo [1/2] Installing Node.js dependencies...
echo This may take a few minutes...
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install npm dependencies
    echo Make sure Node.js 16+ is installed
    pause
    exit /b 1
)
echo.

echo [2/2] Testing build configuration...
call npm run build
if errorlevel 1 (
    echo WARNING: Build test failed, but dependencies are installed
    echo You can still run the dev server
)
echo.

echo ========================================
echo Frontend setup completed successfully!
echo ========================================
echo.
echo To start the frontend dev server, run:
echo   cd frontend
echo   npm run dev
echo.
pause
