@echo off
echo ========================================
echo GARUD Frontend - Manual Start
echo ========================================
echo.
echo This script will start ONLY the frontend server
echo.

cd /d "%~dp0frontend"

echo Starting Vite dev server...
echo Frontend will be available at: http://localhost:5173
echo.
echo Press Ctrl+C to stop the server
echo.

npm run dev

pause
