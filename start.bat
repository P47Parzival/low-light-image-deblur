@echo off
echo ========================================
echo GARUD - Starting Full Application
echo ========================================
echo.
echo This will start both Backend and Frontend servers
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Press Ctrl+C in server windows to stop
echo ========================================
echo.

cd /d "%~dp0"

:: Check if backend venv exists
if not exist "full model\src\api\venv" (
    echo ERROR: Backend not set up!
    echo Please run setup_backend.bat first
    pause
    exit /b 1
)

:: Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo ERROR: Frontend not set up!
    echo Please run setup_frontend.bat first
    pause
    exit /b 1
)

:: Start backend in new window with proper error handling
echo Starting Backend Server...
start "GARUD Backend" cmd /k "cd /d "%~dp0" && cd "full model\src\api" && call venv\Scripts\activate.bat && echo [Backend] Starting FastAPI server... && python main.py || echo [ERROR] Backend failed to start! && pause"

:: Wait a bit for backend to start
timeout /t 5 /nobreak >nul

:: Start frontend in new window
echo Starting Frontend Server...
start "GARUD Frontend" cmd /k "cd /d "%~dp0frontend" && echo [Frontend] Starting Vite dev server... && npm run dev"

:: Wait a bit for frontend to start
timeout /t 5 /nobreak >nul

:: Open browser
echo Opening browser...
start http://localhost:5173

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Check the server windows for logs
echo Close the server windows to stop the application
echo.
pause
