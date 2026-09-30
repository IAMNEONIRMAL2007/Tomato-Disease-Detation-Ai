# Created By NirmalBorole
@echo off
title AgriVision AI - Tomato Disease Detection Local Deployment
color 0A

echo ======================================================================
echo    TOMATO LEAF DISEASE AI DIAGNOSIS - LOCAL DEPLOYMENT
echo ======================================================================
echo.
echo [*] Initializing local server on http://localhost:8000 ...
echo [*] Launching interactive web dashboard in your browser...
echo.

:: Open default browser after a 2 second delay in background
start "" timeout /t 2 /nobreak >nul & start http://localhost:8000

:: Start the Python server (prefer venv with TensorFlow if available)
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0server.py"
) else (
    python "%~dp0server.py"
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [*] Falling back to WSL2 Python server...
    wsl -e bash -c "cd '/mnt/c/Users/nirma/OneDrive/Desktop/tomato disease detection' && source /home/nirma/tomato_env/bin/activate && python3 server.py"
)

pause
