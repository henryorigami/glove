@echo off
title hand_capture dashboard
cd /d "C:\Users\henry\Desktop\hand_capture"

REM Kill any leftover dashboard / orchestrator processes from a previous run.
echo Cleaning up any stale processes...
for /f "tokens=5" %%P in ('netstat -aon ^| findstr ":8080.*LISTENING"') do (
    echo   stale PID %%P on port 8080
    taskkill /PID %%P /F >nul 2>&1
)
taskkill /F /IM rerun.exe >nul 2>&1
taskkill /F /IM manus_logger.exe >nul 2>&1

REM Give the OS a moment to release the socket.
timeout /t 1 /nobreak >nul

echo Starting dashboard at http://127.0.0.1:8080/
start "" "http://127.0.0.1:8080/"
python -u -m dashboard.server

echo.
echo --- dashboard.server exited ---
pause
