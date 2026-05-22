@echo off
title hand_capture calibration
cd /d "C:\Users\henry\Desktop\hand_capture"
echo Two-phase calibration: --side right
echo.
python -m calibration.collect --side %1
echo.
echo Solving most recent calibration session...
for /f "delims=" %%d in ('dir /b /od "calibration\raw"') do set "LATEST=%%d"
python -m calibration.solve "calibration\raw\%LATEST%"
pause
