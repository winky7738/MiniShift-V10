@echo off
chcp 65001 >nul
title MiniShiftAD V10 AI Handoff 12-Category Training

set "MINISHIFT_PROJECT_WIN=%~dp0"
set "MINISHIFT_PROJECT_WIN=%MINISHIFT_PROJECT_WIN:~0,-1%"
for /f "usebackq delims=" %%I in (`wsl.exe -d Ubuntu-22.04 -- wslpath -a "%MINISHIFT_PROJECT_WIN%"`) do set "MINISHIFT_PROJECT_WSL=%%I"
if not defined MINISHIFT_PROJECT_WSL (
    echo Failed to convert the project folder to a WSL path.
    pause
    exit /b 1
)

echo Starting the V10 12-category handoff job in WSL...
echo Project: %MINISHIFT_PROJECT_WSL%
echo Console output is also saved to logs\MiniShift_V10_AI_handoff_12cat_batch_console.log
wsl.exe -d Ubuntu-22.04 -- bash -lc "set -o pipefail; source /home/xu/simple3d-env.sh && cd '%MINISHIFT_PROJECT_WSL%' && mkdir -p logs && python -u train_single_category.py 2>&1 | tee -a ./logs/MiniShift_V10_AI_handoff_12cat_batch_console.log"

if errorlevel 1 (
    echo.
    echo Training stopped with an error. Review the message above.
) else (
    echo.
    echo Training completed successfully.
)

pause
