@echo off
rem ---------------------------------------------------------------
rem  Build a standalone dist\AudioVisualizer.exe (PyInstaller).
rem  Run install.bat first.
rem ---------------------------------------------------------------
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Dependencies are not installed yet. Please run install.bat first.
    pause
    exit /b 1
)

echo [1/2] Installing build dependencies ...
venv\Scripts\python.exe -m pip install -r requirements-build.txt
if errorlevel 1 goto fail

echo [2/2] Building dist\AudioVisualizer.exe ...
venv\Scripts\python.exe -m PyInstaller --noconfirm --clean audio_visualizer.spec
if errorlevel 1 goto fail

echo.
echo Done. The executable is at dist\AudioVisualizer.exe
pause
exit /b 0

:fail
echo.
echo Build failed. See the output above for details.
pause
exit /b 1
