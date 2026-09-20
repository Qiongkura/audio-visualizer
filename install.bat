@echo off
rem ---------------------------------------------------------------
rem  Install dependencies into a local venv (run once).
rem  All messages are plain ASCII on purpose: Chinese text in .bat
rem  files gets garbled under some Windows code pages (GBK/437).
rem ---------------------------------------------------------------
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo [1/3] Creating virtual environment in .\venv ...
python -m venv venv
if errorlevel 1 goto fail

echo [2/3] Installing dependencies ...
if exist requirements.lock (
    venv\Scripts\python.exe -m pip install -r requirements.lock
) else (
    venv\Scripts\python.exe -m pip install -r requirements.txt
)
if errorlevel 1 goto fail

echo [3/3] Done. Run start.bat to launch the visualizer.
pause
exit /b 0

:fail
echo.
echo Installation failed. Check your network connection and Python install,
echo then run this script again.
pause
exit /b 1
