@echo off
rem ---------------------------------------------------------------
rem  Launch the visualizer (no console window).
rem  Run install.bat first if .\venv is missing.
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\pythonw.exe (
    echo Dependencies are not installed yet. Please run install.bat first.
    pause
    exit /b 1
)

start "" venv\Scripts\pythonw.exe main.py
exit /b 0
