@echo off
cd /d %~dp0
if not exist venv\Scripts\pythonw.exe (
    echo 尚未安装依赖，请先运行 install.bat
    pause
    exit /b 1
)
start "" venv\Scripts\pythonw.exe main.py
