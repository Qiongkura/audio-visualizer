@echo off
cd /d %~dp0
echo [1/2] 创建虚拟环境...
python -m venv venv
echo [2/2] 安装依赖 (pyaudiowpatch, numpy, customtkinter)...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo 安装失败，请检查网络后重试。
)
pause
