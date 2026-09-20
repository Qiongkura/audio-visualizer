# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（单文件 exe）。

用法: venv\\Scripts\\python.exe -m PyInstaller --noconfirm --clean audio_visualizer.spec
产物: dist\\AudioVisualizer.exe

说明:
- 入口是根目录的 main.py，包 audio_visualizer 会被自动收进来；
- customtkinter 的主题资源是数据文件，必须显式收集，否则界面会缺资源；
- 无控制台窗口，运行期异常写入 exe 同目录的 error.log。
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("customtkinter")
hiddenimports = collect_submodules("customtkinter")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # 测试和打包工具本身不需要进 exe
    excludes=["tests", "unittest", "pyinstaller", "PyInstaller",
              "numpy.f2py", "numpy.testing", "pydoc_data"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AudioVisualizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
