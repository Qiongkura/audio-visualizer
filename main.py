# -*- coding: utf-8 -*-
"""音频可视化程序入口（薄壳）。

自动识别电脑中正在播放的声音（WASAPI 环回采集，无需虚拟声卡），
实时显示频谱（对数刻度、Instagram 渐变配色）与波形（自动增益包络）。

实际实现按职责拆在 audio_visualizer/ 包里：
    config.py        常量与配色
    audio_engine.py  采集 + 设备异常恢复状态机
    dsp.py           环形缓冲区 / FFT / 波形 / DSP 工作线程
    renderer.py      numpy 帧缓冲 + Tk 上屏
    app.py           Tk 界面与主循环

依赖: pip install -r requirements.txt
运行: python main.py
"""
import sys

from audio_visualizer.app import main

if __name__ == "__main__":
    sys.exit(main())
