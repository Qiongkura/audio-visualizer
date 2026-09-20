# -*- coding: utf-8 -*-
"""全局配置：采集 / DSP / 渲染 / 配色 常量，以及配色工具函数。

这里只放常量和纯函数，不依赖 Tk、不依赖音频库，便于被测试直接 import。
"""
import numpy as np

APP_TITLE = "Audio Visualizer · 音频可视化"

# ---------------- 采集 ----------------
RING_SIZE = 1 << 15           # 环形缓冲区容量（采样数）32768
CALLBACK_FRAMES = 512         # PortAudio 每次回调的帧数
RETRY_DELAY = 1.5             # 采集失败后的重连间隔（秒）
WATCHDOG_MS = 1500            # 看门狗轮询周期（毫秒）
ENUM_BACKOFF = 3.0            # 设备枚举失败后的退避（秒）
STATUS_LOG_THROTTLE = 5.0     # PortAudio status 日志节流（秒）

# ---------------- DSP ----------------
MIN_F, MAX_F = 30.0, 16000.0  # 频谱显示频率范围（Hz）
DB_FLOOR = -85.0              # 频谱底噪（dB）
FFT_WINDOW_SEC = 0.043        # 目标分析时窗（秒）
FFT_MIN, FFT_MAX = 1024, 16384
WAVE_WINDOW_SEC = 0.09        # 波形时窗（秒）
WAVE_MIN_SAMPLES = 256        # 少于这个采样数不画波形
WAVE_TARGET = 0.42            # 自动增益目标峰值（相对半高，0.5 为满幅）
WAVE_MAX_GAIN = 1500.0
WAVE_ATTACK = 0.05            # 增益上升平滑系数（下降是瞬时的）
WAVE_MIN_PEAK = 1e-5

# ---------------- 显示 ----------------
TICK_MS = 16                  # 主循环节拍
DSP_INTERVAL = 1.0 / 60.0     # DSP 线程计算节拍
FPS_INTERVAL = 0.5            # FPS 统计窗口（秒）
RESIZE_DEBOUNCE_MS = 160      # 窗口缩放防抖（毫秒）
BAR_RELEASE = 1.5             # 频谱柱每帧下落 dB
PEAK_FALL = 0.55              # 峰值帽每帧下落 dB
ERROR_LOG_THROTTLE = 1.0      # 主循环异常落盘节流（秒）

# ---------------- 配色（Instagram 风格） ----------------
BG = "#F4F4F6"
CARD = "#FFFFFF"
TXT = "#141419"
SUB = "#9A9AA3"
FAINT = "#C9C9D1"
ACCENT = "#E1306C"
ACCENT_RGB = (225, 48, 108)
LINE_RGB = (241, 241, 245)
WASH_TOP = np.array([255, 255, 255], dtype=np.uint8)
WASH_BOT = np.array([252, 246, 242], dtype=np.uint8)
GRAD_STOPS = ("#833AB4", "#C13584", "#E1306C", "#FD1D1D", "#F77737", "#FCAF45")

# ---------------- 排版 ----------------
FONT_FAMILY = "Microsoft YaHei UI"
SPEC_L, SPEC_R = 46, 16       # 频谱卡左右留白
SPEC_TOP, SPEC_BASE = 32, 26  # 频谱卡上/下留白
WAVE_L, WAVE_R = 14, 14
BAR_GAP = 3.0
SPEC_BARS_MIN, SPEC_BARS_MAX = 24, 90
SPEC_BAR_PITCH = 11           # 每根柱子占用的水平像素（含间隔），用于推算柱子数量


def hex2rgb(s):
    return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))


def rgb2hex(c):
    return "#%02x%02x%02x" % tuple(c)


def lerp3(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def gradient_lut(stops, n):
    """把一组色标插值成 n 级渐变查找表"""
    cols = [hex2rgb(s) for s in stops]
    lut = []
    for i in range(n):
        t = i / (n - 1) * (len(cols) - 1)
        k = min(int(t), len(cols) - 2)
        lut.append(lerp3(cols[k], cols[k + 1], t - k))
    return lut


BAR_RGB = gradient_lut(GRAD_STOPS, 96)                        # 频谱柱：按频率位置渐变
CAP_RGB = [tuple(int(v * 0.72) for v in c) for c in BAR_RGB]  # 峰值帽：同色系加深
