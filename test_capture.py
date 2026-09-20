# -*- coding: utf-8 -*-
"""环回采集自检脚本（需要真实声卡，所以不进单元测试，手动运行）：

1. 列出所有可采集的输出设备
2. 用系统默认输出播放 440Hz 测试音
3. 通过 WASAPI 环回采集 3 秒
4. 校验采集到的音量、主频，以及频谱柱映射是否落在正确的频率区间

运行: venv\\Scripts\\python.exe test_capture.py
"""
import os
import sys
import tempfile
import time
import wave

import numpy as np

from audio_visualizer.audio_engine import AudioEngine, EngineError
from audio_visualizer.config import RING_SIZE
from audio_visualizer.dsp import SpectrumAnalyzer, fft_size_for
from audio_visualizer.renderer import freq_axis_top

DUR = 3.0
FREQ = 440.0
SR = 48000
N_BARS = 64


def make_tone_wav(path, freq=FREQ, sr=SR, seconds=1.0):
    """生成一段带淡入淡出的测试音 wav"""
    t = np.arange(int(sr * seconds)) / sr
    x = 0.35 * np.sin(2 * np.pi * freq * t)
    n = len(x)
    env = np.ones(n)
    env[:2000] = np.linspace(0, 1, 2000)
    env[-2000:] = np.linspace(1, 0, 2000)
    x = (x * env * 32767).astype(np.int16)
    stereo = np.column_stack([x, x])
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())
    return path


def check(engine):
    """采集并校验，返回 (ok, 报告行列表)"""
    lines = []
    sr = engine.sample_rate or SR
    n_fft = fft_size_for(sr)
    # 取整个环形缓冲区：停播后回调仍在写静音，只取最后一小段会读到静音尾巴
    x = engine.read_last(RING_SIZE)
    if x.size < n_fft:
        return False, ["FAIL: 只采到 %d 个采样，不够做一次 FFT" % x.size]

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.abs(x).max())
    spec = np.abs(np.fft.rfft(x[-n_fft:] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    dom = float(freqs[int(np.argmax(spec))])
    lines.append("采集 %d 采样 (%.2fs @ %dHz, %dch)"
                 % (x.size, x.size / sr, sr, engine.channels))
    lines.append("RMS=%.4f  Peak=%.4f  主频=%.1fHz (期望 ~%.0fHz)"
                 % (rms, peak, dom, FREQ))

    an = SpectrumAnalyzer(N_BARS)
    bars = an.compute(x, sr, N_BARS)
    if bars is None:
        return False, lines + ["FAIL: 频谱分析返回空"]
    k = int(np.argmax(bars))
    edges = an.edges
    lo, hi = an.bands
    lines.append("最强柱 #%d 覆盖 %.0f~%.0fHz，峰值 %.1fdB（轴上限 %.0fHz）"
                 % (k, edges[k], edges[k + 1], bars[k], freq_axis_top(sr)))
    lines.append("第一根柱子起始 bin=%d (%.0fHz) / 最后一根结束 bin=%d (%.0fHz)"
                 % (lo[0], freqs[lo[0]], hi[-1], freqs[min(hi[-1], freqs.size - 1)]))

    ok = rms > 0.01 and abs(dom - FREQ) < 20
    ok = ok and edges[k] <= FREQ * 1.5 and edges[k + 1] >= FREQ * 0.6
    if not ok:
        lines.append("FAIL: 采集到的信号异常（静音、频率不符或频谱柱映射不对）")
    return ok, lines


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        engine = AudioEngine()
    except EngineError as e:
        print("FAIL: %s" % e)
        return 1

    tone_path = os.path.join(tempfile.gettempdir(), "av_test_tone.wav")
    playing = False
    try:
        try:
            devices = engine.list_loopbacks()
        except EngineError as e:
            print("FAIL: %s" % e)
            return 1
        print("发现 %d 个可采集的输出设备:" % len(devices))
        for d in devices:
            print("  [%d] %s  @%dHz  %dch" % (d.index, d.name, d.rate, d.channels))
        if not devices:
            print("FAIL: 没有任何音频输出设备，无法环回采集")
            return 1

        if not engine.start(None):
            print("FAIL: 打开环回采集失败: %s" % engine.error)
            return 1
        print("目标设备: %s" % engine.device.name)

        import winsound
        make_tone_wav(tone_path)
        winsound.PlaySound(tone_path, winsound.SND_ASYNC | winsound.SND_LOOP)
        playing = True
        time.sleep(DUR)
        winsound.PlaySound(None, winsound.SND_PURGE)
        playing = False

        ok, lines = check(engine)
        for line in lines:
            print(line)
        print("PASS" if ok else "FAIL")
        return 0 if ok else 2
    finally:
        if playing:
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        engine.close()
        try:
            os.remove(tone_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
