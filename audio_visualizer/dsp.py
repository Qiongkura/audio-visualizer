# -*- coding: utf-8 -*-
"""纯 DSP 层：环形缓冲区、频谱分析、波形分析、DSP 工作线程。

这一层不依赖 Tk，也不依赖音频设备，可以被单元测试完整覆盖。
"""
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import (DB_FLOOR, DSP_INTERVAL, FFT_MAX, FFT_MIN, FFT_WINDOW_SEC,
                     MAX_F, MIN_F, RING_SIZE, WAVE_ATTACK, WAVE_MAX_GAIN,
                     WAVE_MIN_PEAK, WAVE_MIN_SAMPLES, WAVE_TARGET,
                     WAVE_WINDOW_SEC)

log = logging.getLogger(__name__)


# ============================================================ 环形缓冲区
class RingBuffer:
    """定长环形缓冲区。

    与常见实现的关键区别：额外维护 ``valid``（已写入的有效采样数）。
    启动瞬间、切换设备、设备重连之后，读到的只会是真正写入过的数据，
    不会把初始化零值或上一次设备的残留音频当成有效信号。
    """

    def __init__(self, size=RING_SIZE):
        size = int(size)
        if size <= 0:
            raise ValueError("环形缓冲区容量必须为正数")
        self.size = size
        self._buf = np.zeros(size, dtype=np.float32)
        self._wpos = 0
        self._valid = 0

    @property
    def valid(self):
        """当前可用于分析的有效采样数（<= size）"""
        return self._valid

    @property
    def wpos(self):
        return self._wpos

    def clear(self):
        """清空：切换设备 / 重连前必须调用"""
        self._buf.fill(0)
        self._wpos = 0
        self._valid = 0

    def write(self, mono):
        """写入一段单声道数据（多余部分丢弃最旧的）"""
        x = np.asarray(mono, dtype=np.float32).ravel()
        n = x.size
        if n == 0:
            return
        if n >= self.size:
            self._buf[:] = x[-self.size:]
            self._wpos = 0
            self._valid = self.size
            return
        first = min(n, self.size - self._wpos)
        self._buf[self._wpos:self._wpos + first] = x[:first]
        if n > first:
            self._buf[:n - first] = x[first:]
        self._wpos = (self._wpos + n) % self.size
        self._valid = min(self._valid + n, self.size)

    def read_last(self, n):
        """按时间先后返回最近 n 个有效采样（返回副本，长度 <= valid）"""
        n = int(min(n, self._valid))
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        if n >= self.size:                      # 整个缓冲区都是有效数据
            start = self._wpos
            return np.concatenate((self._buf[start:], self._buf[:start]))
        start = (self._wpos - n) % self.size
        if start + n <= self.size:
            return self._buf[start:start + n].copy()
        head = self.size - start
        return np.concatenate((self._buf[start:], self._buf[:n - head]))


def stereo_to_mono(frames):
    """多声道 -> 单声道，逐采样取绝对值更大的声道。

    不用简单平均：左右声道反相时平均会互相抵消（有声音却像静音）。
    逐采样取较大声道既避免抵消，又保留符号，波形图不会被整流。
    """
    if frames.ndim != 2:
        return np.asarray(frames, dtype=np.float32).ravel()
    ch = frames.shape[1]
    if ch <= 1:
        return np.ascontiguousarray(frames[:, 0])
    k = np.abs(frames).argmax(axis=1)[:, None]
    return np.take_along_axis(frames, k, axis=1)[:, 0]


# ============================================================ 频谱
def fft_size_for(sample_rate):
    """按采样率自适应 FFT 点数，保持约 43ms 时窗（48k->2048, 192k->8192）"""
    sr = max(8000, int(sample_rate or 48000))
    target = max(FFT_MIN, int(FFT_WINDOW_SEC * sr))
    n = 1 << int(round(np.log2(target)))
    return int(min(max(n, FFT_MIN), FFT_MAX))


def wave_size_for(sample_rate):
    """波形分析窗采样数"""
    sr = max(8000, int(sample_rate or 48000))
    return int(max(WAVE_MIN_SAMPLES, WAVE_WINDOW_SEC * sr))


def spectrum_edges(n_bars, sample_rate, n_fft, min_f=MIN_F, max_f=MAX_F):
    """返回 (edges, lo, hi)：对数分频边界，以及每根柱子对应的 FFT bin 区间 [lo, hi)。

    这里刻意不用 ``idx[0] = 0`` / ``idx[-1] = len(freqs)``：
    那会让第一根柱子混进 DC 和 30Hz 以下的内容、最后一根柱子一路吃到 Nyquist，
    实际显示范围远超声明的 30Hz~16kHz。
    """
    freqs = np.fft.rfftfreq(int(n_fft), 1.0 / float(sample_rate))
    nyq = float(sample_rate) / 2.0
    top_f = min(float(max_f), nyq - 1.0)
    top_f = max(top_f, float(min_f) * 1.01)      # 极低采样率兜底
    edges = np.geomspace(float(min_f), top_f, int(n_bars) + 1)
    lo = np.searchsorted(freqs, edges[:-1], side="left")
    hi = np.searchsorted(freqs, edges[1:], side="left")
    hi[-1] = np.searchsorted(freqs, top_f, side="right")   # 上限严格截断
    hi = np.maximum(hi, lo + 1)                  # 每根柱子至少 1 个 bin
    np.clip(lo, 0, freqs.size - 1, out=lo)
    np.clip(hi, 1, freqs.size, out=hi)
    return edges, lo, hi


class SpectrumAnalyzer:
    """把时域数据算成对数分布的频谱柱（dB）。

    窗函数与 bin 映射按 (n_fft, sr, n_bars) 缓存，参数不变时不重复计算。
    有效数据不足一个完整 FFT 窗时返回 None —— 宁可这一帧不画，
    也不要拿补零数据算出一帧假频谱。
    """

    def __init__(self, n_bars=64, min_f=MIN_F, max_f=MAX_F, db_floor=DB_FLOOR):
        self.min_f = float(min_f)
        self.max_f = float(max_f)
        self.db_floor = float(db_floor)
        self.n_bars = int(n_bars)
        self._key = None
        self._win = None
        self._lo = None
        self._hi = None
        self._edges = None

    @property
    def top_f(self):
        return float(self._edges[-1]) if self._edges is not None else self.max_f

    @property
    def edges(self):
        """对数分频边界（长度 n_bars+1）"""
        return self._edges

    @property
    def bands(self):
        """每根柱子对应的 FFT bin 区间 [lo, hi)"""
        return self._lo, self._hi

    def _prepare(self, n_fft, sample_rate, n_bars):
        key = (int(n_fft), int(sample_rate), int(n_bars))
        if key == self._key:
            return
        edges, lo, hi = spectrum_edges(n_bars, sample_rate, n_fft,
                                       self.min_f, self.max_f)
        self._edges, self._lo, self._hi = edges, lo, hi
        self._win = np.hanning(int(n_fft))
        self._key = key

    def compute(self, x, sample_rate, n_bars=None):
        """返回长度 n_bars 的 dB 数组；有效数据不足时返回 None"""
        n_bars = int(self.n_bars if n_bars is None else n_bars)
        if n_bars <= 0:
            return None
        sr = max(8000, int(sample_rate or 48000))
        n_fft = fft_size_for(sr)
        self._prepare(n_fft, sr, n_bars)

        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size < n_fft:
            return None
        x = x[-n_fft:]
        spec = np.abs(np.fft.rfft(x * self._win))
        db = 20.0 * np.log10(spec / (n_fft / 4.0) + 1e-10)
        np.clip(db, self.db_floor, 0.0, out=db)

        vals = np.full(n_bars, self.db_floor)
        for i in range(n_bars):
            a, b = int(self._lo[i]), int(self._hi[i])
            if b > a:
                vals[i] = db[a:b].max()
        return vals


# ============================================================ 波形
class WaveformAnalyzer:
    """波形降采样 + 自动增益。

    增益按“目标峰值 / 当前峰值”计算，下降瞬时（防削顶）、上升平滑（不抖）。
    """

    def __init__(self, target=WAVE_TARGET, max_gain=WAVE_MAX_GAIN,
                 attack=WAVE_ATTACK, min_peak=WAVE_MIN_PEAK, gain=20.0):
        self.target = float(target)
        self.max_gain = float(max_gain)
        self.attack = float(attack)
        self.min_peak = float(min_peak)
        self.gain = float(gain)

    def reset(self):
        self.gain = 20.0

    def update_gain(self, x):
        peak = float(np.abs(x).max()) if x.size else 0.0
        if peak > self.min_peak:
            target = min(self.target / peak, self.max_gain)
            if target < self.gain:
                self.gain = target
            else:
                self.gain += (target - self.gain) * self.attack
        return self.gain

    @staticmethod
    def _reduce(x, cols):
        """把 x 压成 cols 个点：样本多时按桶取均值（相当于低通），少时线性插值"""
        if cols <= 0:
            return np.empty(0, dtype=np.float64)
        if x.size == 0:
            return np.zeros(cols, dtype=np.float64)
        if x.size < cols:
            src = np.arange(x.size, dtype=np.float64)
            return np.interp(np.linspace(0, x.size - 1, cols), src, x)
        m = (x.size // cols) * cols
        return x[:m].reshape(cols, -1).mean(axis=1)

    def process(self, x, cols):
        """返回 (ys, gain)：ys 为归一化波形曲线，gain 为当前自动增益"""
        x = np.asarray(x, dtype=np.float32).ravel()
        gain = self.update_gain(x)
        if x.size == 0:
            return np.zeros(max(1, int(cols)), dtype=np.float64), gain
        ys = self._reduce(x, int(cols))
        if ys.size >= 3:
            ys = np.convolve(ys, np.ones(3) / 3.0, mode="same")
        return ys, gain


# ============================================================ DSP 工作线程
@dataclass
class DspFrame:
    """一帧 DSP 结果，只在 Tk 线程里被读取用于绘制"""
    seq: int
    sample_rate: int
    valid: int
    bars: object = None      # np.ndarray | None
    wave: object = None      # np.ndarray | None
    gain: float = 1.0


class DspWorker(threading.Thread):
    """在独立线程里做 FFT 与波形降采样。

    只保留“最新一帧”，UI 线程来不及消费时旧帧直接丢弃，
    不会像普通队列那样越积越多。
    """

    def __init__(self, read_samples, sample_rate, wave_cols, n_bars=64,
                 interval=DSP_INTERVAL):
        super().__init__(name="dsp-worker", daemon=True)
        self.spectrum = SpectrumAnalyzer(n_bars)
        self.waveform = WaveformAnalyzer()
        self._read = read_samples
        self._sample_rate = sample_rate
        self._wave_cols = wave_cols
        self._interval = float(interval)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._errors = 0

    # ---- 外部可调参数（int 赋值在 GIL 下是原子的）----
    @property
    def n_bars(self):
        return self.spectrum.n_bars

    @n_bars.setter
    def n_bars(self, v):
        self.spectrum.n_bars = int(v)

    def reset(self):
        """设备切换 / 重连后调用：丢掉旧帧并重置波形增益"""
        with self._lock:
            self._latest = None
        self.waveform.reset()

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self, timeout=1.5):
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)

    def compute_once(self):
        """同步算一帧（供测试使用），返回 DspFrame"""
        sr = max(8000, int(self._sample_rate() or 48000))
        n_bars = int(self.spectrum.n_bars)
        n_fft = fft_size_for(sr)
        n_wave = wave_size_for(sr)
        x = np.asarray(self._read(max(n_fft, n_wave)), dtype=np.float32)
        valid = int(x.size)

        bars = self.spectrum.compute(x, sr, n_bars) if valid >= n_fft else None
        wave = None
        if valid >= WAVE_MIN_SAMPLES:
            seg = x[-n_wave:] if valid > n_wave else x
            wave, _ = self.waveform.process(seg, max(2, int(self._wave_cols())))
        return DspFrame(seq=self._seq, sample_rate=sr, valid=valid,
                        bars=bars, wave=wave, gain=self.waveform.gain)

    def run(self):
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                self._seq += 1
                frame = self.compute_once()
                with self._lock:
                    self._latest = frame
            except Exception:                       # 后台线程绝不能因为异常退出
                self._errors += 1
                if self._errors <= 3:
                    log.exception("DSP 线程计算失败")
            dt = self._interval - (time.perf_counter() - t0)
            if dt > 0:
                self._stop_evt.wait(dt)
