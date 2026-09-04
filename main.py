# -*- coding: utf-8 -*-
"""
音频可视化程序
自动识别电脑中正在播放的声音（WASAPI 环回采集，无需虚拟声卡），
实时显示频谱（对数刻度柱状图 + 峰值保持）与波形（包络自动增益）。

依赖: pip install pyaudiowpatch numpy
运行: python main.py
"""
import ctypes
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
import tkinter as tk
from tkinter import ttk, messagebox

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

RING_SIZE = 1 << 15          # 环形缓冲区 32768 采样
MIN_F, MAX_F = 30.0, 16000.0 # 频谱显示频率范围
DB_FLOOR = -85.0             # 频谱底噪（dB）
BAR_RELEASE = 1.5            # 频谱柱每帧下落 dB
PEAK_FALL = 0.55             # 峰值帽每帧下落 dB

BG = "#0b0f14"
PANEL = "#0f1520"
FG = "#e5e7eb"
DIM = "#94a3b8"
ACCENT = "#38bdf8"
GRID = "#1e293b"
FONT = ("Microsoft YaHei UI", 9)
FONT_S = ("Microsoft YaHei UI", 8)


def lerp3(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def build_color_lut(n=128):
    """频谱柱颜色: 绿 -> 黄 -> 红（按响度）"""
    green, yellow, red = (34, 197, 94), (250, 204, 21), (239, 68, 68)
    lut = []
    for i in range(n):
        t = i / (n - 1)
        if t < 0.55:
            c = lerp3(green, yellow, t / 0.55)
        else:
            c = lerp3(yellow, red, (t - 0.55) / 0.45)
        lut.append("#%02x%02x%02x" % c)
    return lut


COLOR_LUT = build_color_lut()


class AudioEngine:
    """WASAPI 环回采集：把系统正在输出的声音写入环形缓冲区"""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.device = None
        self.channels = 0
        self.sample_rate = 0
        self.ring = np.zeros(RING_SIZE, dtype=np.float32)
        self.wpos = 0
        self.lock = threading.Lock()
        self.alive = False
        self.error = ""

    # ---------- 设备 ----------
    def list_loopbacks(self):
        """所有可采集的输出设备（环回形式）"""
        out = []
        for d in self.pa.get_loopback_device_info_generator():
            out.append({
                "index": int(d["index"]),
                "name": d["name"],
                "rate": int(d["defaultSampleRate"]),
                "channels": int(d["maxInputChannels"]),
            })
        return out

    def default_loopback(self):
        """当前系统默认输出设备对应的环回设备"""
        try:
            wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            spk = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        except Exception:
            return None
        if spk.get("isLoopbackDevice"):
            return spk
        for lb in self.pa.get_loopback_device_info_generator():
            if spk["name"] in lb["name"]:
                return lb
        return None

    # ---------- 采集 ----------
    def start(self, device=None):
        """device=None 表示自动跟随系统默认输出"""
        self.stop()
        try:
            if device is None:
                device = self.default_loopback()
            if device is None:
                self.error = "未找到可用的音频输出设备"
                return False
            self.device = device
            self.channels = max(1, int(device.get("channels",
                                                   device.get("maxInputChannels", 2))))
            self.sample_rate = int(device.get("rate",
                                              device.get("defaultSampleRate", 48000)))
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=512,
                input=True,
                input_device_index=device["index"],
                stream_callback=self._callback,
            )
            self.alive = True
            self.error = ""
            return True
        except Exception as e:
            self.error = str(e)
            self.alive = False
            return False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.alive = False

    def close(self):
        self.stop()
        try:
            self.pa.terminate()
        except Exception:
            pass

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            data = np.frombuffer(in_data, dtype=np.float32)
            if self.channels > 1 and data.size % self.channels == 0:
                mono = data.reshape(-1, self.channels).mean(axis=1)
            else:
                mono = data
            n = min(mono.size, RING_SIZE)
            if n < mono.size:
                mono = mono[-n:]
            with self.lock:
                first = min(n, RING_SIZE - self.wpos)
                self.ring[self.wpos:self.wpos + first] = mono[:first]
                if n > first:
                    self.ring[:n - first] = mono[first:n]
                self.wpos = (self.wpos + n) % RING_SIZE
            return (None, pyaudio.paContinue)
        except Exception as e:
            self.error = str(e)
            self.alive = False
            return (None, pyaudio.paAbort)

    def read_last(self, n):
        n = min(n, RING_SIZE)
        with self.lock:
            if n == RING_SIZE:
                return self.ring.copy()
            start = (self.wpos - n) % RING_SIZE
            if start <= self.wpos:
                out = self.ring[start:self.wpos].copy()
            else:
                out = np.concatenate((self.ring[start:], self.ring[:self.wpos]))
        return out


class Visualizer:
    def __init__(self, root):
        self.root = root
        root.title("音频可视化 - 系统声音频谱/波形")
        root.configure(bg=BG)
        root.geometry("1080x640")
        root.minsize(720, 460)

        self.engine = AudioEngine()
        self.paused = False
        self.retry_at = 0.0
        self.n_bars = 64
        self.bars = np.full(1, DB_FLOOR, dtype=np.float64)
        self.peaks = np.full(1, DB_FLOOR, dtype=np.float64)
        self.wave_gain = 20.0
        self.follow_default = tk.BooleanVar(value=True)
        self.device_map = {}
        self.frames = 0
        self.fps_t0 = time.time()
        self.fps = 0.0

        self._build_ui()
        self._populate_devices()
        self._start_engine(None)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(16, self._tick)          # ~60 FPS
        root.after(1500, self._watchdog)    # 设备监控/自动重连

    # ================= UI =================
    def _build_ui(self):
        top = tk.Frame(self.root, bg=PANEL)
        top.pack(fill="x")

        tk.Label(top, text="采集设备:", bg=PANEL, fg=DIM, font=FONT).pack(side="left", padx=(10, 4), pady=8)
        self.combo = ttk.Combobox(top, width=46, state="readonly", font=FONT)
        self.combo.pack(side="left", padx=2)
        self.combo.bind("<<ComboboxSelected>>", self._on_pick_device)

        ttk.Button(top, text="刷新", width=6, command=self._populate_devices).pack(side="left", padx=4)

        cb = tk.Checkbutton(top, text="跟随默认输出设备", variable=self.follow_default,
                            command=self._on_follow_toggle, bg=PANEL, fg=FG,
                            activebackground=PANEL, activeforeground=FG,
                            selectcolor="#1e293b", font=FONT)
        cb.pack(side="left", padx=10)

        self.pause_btn = tk.Button(top, text="暂停", width=8, command=self._toggle_pause,
                                   bg="#1e293b", fg=FG, activebackground="#334155",
                                   activeforeground=FG, relief="flat", font=FONT)
        self.pause_btn.pack(side="right", padx=10)

        self.status_var = tk.StringVar(value="正在启动…")
        status = tk.Label(self.root, textvariable=self.status_var, bg=BG, fg=DIM,
                          font=FONT_S, anchor="w")
        status.pack(fill="x", padx=12, pady=(6, 2))

        self.spec_canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.spec_canvas.pack(fill="both", expand=True, padx=8, pady=(2, 3))
        self.wave_canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True, padx=8, pady=(3, 8))

        self.spec_canvas.bind("<Configure>", lambda e: self._recalc_bars())
        self._recalc_bars()

    def _recalc_bars(self):
        w = self.spec_canvas.winfo_width()
        if w > 50:
            self.n_bars = max(24, min(100, (w - 24) // 9))
        if self.bars.size != self.n_bars:
            self.bars = np.full(self.n_bars, DB_FLOOR, dtype=np.float64)
            self.peaks = np.full(self.n_bars, DB_FLOOR, dtype=np.float64)

    def _populate_devices(self):
        devices = self.engine.list_loopbacks()
        self.device_map = {"默认输出设备（自动）": None}
        names = ["默认输出设备（自动）"]
        for d in devices:
            label = f"{d['name']}   @{d['rate']}Hz"
            while label in self.device_map:
                label += " "
            self.device_map[label] = d
            names.append(label)
        self.combo["values"] = names
        if self.follow_default.get() or self.engine.device is None:
            self.combo.current(0)

    # ================= 采集控制 =================
    def _start_engine(self, device):
        if self.engine.start(device):
            d = self.engine.device
            self.status_var.set(
                f"● 正在采集: {d['name']}   {self.engine.sample_rate} Hz / {self.engine.channels}ch")
        else:
            self.status_var.set(f"✕ 采集失败: {self.engine.error} （1.5 秒后自动重试）")
            self.retry_at = time.time() + 1.5

    def _on_pick_device(self, _event=None):
        dev = self.device_map.get(self.combo.get())
        if dev is None:
            self.follow_default.set(True)
            self._start_engine(None)
        else:
            self.follow_default.set(False)
            self._start_engine(dev)

    def _on_follow_toggle(self):
        if self.follow_default.get():
            self.combo.current(0)
            self._start_engine(None)

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn["text"] = "继续" if self.paused else "暂停"

    def _watchdog(self):
        """跟随默认设备切换 / 掉线自动重连"""
        try:
            if not self.paused:
                if time.time() >= self.retry_at:
                    st = self.engine.stream
                    active = False
                    if st is not None:
                        try:
                            active = st.is_active()
                        except Exception:
                            active = False
                    if not active or self.engine.error:
                        self._start_engine(None if self.follow_default.get()
                                           else self.device_map.get(self.combo.get()))
                    elif self.follow_default.get():
                        d = self.engine.default_loopback()
                        if d is not None and self.engine.device is not None \
                                and d["name"] != self.engine.device["name"]:
                            self._start_engine(d)
        except Exception:
            pass
        self.root.after(1500, self._watchdog)

    def _on_close(self):
        try:
            self.engine.close()
        finally:
            self.root.destroy()

    # ================= 绘制 =================
    def _fft_size(self):
        """按采样率自适应 FFT 点数，保持 ~43ms 时窗（48k→2048, 192k→8192）"""
        sr = self.engine.sample_rate or 48000
        target = max(1024, int(0.043 * sr))
        return min(1 << int(round(np.log2(target))), 16384)

    def _wave_size(self):
        """波形窗口采样数，保持 ~45ms 时窗"""
        return max(1024, int(0.045 * (self.engine.sample_rate or 48000)))

    def _tick(self):
        if not self.paused:
            try:
                self._draw_spectrum()
                self._draw_wave()
            except Exception:
                pass
        self.frames += 1
        now = time.time()
        if now - self.fps_t0 >= 0.5:
            self.fps = self.frames / (now - self.fps_t0)
            self.frames, self.fps_t0 = 0, now
            base = self.status_var.get().split("   [")[0].split("   FPS")[0]
            tag = "   [已暂停]" if self.paused else ""
            self.status_var.set(f"{base}{tag}   FPS {self.fps:.0f}")
        self.root.after(16, self._tick)

    def _compute_spectrum(self, x):
        sr = max(8000, self.engine.sample_rate or 48000)
        N = self._fft_size()
        if x.size < N:
            x = np.pad(x, (0, N - x.size))
        else:
            x = x[-N:]
        win = np.hanning(N)
        spec = np.abs(np.fft.rfft(x * win))
        db = 20.0 * np.log10(spec / (N / 4.0) + 1e-10)
        np.clip(db, DB_FLOOR, 0.0, out=db)

        freqs = np.fft.rfftfreq(N, 1.0 / sr)
        top_f = min(MAX_F, sr / 2.0 - 1.0)
        edges = np.geomspace(MIN_F, top_f, self.n_bars + 1)
        idx = np.searchsorted(freqs, edges)
        idx[0] = 0
        idx[-1] = len(freqs)
        vals = np.empty(self.n_bars)
        for i in range(self.n_bars):
            a, b = idx[i], max(idx[i + 1], idx[i] + 1)
            b = min(b, len(freqs))
            vals[i] = db[a:b].max() if b > a else DB_FLOOR
        return vals

    def _draw_spectrum(self):
        c = self.spec_canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return
        c.delete("all")

        baseline = h - 16
        top_m = 22.0
        span = -DB_FLOOR

        def y_of(db):
            return baseline - (db - DB_FLOOR) / span * (baseline - top_m)

        # 网格 + 刻度
        for db in (0, -20, -40, -60, -80):
            y = y_of(db)
            c.create_line(34, y, w - 4, y, fill=GRID)
            c.create_text(30, y, text=f"{db}", fill=DIM, font=FONT_S, anchor="e")
        sr = max(8000, self.engine.sample_rate or 48000)
        top_f = min(MAX_F, sr / 2.0 - 1.0)
        for f, label in ((100, "100Hz"), (1000, "1kHz"), (10000, "10kHz")):
            if MIN_F < f < top_f:
                x = 34 + (np.log(f) - np.log(MIN_F)) / (np.log(top_f) - np.log(MIN_F)) * (w - 44)
                c.create_line(x, top_m - 6, x, baseline, fill=GRID)
                c.create_text(x, baseline + 9, text=label, fill=DIM, font=FONT_S)
        c.create_text(8, 6, text=f"频谱 SPECTRUM  {MIN_F:.0f}Hz - {top_f / 1000:.1f}kHz  (对数刻度)",
                      fill=DIM, font=FONT_S, anchor="w")

        # 频谱柱
        x = self.engine.read_last(self._fft_size())
        vals = self._compute_spectrum(x)
        if self.bars.size != self.n_bars:
            self.bars = np.full(self.n_bars, DB_FLOOR, dtype=np.float64)
            self.peaks = np.full(self.n_bars, DB_FLOOR, dtype=np.float64)
        self.bars = np.maximum(vals, self.bars - BAR_RELEASE)
        self.peaks = np.maximum(self.bars, self.peaks - PEAK_FALL)

        n = self.n_bars
        gap = 2.0
        bw = max(1.0, (w - 44 - gap * (n + 1)) / n)
        x0 = 36.0
        lut = COLOR_LUT
        for i in range(n):
            v = float(self.bars[i])
            if v <= DB_FLOOR + 0.5:
                continue
            t = min(1.0, max(0.0, (v - DB_FLOOR) / span))
            col = lut[int(t * (len(lut) - 1))]
            xa = x0 + i * (bw + gap)
            ya = y_of(v)
            c.create_rectangle(xa, ya, xa + bw, baseline, fill=col, width=0)
            pk = float(self.peaks[i])
            yp = y_of(pk)
            c.create_rectangle(xa, yp - 2, xa + bw, yp, fill="#e2e8f0", width=0)
        c.create_line(34, baseline, w - 4, baseline, fill=GRID)

    def _draw_wave(self):
        c = self.wave_canvas
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return
        c.delete("all")
        c.create_text(8, 6, text="波形 WAVEFORM", fill=DIM, font=FONT_S, anchor="w")
        cx = h / 2.0

        for frac in (-0.5, 0.5):
            c.create_line(4, cx + cx * frac, w - 4, cx + cx * frac, fill=GRID)

        x = self.engine.read_last(self._wave_size())
        if x.size < 32:
            return
        peak = float(np.abs(x).max())
        if peak > 1e-5:
            target = (h * 0.42) / max(peak, 1e-4)
            target = min(target, 150.0)
            if target < self.wave_gain:
                self.wave_gain = target          # 信号变大立刻压低
            else:
                self.wave_gain += (target - self.wave_gain) * 0.05
        g = self.wave_gain

        cols = int(min(w, x.size))
        m = (x.size // cols) * cols
        xr = x[:m].reshape(cols, -1)
        mins = xr.min(axis=1)
        maxs = xr.max(axis=1)
        step = (w - 8) / (cols - 1)

        pts = []
        for i in range(cols):
            pts.append(4 + i * step)
            pts.append(cx - maxs[i] * g)
        # 上/下包络线 + 填充
        top_pts = pts
        bot_pts = []
        for i in range(cols - 1, -1, -1):
            bot_pts.append(4 + i * step)
            bot_pts.append(cx - mins[i] * g)
        poly = top_pts + bot_pts
        c.create_polygon(*poly, fill="#134e6f", outline="", width=0)
        c.create_line(*top_pts, fill=ACCENT, width=1)
        c.create_line(*bot_pts, fill=ACCENT, width=1)
        c.create_line(4, cx, w - 4, cx, fill="#334155")

    # ================= 启动失败兜底 =================
    def report_init_error(self, msg):
        self.status_var.set(f"✕ {msg}")


def main():
    root = tk.Tk()
    try:
        app = Visualizer(root)
    except Exception as e:
        messagebox.showerror("音频可视化", f"初始化音频系统失败：\n{e}\n\n"
                                        "请确认电脑有可用的音频输出设备。")
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
