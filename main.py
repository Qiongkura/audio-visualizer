# -*- coding: utf-8 -*-
"""
音频可视化程序（ins 风格 UI）
自动识别电脑中正在播放的声音（WASAPI 环回采集，无需虚拟声卡），
实时显示频谱（对数刻度、Instagram 渐变配色）与波形（自动增益包络）。

渲染方式：每帧用 numpy 把频谱+波形整帧画进 RGB 缓冲，
再作为单个 PhotoImage 上屏（避免每帧创建数百个画布元素，
在远程桌面/高负载环境下 Tk 逐元素渲染会掉到个位数帧率）。

依赖: pip install pyaudiowpatch numpy customtkinter
运行: python main.py
"""
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

RING_SIZE = 1 << 15          # 环形缓冲区 32768 采样
MIN_F, MAX_F = 30.0, 16000.0 # 频谱显示频率范围
DB_FLOOR = -85.0             # 频谱底噪（dB）
BAR_RELEASE = 1.5            # 频谱柱每帧下落 dB
PEAK_FALL = 0.55             # 峰值帽每帧下落 dB

# ---------- 配色（Instagram 风格） ----------
BG      = "#F4F4F6"   # 应用背景
CARD    = "#FFFFFF"   # 卡片
TXT     = "#141419"   # 主文字
SUB     = "#9A9AA3"   # 次要文字
FAINT   = "#C9C9D1"   # 坐标刻度
LINE_RGB  = (241, 241, 245)          # 网格线
ACCENT    = "#E1306C"
ACCENT_RGB  = (225, 48, 108)
WASH_TOP    = np.array([255, 255, 255], dtype=np.uint8)
WASH_BOT    = np.array([252, 246, 242], dtype=np.uint8)
GRAD_STOPS = ("#833AB4", "#C13584", "#E1306C", "#FD1D1D", "#F77737", "#FCAF45")

FONT_FAMILY = "Microsoft YaHei UI"
SPEC_L, SPEC_R = 46, 16       # 频谱卡左右留白
SPEC_TOP, SPEC_BASE = 32, 26  # 频谱卡上/下留白
WAVE_L, WAVE_R = 14, 14


def hex2rgb(s):
    return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))


def rgb2hex(c):
    return "#%02x%02x%02x" % c


def lerp3(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def gradient_lut(stops, n):
    cols = [hex2rgb(s) for s in stops]
    lut = []
    for i in range(n):
        t = i / (n - 1) * (len(cols) - 1)
        k = min(int(t), len(cols) - 2)
        lut.append(lerp3(cols[k], cols[k + 1], t - k))
    return lut


BAR_RGB = gradient_lut(GRAD_STOPS, 96)                  # 频谱柱: 按频率位置渐变
CAP_RGB = [tuple(int(v * 0.72) for v in c) for c in BAR_RGB]  # 峰值帽: 同色系加深


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

    def list_loopbacks(self):
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

    def start(self, device=None):
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
        ctk.set_appearance_mode("light")
        root.title("Audio Visualizer · 音频可视化")
        root.geometry("1080x660")
        root.minsize(760, 500)
        root.configure(fg_color=BG)

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

        # 每张画布的渲染状态
        self._render = {}   # canvas -> dict(img_id, photo, buf, ...)
        self._sr_used = 0

        self._build_ui()
        self._populate_devices()
        self._start_engine(None)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(16, self._tick)
        root.after(1500, self._watchdog)

    # ================= UI =================
    def _build_ui(self):
        f_title = ctk.CTkFont(FONT_FAMILY, 20, "bold")
        f_sub = ctk.CTkFont(FONT_FAMILY, 11)
        f_ui = ctk.CTkFont(FONT_FAMILY, 12)
        f_s = ctk.CTkFont(FONT_FAMILY, 10)

        head = ctk.CTkFrame(self.root, fg_color="transparent")
        head.pack(fill="x", padx=22, pady=(16, 4))

        logo = tk.Canvas(head, width=104, height=36, bg=BG, highlightthickness=0)
        logo.pack(side="left")
        for i, col in enumerate(GRAD_STOPS):
            x = 8 + i * 19
            logo.create_oval(x - 5, 13, x + 5, 23, fill=col, width=0)

        titles = ctk.CTkFrame(head, fg_color="transparent")
        titles.pack(side="left", padx=(6, 0))
        ctk.CTkLabel(titles, text="Audio Visualizer", font=f_title,
                     text_color=TXT).pack(anchor="w")
        ctk.CTkLabel(titles, text="系统声音 · 实时频谱与波形", font=f_sub,
                     text_color=SUB).pack(anchor="w")

        self.pause_btn = ctk.CTkButton(head, text="暂停", width=92, height=36,
                                       corner_radius=18,
                                       font=ctk.CTkFont(FONT_FAMILY, 13, "bold"),
                                       fg_color=ACCENT, hover_color="#C13584",
                                       command=self._toggle_pause)
        self.pause_btn.pack(side="right", padx=(10, 0))

        ctrl = ctk.CTkFrame(self.root, fg_color="transparent")
        ctrl.pack(fill="x", padx=22, pady=(6, 2))

        self.combo = ctk.CTkComboBox(
            ctrl, width=380, height=34, corner_radius=10, font=f_ui,
            border_width=1, border_color="#E6E6EB", fg_color=CARD,
            text_color=TXT, button_color="#E6E6EB", button_hover_color="#D9D9E0",
            dropdown_fg_color=CARD, dropdown_text_color=TXT,
            dropdown_hover_color="#F8E8EE", dropdown_font=f_ui,
            command=self._on_pick_device)
        self.combo.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(ctrl, text="刷新", width=68, height=34, corner_radius=10,
                      font=f_ui, fg_color="#EBEBEF", hover_color="#E0E0E6",
                      text_color=TXT,
                      command=self._populate_devices).pack(side="left", padx=(10, 10))

        ctk.CTkSwitch(ctrl, text="跟随默认输出设备", variable=self.follow_default,
                      command=self._on_follow_toggle, font=f_ui, text_color=TXT,
                      progress_color=ACCENT).pack(side="left")

        srow = ctk.CTkFrame(self.root, fg_color="transparent")
        srow.pack(fill="x", padx=24, pady=(6, 6))
        self.status_var = tk.StringVar(value="正在启动…")
        ctk.CTkLabel(srow, textvariable=self.status_var, font=f_s,
                     text_color=SUB, anchor="w").pack(side="left")
        self.fps_lbl = ctk.CTkLabel(srow, text="— FPS", font=f_s,
                                    text_color="#6F6F78", fg_color="#EBEBEF",
                                    corner_radius=11, height=22, width=64)
        self.fps_lbl.pack(side="right")

        spec_card = ctk.CTkFrame(self.root, fg_color=CARD, corner_radius=18)
        spec_card.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        self.spec_canvas = tk.Canvas(spec_card, bg=CARD, highlightthickness=0, bd=0)
        self.spec_canvas.pack(fill="both", expand=True, padx=2, pady=2)

        wave_card = ctk.CTkFrame(self.root, fg_color=CARD, corner_radius=18)
        wave_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.wave_canvas = tk.Canvas(wave_card, bg=CARD, highlightthickness=0, bd=0)
        self.wave_canvas.pack(fill="both", expand=True, padx=2, pady=2)

        self.spec_canvas.bind("<Configure>", lambda e: self._setup_render(self.spec_canvas, "spec"))
        self.wave_canvas.bind("<Configure>", lambda e: self._setup_render(self.wave_canvas, "wave"))

    # ================= 渲染状态（resize / 采样率变化时重建） =================
    def _setup_render(self, c, kind):
        w, h = c.winfo_width(), c.winfo_height()
        if w < 60 or h < 60:
            return
        if kind == "spec":
            self.n_bars = max(24, min(90, (w - 58) // 11))
            if self.bars.size != self.n_bars:
                self.bars = np.full(self.n_bars, DB_FLOOR, dtype=np.float64)
                self.peaks = np.full(self.n_bars, DB_FLOOR, dtype=np.float64)

        c.delete("all")
        img_id = c.create_image(0, 0, anchor="nw")

        # 纵向渐变底纹
        t = np.linspace(0, 1, h).reshape(h, 1, 1)
        buf = (WASH_TOP.reshape(1, 1, 3) * (1 - t) +
               WASH_BOT.reshape(1, 1, 3) * t).astype(np.uint8)
        buf = np.repeat(buf, w, axis=1)

        baseline = h - SPEC_BASE
        top_m = SPEC_TOP

        def y_of(db):
            return baseline - (db - DB_FLOOR) / (-DB_FLOOR) * (baseline - top_m)

        baseline = h - SPEC_BASE
        top_m = SPEC_TOP

        def y_of(db):
            return baseline - (db - DB_FLOOR) / (-DB_FLOOR) * (baseline - top_m)

        # 静态网格线直接烘焙进底图
        line = np.array(LINE_RGB, dtype=np.uint8)
        if kind == "spec":
            for db in (0, -20, -40, -60, -80):
                y = int(y_of(db))
                buf[y, SPEC_L:w - SPEC_R] = line
            sr = max(8000, self.engine.sample_rate or 48000)
            top_f = min(MAX_F, sr / 2.0 - 1.0)
            for f in (100, 1000, 10000):
                if MIN_F < f < top_f:
                    x = int(SPEC_L + (np.log(f) - np.log(MIN_F)) /
                            (np.log(top_f) - np.log(MIN_F)) * (w - SPEC_L - SPEC_R))
                    buf[top_m - 4:baseline, x] = line
            buf[baseline, :] = line
        else:
            cy = h // 2
            buf[cy, WAVE_L:w - WAVE_R] = line
            buf[int(cy - cy * 0.5), WAVE_L:w - WAVE_R] = line
            buf[int(cy + cy * 0.5), WAVE_L:w - WAVE_R] = line

        wash = buf.copy()
        st = {"img_id": img_id, "photo": None, "buf": buf, "wash": wash,
              "w": w, "h": h, "kind": kind, "last": 0.0}
        self._render[c] = st
        self._sr_used = self.engine.sample_rate

        # 文本刻度（静态画布元素，位于图片上层）
        if kind == "spec":
            for db in (0, -20, -40, -60, -80):
                y = y_of(db)
                c.create_text(SPEC_L - 6, y, text=f"{db}", fill=FAINT,
                              font=(FONT_FAMILY, 8), anchor="e")
            sr = max(8000, self.engine.sample_rate or 48000)
            top_f = min(MAX_F, sr / 2.0 - 1.0)
            for f, label in ((100, "100Hz"), (1000, "1kHz"), (10000, "10kHz")):
                if MIN_F < f < top_f:
                    x = SPEC_L + (np.log(f) - np.log(MIN_F)) / \
                        (np.log(top_f) - np.log(MIN_F)) * (w - SPEC_L - SPEC_R)
                    c.create_text(x, baseline + 10, text=label, fill=FAINT,
                                  font=(FONT_FAMILY, 8))
            c.create_text(18, 10, text="SPECTRUM · 频谱", fill=SUB,
                          font=(FONT_FAMILY, 9, "bold"), anchor="w")
        else:
            c.create_text(18, 10, text="WAVEFORM · 波形", fill=SUB,
                          font=(FONT_FAMILY, 9, "bold"), anchor="w")

    def _push(self, c):
        """把 numpy 缓冲作为单张图片上屏"""
        st = self._render[c]
        buf = st["buf"]
        data = b"P6\n%d %d\n255\n" % (buf.shape[1], buf.shape[0]) + buf.tobytes()
        ph = tk.PhotoImage(data=data, format="ppm", master=c)
        c.itemconfigure(st["img_id"], image=ph)
        st["photo"] = ph   # 持有引用防止被 GC

    # ================= 采集控制 =================
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
        self.combo.configure(values=names)
        if self.follow_default.get() or self.engine.device is None:
            self.combo.set(names[0])

    def _start_engine(self, device):
        if self.engine.start(device):
            d = self.engine.device
            self.status_var.set(
                f"● 正在采集  {d['name']}   {self.engine.sample_rate} Hz / {self.engine.channels}ch")
        else:
            self.status_var.set(f"✕ 采集失败：{self.engine.error} （1.5 秒后自动重试）")
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
            self.combo.set("默认输出设备（自动）")
            self._start_engine(None)

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="继续" if self.paused else "暂停")

    def _watchdog(self):
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

    # ================= 主循环 =================
    def _fft_size(self):
        """按采样率自适应 FFT 点数，保持 ~43ms 时窗（48k→2048, 192k→8192）"""
        sr = self.engine.sample_rate or 48000
        target = max(1024, int(0.043 * sr))
        return min(1 << int(round(np.log2(target))), 16384)

    def _wave_size(self):
        return max(1024, int(0.09 * (self.engine.sample_rate or 48000)))

    def _tick(self):
        # 先排下一帧定时器再绘制，避免绘制耗时造成节拍漂移
        self.root.after(16, self._tick)
        if self.paused:
            return
        try:
            if self._sr_used != self.engine.sample_rate:
                for c, st in list(self._render.items()):
                    self._setup_render(c, st["kind"])
            if self._render:
                # 两张画布交替重绘（每帧只画一张，单帧开销减半）
                c = min(self._render, key=lambda k: self._render[k]["last"])
                self._render[c]["last"] = time.perf_counter()
                if c is self.spec_canvas:
                    self._draw_spectrum()
                else:
                    self._draw_wave()
                self.frames += 1
        except Exception:
            pass
        now = time.time()
        if now - self.fps_t0 >= 0.5:
            self.fps = self.frames / (now - self.fps_t0)
            self.frames, self.fps_t0 = 0, now
            self.fps_lbl.configure(text=f"{self.fps:.0f} FPS")
            base = self.status_var.get().split("   [")[0].split("   FPS")[0]
            self.status_var.set(base)

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
        if c not in self._render:
            return
        st = self._render[c]
        w, h, buf = st["w"], st["h"], st["buf"]
        buf[:] = st["wash"]                       # 每帧从干净底图开始
        baseline = h - SPEC_BASE
        top_m = SPEC_TOP
        span = -DB_FLOOR

        def y_of(db):
            return baseline - (db - DB_FLOOR) / span * (baseline - top_m)

        # 频谱柱
        x = self.engine.read_last(self._fft_size())
        vals = self._compute_spectrum(x)
        self.bars = np.maximum(vals, self.bars - BAR_RELEASE)
        self.peaks = np.maximum(self.bars, self.peaks - PEAK_FALL)

        n = self.n_bars
        gap = 3.0
        bw = max(2.0, (w - 58 - gap * (n + 1)) / n)
        for i in range(n):
            li = int(i / max(1, n - 1) * (len(BAR_RGB) - 1))
            xa = int(SPEC_L + i * (bw + gap))
            xb = min(w - 2, int(xa + bw))
            v = float(self.bars[i])
            if v > DB_FLOOR + 0.5:
                ya = int(y_of(v))
                col = np.array(BAR_RGB[li], dtype=np.uint8)
                if baseline - ya > 8:
                    buf[ya:ya + 2, xa + 2:xb - 2] = col
                    buf[ya + 2:ya + 4, xa + 1:xb - 1] = col
                    buf[ya + 4:baseline, xa:xb] = col
                else:
                    buf[ya:baseline, xa:xb] = col
            pk = float(self.peaks[i])
            yp = min(int(y_of(pk)), baseline - 2)
            if yp > top_m + 2:
                buf[yp - 1:yp + 2, xa + 1:xb - 1] = np.array(CAP_RGB[li], dtype=np.uint8)
        self._push(c)

    def _draw_wave(self):
        c = self.wave_canvas
        if c not in self._render:
            return
        st = self._render[c]
        w, h, buf = st["w"], st["h"], st["buf"]
        buf[:] = st["wash"]                       # 每帧从干净底图开始
        cx = h // 2

        x = self.engine.read_last(self._wave_size())
        if x.size < 32:
            self._push(c)
            return
        peak = float(np.abs(x).max())
        if peak > 1e-5:
            target = (h * 0.42) / max(peak, 1e-4)
            target = min(target, 1500.0)
            if target < self.wave_gain:
                self.wave_gain = target
            else:
                self.wave_gain += (target - self.wave_gain) * 0.05
        g = self.wave_gain

        cols = int(min(600, (w - WAVE_L - WAVE_R), x.size))
        m = (x.size // cols) * cols
        xr = x[:m].reshape(cols, -1)
        # 分桶均值 = 低通后的平滑波形曲线
        ys = xr.mean(axis=1)
        ys = np.convolve(ys, np.ones(3) / 3.0, mode="same")
        step = (w - WAVE_L - WAVE_R - 1) / max(1, cols - 1)

        accent = np.array(ACCENT_RGB, dtype=np.uint8)
        yv = np.clip((cx - ys * g).astype(int), 0, h - 1)
        # 逐列连线：每段 x 覆盖到下一个点的横坐标，纵向填满两点之间，
        # 宽度按实际点间距取，构造上保证没有横向缝隙
        for i in range(cols):
            x0 = WAVE_L + int(i * step)
            x1 = WAVE_L + int((i + 1) * step) if i + 1 < cols else x0 + 2
            x1 = max(x1, x0 + 2)
            a, b = int(yv[i]), int(yv[min(i + 1, cols - 1)])
            lo, hi = (a, b) if a <= b else (b, a)
            buf[lo:hi + 1, x0:x1] = accent
        self._push(c)


def main():
    root = ctk.CTk()
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
