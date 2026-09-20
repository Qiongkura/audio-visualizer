# -*- coding: utf-8 -*-
"""Tk 界面层：只负责显示与交互，DSP 全部交给后台线程。

主循环每帧只做两件事：取“最新一帧”DSP 结果、把它画进帧缓冲并上屏。
FFT、降采样、增益计算都不在 Tk 线程里跑，窗口缩放也不会每帧重建底图。
"""
import logging
import os
import sys
import time

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

from .audio_engine import AudioEngine, DeviceSupervisor, EngineError
from .config import (ACCENT, APP_TITLE, BG, CARD, FPS_INTERVAL, FONT_FAMILY,
                     GRAD_STOPS, RESIZE_DEBOUNCE_MS, RETRY_DELAY,
                     SPEC_BARS_MAX, SPEC_BARS_MIN, SPEC_BAR_PITCH, SUB,
                     TICK_MS, TXT, WATCHDOG_MS, WAVE_L, WAVE_R)
from .dsp import DspWorker
from .renderer import CanvasRenderer

log = logging.getLogger("audio_visualizer.app")

AUTO_LABEL = "默认输出设备（自动）"


def log_path():
    """日志与 exe 同目录（打包后工作目录不确定，不能用相对路径）"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "error.log")


def setup_logging(path=None, level=logging.INFO):
    """把日志同时写到 error.log 和 stderr（pythonw 下没有 stderr，忽略即可）"""
    root = logging.getLogger("audio_visualizer")
    if root.handlers:
        return root
    root.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    try:
        fh = logging.FileHandler(path or log_path(), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    return root


class Visualizer:
    def __init__(self, root):
        self.root = root
        self.paused = False
        self.follow_default = tk.BooleanVar(value=True)
        self.device_map = {}
        self.manual_device = None
        self.frames = 0
        self.fps_t0 = time.time()
        self.fps = 0.0
        self._err_throttle = 0.0
        self._sr_used = 0
        self._renderers = {}
        self._last_drawn = {"spec": 0.0, "wave": 0.0}
        self._resize_jobs = {}
        self._wave_cols = 256
        self._last_frame = None
        self._state = ("starting", "正在启动…")

        ctk.set_appearance_mode("light")
        root.title(APP_TITLE)
        root.geometry("1080x660")
        root.minsize(760, 500)
        root.configure(fg_color=BG)

        self.engine = AudioEngine()
        self.supervisor = DeviceSupervisor(self.engine)

        self._build_ui()

        self.dsp = DspWorker(read_samples=self.engine.read_last,
                             sample_rate=lambda: self.engine.sample_rate,
                             wave_cols=lambda: self._wave_cols,
                             n_bars=64)
        self.dsp.start()

        root.update_idletasks()
        for kind in ("spec", "wave"):
            self._apply_resize(self._renderers[kind].canvas, kind)

        self._populate_devices()
        self._start_engine(None)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.report_callback_exception = self._log_tk_error
        root.after(TICK_MS, self._tick)
        root.after(WATCHDOG_MS, self._watchdog)

    # ================= 日志 =================
    def _log_tk_error(self, exc, val, tb):
        """无控制台运行(pythonw)时把 Tk 回调异常落盘，便于排查"""
        log.error("Tk 回调异常", exc_info=(exc, val, tb))

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
        for i, col in enumerate(("#833AB4", "#C13584", "#E1306C",
                                 "#FD1D1D", "#F77737", "#FCAF45")):
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
        self.status_var = tk.StringVar(value=self._state[1])
        self.status_lbl = ctk.CTkLabel(srow, textvariable=self.status_var,
                                       font=f_s, text_color=SUB, anchor="w")
        self.status_lbl.pack(side="left")
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

        self._renderers["spec"] = CanvasRenderer(self.spec_canvas, "spec")
        self._renderers["wave"] = CanvasRenderer(self.wave_canvas, "wave")
        for kind, canvas in (("spec", self.spec_canvas), ("wave", self.wave_canvas)):
            canvas.bind("<Configure>",
                        lambda e, k=kind: self._on_configure(e.widget, k))

    # ================= 尺寸变化（防抖） =================
    def _on_configure(self, canvas, kind):
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 60 or h < 60:
            return
        r = self._renderers.get(kind)
        if r is not None and r.size == (w, h):
            return                              # 尺寸没变，跳过
        job = self._resize_jobs.pop(canvas, None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        # 拖动窗口时每 16ms 就会触发一次 Configure，攒够 160ms 再真正重建
        self._resize_jobs[canvas] = self.root.after(
            RESIZE_DEBOUNCE_MS, self._apply_resize, canvas, kind)

    def _apply_resize(self, canvas, kind):
        self._resize_jobs.pop(canvas, None)
        w, h = canvas.winfo_width(), canvas.winfo_height()
        if w < 60 or h < 60:
            return
        r = self._renderers.get(kind)
        if r is None or r.size == (w, h):
            return
        r.attach(w, h, self.engine.sample_rate)
        if kind == "spec":
            n = max(SPEC_BARS_MIN, min(SPEC_BARS_MAX,
                                      (w - 58) // SPEC_BAR_PITCH))
            r.fb.resize_bars(n)
            self.dsp.n_bars = n
            self._last_frame = None             # 柱子数变了，旧帧作废
        else:
            self._wave_cols = max(2, w - WAVE_L - WAVE_R)
        self._sr_used = self.engine.sample_rate
        self._last_drawn[kind] = 0.0

    def _rebuild_all(self):
        sr = self.engine.sample_rate
        for kind, r in self._renderers.items():
            r.attach(r.fb.w, r.fb.h, sr)
            if kind == "spec":
                r.fb.resize_bars(self.dsp.n_bars)
        self._sr_used = sr

    # ================= 状态栏 =================
    def _set_state(self, kind, msg):
        self._state = (kind, msg)
        self._refresh_status()

    def _refresh_status(self):
        kind, msg = self._state
        extra = self.engine.status_snapshot()
        self.status_var.set(msg + ("   [%s]" % extra if extra else ""))
        try:
            self.status_lbl.configure(text_color=SUB if kind == "running" else ACCENT)
        except Exception:
            pass

    # ================= 采集控制 =================
    def _populate_devices(self):
        try:
            devices = self.engine.list_loopbacks()
        except EngineError as e:
            log.error("%s", e)
            self._set_state("error", "✕ " + str(e))
            return False
        self.device_map = {AUTO_LABEL: None}
        names = [AUTO_LABEL]
        for d in devices:
            label = d.label
            while label in self.device_map:
                label += " "
            self.device_map[label] = d
            names.append(label)
        self.combo.configure(values=names)
        if self.follow_default.get() or self.engine.device is None:
            self.combo.set(AUTO_LABEL)
        else:
            self._sync_combo(self.engine.device)
        return True

    def _sync_combo(self, device):
        for label, d in self.device_map.items():
            if d is not None and d.uid == device.uid:
                self.combo.set(label)
                return

    def _start_engine(self, device):
        prev_uid = self.engine.device.uid if self.engine.device else None
        ok = self.engine.start(device)
        self.supervisor.note_started(ok)
        if not ok:
            self._set_state("retry", "⟳ 采集失败：%s（%.1f 秒后重试）"
                            % (self.engine.error, RETRY_DELAY))
            return False
        if self.engine.device.uid != prev_uid:
            self._reset_view()
        d = self.engine.device
        self._set_state("running", "● 正在采集  %s   %d Hz / %dch"
                        % (d.name, self.engine.sample_rate, self.engine.channels))
        if not self.follow_default.get():
            self._sync_combo(d)
        return True

    def _reset_view(self):
        for r in self._renderers.values():
            r.reset_bars()
        self.dsp.reset()
        self._last_frame = None

    def _on_pick_device(self, _event=None):
        dev = self.device_map.get(self.combo.get())
        if dev is None:
            self.follow_default.set(True)
            self.manual_device = None
            self._start_engine(None)
        else:
            self.follow_default.set(False)
            self.manual_device = dev
            self._start_engine(dev)

    def _on_follow_toggle(self):
        if self.follow_default.get():
            self.manual_device = None
            self.combo.set(AUTO_LABEL)
            self._start_engine(None)
        else:
            self.manual_device = self.device_map.get(self.combo.get())

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="继续" if self.paused else "暂停")

    # ================= 看门狗 =================
    def _watchdog(self):
        try:
            d = self.supervisor.tick(paused=self.paused,
                                     follow_default=self.follow_default.get(),
                                     manual_device=self.manual_device)
            if d.action == "reconnect":
                self._start_engine(d.device)
            elif d.action == "switch":
                self._start_engine(d.device)
            elif d.action == "lost":
                # 手动选的设备没了就明确报断开，不偷偷切到别的设备
                log.warning("%s", d.message)
                self._set_state("lost", "✕ %s（请重新选择设备）" % d.message)
            elif d.action == "error":
                log.error("%s", d.message)
                self._set_state("error", "✕ " + d.message)
        except Exception:
            log.exception("看门狗异常")
        self.root.after(WATCHDOG_MS, self._watchdog)

    def _on_close(self):
        try:
            self.dsp.stop()
        finally:
            try:
                self.engine.close()
            finally:
                self.root.destroy()

    # ================= 主循环 =================
    def _tick(self):
        # 先排下一帧定时器再绘制，避免绘制耗时造成节拍漂移
        self.root.after(TICK_MS, self._tick)
        if self.paused or self._resize_jobs:
            return
        try:
            if self._sr_used != self.engine.sample_rate:
                self._rebuild_all()
            frame = self.dsp.latest()
            if frame is not None:
                self._last_frame = frame
            frame = self._last_frame
            if frame is None or not self._renderers:
                return
            # 两张画布交替重绘（每帧只画一张，单帧开销减半）
            kind = "spec" if self._last_drawn["spec"] <= self._last_drawn["wave"] \
                else "wave"
            self._last_drawn[kind] = time.perf_counter()
            r = self._renderers[kind]
            if kind == "spec":
                r.fb.draw_spectrum(frame.bars)
            else:
                r.fb.draw_wave(frame.wave, frame.gain)
            r.blit()
            self.frames += 1
        except Exception:
            now_t = time.time()
            if now_t - self._err_throttle >= 1.0:   # 节流，避免每帧刷爆日志
                self._err_throttle = now_t
                log.exception("渲染帧异常")

        now = time.time()
        if now - self.fps_t0 >= FPS_INTERVAL:
            self.fps = self.frames / (now - self.fps_t0)
            self.frames, self.fps_t0 = 0, now
            self.fps_lbl.configure(text="%.0f FPS" % self.fps)
            self._refresh_status()


def main():
    setup_logging()
    root = ctk.CTk()
    try:
        Visualizer(root)
    except EngineError as e:
        messagebox.showerror("音频可视化", "音频系统初始化失败：\n%s\n\n"
                                           "请确认已安装 pyaudiowpatch 且电脑有音频输出设备。" % e)
        root.destroy()
        return 1
    except Exception as e:
        log.exception("初始化失败")
        messagebox.showerror("音频可视化", "初始化音频系统失败：\n%s\n\n"
                                           "请确认电脑有可用的音频输出设备。" % e)
        root.destroy()
        return 1
    root.mainloop()
    return 0
