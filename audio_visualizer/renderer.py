# -*- coding: utf-8 -*-
"""渲染层：纯 numpy 帧缓冲（可单测）+ Tk 画布上屏。

帧缓冲负责把频谱/波形整帧画进 RGB 数组；上屏只做一次 numpy -> 图片的搬运。
装了 Pillow 时复用同一个 PhotoImage 对象（``paste`` 原地更新），
没装时退回 PPM 字节流方式，功能不受影响。
"""
import tkinter as tk

import numpy as np

from .config import (ACCENT_RGB, BAR_GAP, BAR_RGB, CAP_RGB, DB_FLOOR,
                     FAINT, FONT_FAMILY, LINE_RGB, MAX_F, MIN_F, PEAK_FALL,
                     BAR_RELEASE, SPEC_BASE, SPEC_L, SPEC_R, SPEC_TOP, SUB,
                     WASH_BOT, WASH_TOP, WAVE_L, WAVE_R)

try:
    from PIL import Image, ImageTk
    HAVE_PILLOW = True
except Exception:                       # pragma: no cover - 取决于运行环境
    Image = ImageTk = None
    HAVE_PILLOW = False

DB_TICKS = (0, -20, -40, -60, -80)
FREQ_TICKS = ((100, "100Hz"), (1000, "1kHz"), (10000, "10kHz"))


def freq_axis_top(sample_rate, max_f=MAX_F):
    """频率轴上限：不超过声明上限，也不超过 Nyquist"""
    sr = max(8000, int(sample_rate or 48000))
    return min(float(max_f), sr / 2.0 - 1.0)

def y_of_db(db, h, db_floor=DB_FLOOR, top_m=SPEC_TOP, base=SPEC_BASE):
    baseline = h - base
    return baseline - (db - db_floor) / (-db_floor) * (baseline - top_m)


def x_of_freq(f, w, sample_rate, min_f=MIN_F, max_f=MAX_F,
              left=SPEC_L, right=SPEC_R):
    top_f = freq_axis_top(sample_rate, max_f)
    if top_f <= min_f:
        return float(left)
    return left + (np.log(f) - np.log(min_f)) / \
        (np.log(top_f) - np.log(min_f)) * (w - left - right)


class FrameBuffer:
    """一块 RGB 帧缓冲 + 底图（渐变 + 网格线）。

    每帧从 ``wash`` 拷一份再画，避免重复分配内存。
    """

    def __init__(self, kind):
        if kind not in ("spec", "wave"):
            raise ValueError("未知画布类型: %r" % (kind,))
        self.kind = kind
        self.w = 0
        self.h = 0
        self.buf = None
        self.wash = None
        self._bars = None
        self._peaks = None

    # ---- 构建 ----
    def build(self, w, h, sample_rate):
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            return False
        self.w, self.h = w, h
        t = np.linspace(0, 1, h).reshape(h, 1, 1)
        buf = (WASH_TOP.reshape(1, 1, 3) * (1 - t) +
               WASH_BOT.reshape(1, 1, 3) * t).astype(np.uint8)
        buf = np.repeat(buf, w, axis=1)
        line = np.array(LINE_RGB, dtype=np.uint8)

        if self.kind == "spec":
            for db in DB_TICKS:
                y = int(y_of_db(db, h))
                if 0 <= y < h:
                    buf[y, SPEC_L:w - SPEC_R] = line
            top_f = freq_axis_top(sample_rate)
            for f, _ in FREQ_TICKS:
                if MIN_F < f < top_f:
                    x = int(x_of_freq(f, w, sample_rate))
                    if SPEC_L <= x < w - SPEC_R:
                        buf[SPEC_TOP - 4:h - SPEC_BASE, x] = line
            buf[h - SPEC_BASE, :] = line
            self.reset_bars()
        else:
            cy = h // 2
            buf[cy, WAVE_L:w - WAVE_R] = line
            buf[int(cy - cy * 0.5), WAVE_L:w - WAVE_R] = line
            buf[int(cy + cy * 0.5), WAVE_L:w - WAVE_R] = line

        self.buf = buf
        self.wash = buf.copy()
        return True

    def reset_bars(self):
        n = self._bars.size if self._bars is not None else 0
        if n:
            self._bars = np.full(n, DB_FLOOR, dtype=np.float64)
            self._peaks = np.full(n, DB_FLOOR, dtype=np.float64)

    def resize_bars(self, n):
        self._bars = np.full(int(n), DB_FLOOR, dtype=np.float64)
        self._peaks = np.full(int(n), DB_FLOOR, dtype=np.float64)

    @property
    def bars(self):
        return self._bars

    @property
    def peaks(self):
        return self._peaks

    # ---- 绘制 ----
    def draw_spectrum(self, vals, release=BAR_RELEASE, fall=PEAK_FALL):
        """vals 为长度 == 柱子数的 dB 数组；None 表示这一帧没有新数据（继续下落）"""
        if self.buf is None or self._bars is None:
            return False
        if vals is not None and vals.size != self._bars.size:
            return False                    # 刚 resize，等下一帧
        if vals is not None:
            self._bars = np.maximum(vals, self._bars - release)
        else:
            self._bars = self._bars - release
        np.clip(self._bars, DB_FLOOR, 0.0, out=self._bars)
        self._peaks = np.maximum(self._bars, self._peaks - fall)
        np.clip(self._peaks, DB_FLOOR, 0.0, out=self._peaks)

        w, h, buf = self.w, self.h, self.buf
        buf[:] = self.wash
        baseline = h - SPEC_BASE
        n = self._bars.size
        bw = max(2.0, (w - 58 - BAR_GAP * (n + 1)) / n)
        for i in range(n):
            li = int(i / max(1, n - 1) * (len(BAR_RGB) - 1))
            xa = int(SPEC_L + i * (bw + BAR_GAP))
            xb = min(w - 2, int(xa + bw))
            if xb <= xa:
                continue
            v = float(self._bars[i])
            if v > DB_FLOOR + 0.5:
                ya = int(y_of_db(v, h))
                col = np.array(BAR_RGB[li], dtype=np.uint8)
                if baseline - ya > 8:       # 顶部 4px 做一点收窄，视觉上更像圆头
                    buf[ya:ya + 2, xa + 2:max(xa + 2, xb - 2)] = col
                    buf[ya + 2:ya + 4, xa + 1:max(xa + 1, xb - 1)] = col
                    buf[ya + 4:baseline, xa:xb] = col
                else:
                    buf[ya:baseline, xa:xb] = col
            yp = min(int(y_of_db(float(self._peaks[i]), h)), baseline - 2)
            if yp > SPEC_TOP + 2:
                buf[yp - 1:yp + 2, xa + 1:max(xa + 1, xb - 1)] = \
                    np.array(CAP_RGB[li], dtype=np.uint8)
        return True

    def draw_wave(self, ys, gain):
        """ys 为归一化波形，gain 为自动增益；位移像素 = ys * gain * h"""
        if self.buf is None or ys is None:
            return False
        w, h, buf = self.w, self.h, self.buf
        buf[:] = self.wash
        cx = h // 2
        cols = int(ys.size)
        if cols <= 0:
            return True
        step = (w - WAVE_L - WAVE_R - 1) / max(1, cols - 1)
        accent = np.array(ACCENT_RGB, dtype=np.uint8)
        yv = np.clip((cx - np.asarray(ys, dtype=np.float64) * gain * h)
                     .astype(int), 0, h - 1)
        # 逐列连线：每段覆盖到下一个点的横坐标，纵向填满两点之间，
        # 宽度按实际点间距取，构造上保证没有横向缝隙
        for i in range(cols):
            x0 = WAVE_L + int(i * step)
            x1 = WAVE_L + int((i + 1) * step) if i + 1 < cols else x0 + 1
            x1 = max(x1, x0 + 1)
            a, b = int(yv[i]), int(yv[min(i + 1, cols - 1)])
            lo, hi = (a, b) if a <= b else (b, a)
            buf[lo:hi + 1, x0:x1] = accent
        return True


class CanvasRenderer:
    """把 FrameBuffer 挂到 Tk 画布上，并负责上屏。

    ``attach()`` 只在窗口尺寸/采样率变化时调用（由 app 做防抖），
    正常每帧只走 ``blit()``。
    """

    def __init__(self, canvas, kind):
        self.canvas = canvas
        self.fb = FrameBuffer(kind)
        self.img_id = None
        self._photo = None
        self._blit_count = 0

    @property
    def kind(self):
        return self.fb.kind

    @property
    def size(self):
        return self.fb.w, self.fb.h

    @property
    def blit_count(self):
        return self._blit_count

    def attach(self, w, h, sample_rate):
        """重建底图、图片对象与静态刻度文字"""
        if not self.fb.build(w, h, sample_rate):
            return False
        c = self.canvas
        c.delete("all")
        self.img_id = c.create_image(0, 0, anchor="nw")
        self._photo = None                  # 尺寸变了，图片对象必须重建
        self._draw_ticks(sample_rate)
        return True

    def _draw_ticks(self, sample_rate):
        c, w, h = self.canvas, self.fb.w, self.fb.h
        if self.fb.kind == "spec":
            for db in DB_TICKS:
                c.create_text(SPEC_L - 6, y_of_db(db, h), text="%d" % db,
                              fill=FAINT, font=(FONT_FAMILY, 8), anchor="e")
            top_f = freq_axis_top(sample_rate)
            for f, label in FREQ_TICKS:
                if MIN_F < f < top_f:
                    c.create_text(x_of_freq(f, w, sample_rate), h - SPEC_BASE + 10,
                                  text=label, fill=FAINT, font=(FONT_FAMILY, 8))
            c.create_text(18, 10, text="SPECTRUM · 频谱", fill=SUB,
                          font=(FONT_FAMILY, 9, "bold"), anchor="w")
        else:
            c.create_text(18, 10, text="WAVEFORM · 波形", fill=SUB,
                          font=(FONT_FAMILY, 9, "bold"), anchor="w")

    def reset_bars(self):
        self.fb.reset_bars()

    def blit(self):
        """把帧缓冲推到画布上；返回是否真的上屏了"""
        if self.img_id is None or self.fb.buf is None:
            return False
        buf = self.fb.buf
        if HAVE_PILLOW:
            img = Image.fromarray(buf)
            if self._photo is None:
                self._photo = ImageTk.PhotoImage(img, master=self.canvas)
                self.canvas.itemconfigure(self.img_id, image=self._photo)
            else:
                self._photo.paste(img)      # 复用同一个 PhotoImage，不重建对象
        else:
            data = b"P6\n%d %d\n255\n" % (buf.shape[1], buf.shape[0]) + buf.tobytes()
            ph = tk.PhotoImage(data=data, format="ppm", master=self.canvas)
            self.canvas.itemconfigure(self.img_id, image=ph)
            self._photo = ph                # 持有引用防止被 GC
        self._blit_count += 1
        return True
