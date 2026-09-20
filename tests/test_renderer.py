# -*- coding: utf-8 -*-
"""渲染缓冲单元测试（不需要真窗口，Pillow 路径单独可选测）"""
import unittest
from unittest import mock

import numpy as np

from audio_visualizer import renderer
from audio_visualizer.config import ACCENT_RGB, DB_FLOOR, SPEC_BASE, WASH_BOT
from audio_visualizer.renderer import (CanvasRenderer, FrameBuffer, freq_axis_top,
                                       x_of_freq, y_of_db)
from tests.fakes import FakeCanvas


class StubPhoto:
    """假 tk.PhotoImage，只记录收到的 PPM 数据"""

    instances = []

    def __init__(self, **kw):
        self.kw = kw
        self.data = kw.get("data", b"")
        StubPhoto.instances.append(self)


def ink(buf, wash):
    """统计与底图不同的像素数（即“画上去的东西”）"""
    return int(np.any(buf != wash, axis=2).sum())


class TestGeometryHelpers(unittest.TestCase):
    def test_y_of_db_is_monotonic(self):
        ys = [y_of_db(db, 400) for db in (0, -20, -40, -60, -80, -85)]
        self.assertEqual(ys, sorted(ys))
        self.assertGreaterEqual(min(ys), 0)
        self.assertLessEqual(max(ys), 400 - SPEC_BASE)

    def test_x_of_freq_is_monotonic(self):
        xs = [x_of_freq(f, 900, 48000) for f in (30, 100, 1000, 10000, 16000)]
        self.assertEqual(xs, sorted(xs))
        self.assertLessEqual(max(xs), 900)

    def test_freq_axis_top_never_exceeds_nyquist(self):
        self.assertEqual(freq_axis_top(48000), 16000.0)
        self.assertEqual(freq_axis_top(8000), 3999.0)
        self.assertEqual(freq_axis_top(0), 16000.0)


class TestFrameBuffer(unittest.TestCase):
    def setUp(self):
        self.fb = FrameBuffer("spec")
        self.assertTrue(self.fb.build(800, 400, 48000))
        self.fb.resize_bars(64)

    def test_buffer_shape_and_dtype(self):
        self.assertEqual(self.fb.buf.shape, (400, 800, 3))
        self.assertEqual(self.fb.buf.dtype, np.uint8)
        self.assertFalse(np.shares_memory(self.fb.buf, self.fb.wash))

    def test_invalid_size_is_rejected(self):
        fb = FrameBuffer("spec")
        self.assertFalse(fb.build(0, 100, 48000))
        self.assertIsNone(fb.buf)

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            FrameBuffer("nope")

    def test_grid_is_baked_into_wash(self):
        self.assertTrue(ink(self.fb.wash, self.fb.wash) == 0)
        # 底图本身就应该有网格线和基线，不是纯色
        flat = np.repeat(WASH_BOT.reshape(1, 1, 3), 800, axis=1)
        self.assertGreater(ink(self.fb.wash, flat), 800)

    def test_bar_height_follows_level(self):
        self.fb.draw_spectrum(np.full(64, -10.0))
        loud = ink(self.fb.buf, self.fb.wash)
        self.fb.resize_bars(64)
        self.fb.draw_spectrum(np.full(64, -60.0))
        quiet = ink(self.fb.buf, self.fb.wash)
        self.assertGreater(loud, quiet)
        self.assertGreater(quiet, 0)

    def test_bars_decay_without_new_data(self):
        self.fb.draw_spectrum(np.zeros(64))
        top = float(self.fb.bars.max())
        for _ in range(5):
            self.fb.draw_spectrum(None)
        self.assertLess(float(self.fb.bars.max()), top)
        self.assertGreaterEqual(float(self.fb.bars.max()), DB_FLOOR)

    def test_peaks_hold_above_bars(self):
        self.fb.draw_spectrum(np.zeros(64))
        self.fb.draw_spectrum(np.full(64, -30.0))
        self.fb.draw_spectrum(np.full(64, -70.0))
        self.assertTrue(np.all(self.fb.peaks >= self.fb.bars))
        self.assertTrue(np.all(self.fb.peaks <= 0.0))
        self.assertTrue(np.all(self.fb.bars >= DB_FLOOR))

    def test_mismatched_bar_count_is_skipped(self):
        self.assertFalse(self.fb.draw_spectrum(np.zeros(32)))
        self.assertTrue(self.fb.draw_spectrum(np.zeros(64)))

    def test_wave_draws_accent_pixels(self):
        fb = FrameBuffer("wave")
        fb.build(600, 300, 48000)
        ys = np.sin(np.linspace(0, 20, 400))
        self.assertTrue(fb.draw_wave(ys, 1.0))
        pixels = np.all(fb.buf == np.array(ACCENT_RGB, dtype=np.uint8), axis=2)
        self.assertGreater(int(pixels.sum()), 100)

    def test_wave_handles_missing_data(self):
        fb = FrameBuffer("wave")
        fb.build(600, 300, 48000)
        self.assertFalse(fb.draw_wave(None, 1.0))

    def test_wave_single_column(self):
        fb = FrameBuffer("wave")
        fb.build(60, 60, 48000)
        self.assertTrue(fb.draw_wave(np.array([0.5]), 1.0))

    def test_extreme_levels_do_not_crash(self):
        self.fb.draw_spectrum(np.full(64, 0.0))
        self.fb.draw_spectrum(np.full(64, DB_FLOOR))
        self.fb.draw_spectrum(np.full(64, -1e9))


class TestCanvasRenderer(unittest.TestCase):
    def test_attach_draws_ticks(self):
        c = FakeCanvas(800, 400)
        r = CanvasRenderer(c, "spec")
        self.assertTrue(r.attach(800, 400, 48000))
        self.assertEqual(r.size, (800, 400))
        labels = [k.get("text") for _, k in c.texts]
        self.assertIn("SPECTRUM · 频谱", labels)
        self.assertIn("1kHz", labels)

    def test_blit_ppm_fallback(self):
        """没装 Pillow 时走 PPM 字节流，仍然只创建一次图片对象"""
        StubPhoto.instances = []
        with mock.patch.object(renderer, "HAVE_PILLOW", False), \
                mock.patch.object(renderer.tk, "PhotoImage", StubPhoto):
            c = FakeCanvas(400, 200)
            r = CanvasRenderer(c, "spec")
            r.attach(400, 200, 48000)
            r.fb.resize_bars(32)
            r.fb.draw_spectrum(np.full(32, -20.0))
            self.assertTrue(r.blit())
            self.assertEqual(r.blit_count, 1)
            self.assertTrue(StubPhoto.instances[0].data.startswith(b"P6\n400 200\n255\n"))
            self.assertIn("image", c.conf)

    def test_blit_before_attach_is_noop(self):
        r = CanvasRenderer(FakeCanvas(), "wave")
        self.assertFalse(r.blit())

    @unittest.skipUnless(renderer.HAVE_PILLOW, "未安装 Pillow")
    def test_blit_with_pillow_reuses_photo(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except Exception as e:                    # pragma: no cover
            self.skipTest("没有可用的 Tk 环境: %s" % e)
        try:
            root.withdraw()
            canvas = tk.Canvas(root, width=200, height=120)
            canvas.pack()
            root.update_idletasks()
            r = CanvasRenderer(canvas, "spec")
            self.assertTrue(r.attach(200, 120, 48000))
            r.fb.resize_bars(16)
            r.fb.draw_spectrum(np.full(16, -20.0))
            self.assertTrue(r.blit())
            first = r._photo
            r.blit()
            self.assertIs(r._photo, first)        # 复用同一个 PhotoImage
            self.assertEqual(r.blit_count, 2)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
