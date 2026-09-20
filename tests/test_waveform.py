# -*- coding: utf-8 -*-
"""波形降采样与自动增益单元测试"""
import unittest

import numpy as np

from audio_visualizer.dsp import WaveformAnalyzer


def sine(amp, n=4096, f=440.0, sr=48000):
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


class TestWaveformAnalyzer(unittest.TestCase):
    def test_output_length_matches_columns(self):
        wa = WaveformAnalyzer()
        for cols in (2, 17, 200, 4096, 5000):
            ys, _ = wa.process(sine(0.3), cols)
            self.assertEqual(ys.size, cols)

    def test_empty_input_is_flat(self):
        wa = WaveformAnalyzer()
        ys, gain = wa.process(np.empty(0, dtype=np.float32), 64)
        self.assertEqual(ys.size, 64)
        self.assertTrue(np.all(ys == 0.0))
        self.assertGreater(gain, 0)

    def test_gain_scales_signal_to_target(self):
        """自动增益的目标是让峰值占到约 42% 半高"""
        wa = WaveformAnalyzer()
        h = 400.0
        for amp in (0.02, 0.1, 0.5, 1.0):
            with self.subTest(amp=amp):
                wa = WaveformAnalyzer()
                for _ in range(40):                 # 让增益收敛
                    ys, gain = wa.process(sine(amp), 400)
                disp = float(np.abs(ys).max()) * gain * h
                self.assertGreater(disp, 0.42 * h * 0.85)
                self.assertLess(disp, 0.42 * h * 1.15)

    def test_gain_drops_instantly_on_loud_input(self):
        wa = WaveformAnalyzer(gain=1000.0)
        wa.process(sine(1.0), 64)
        self.assertLess(wa.gain, 1.0)

    def test_gain_rises_smoothly_on_quiet_input(self):
        wa = WaveformAnalyzer(gain=1.0)
        wa.process(sine(0.001), 64)
        self.assertGreater(wa.gain, 1.0)        # 确实在往上追
        self.assertLess(wa.gain, 30.0)          # 但一次只走一小步，不会跳变

    def test_gain_is_capped(self):
        wa = WaveformAnalyzer()
        for _ in range(200):
            wa.process(sine(1e-4), 64)
        self.assertGreater(wa.gain, wa.max_gain * 0.999)
        self.assertLessEqual(wa.gain, wa.max_gain)

    def test_silence_keeps_gain(self):
        wa = WaveformAnalyzer(gain=12.0)
        ys, gain = wa.process(np.zeros(4096, dtype=np.float32), 128)
        self.assertAlmostEqual(gain, 12.0)
        self.assertAlmostEqual(float(np.abs(ys).max()), 0.0)

    def test_reset(self):
        wa = WaveformAnalyzer()
        wa.process(sine(0.5), 64)
        wa.reset()
        self.assertAlmostEqual(wa.gain, 20.0)

    def test_short_signal_is_stretched(self):
        """样本比像素少时用插值拉伸，不能崩也不能丢长度"""
        wa = WaveformAnalyzer()
        ys, _ = wa.process(sine(0.5, n=8), 64)
        self.assertEqual(ys.size, 64)
        self.assertGreater(float(np.abs(ys).max()), 0.0)

    def test_reduction_uses_bucket_mean(self):
        wa = WaveformAnalyzer()
        x = np.concatenate([np.ones(100, dtype=np.float32),
                            -np.ones(100, dtype=np.float32)])
        ys, _ = wa.process(x, 2)
        # 两个桶分别是 +1 和 -1；3 点平滑在 2 个点上会轻微衰减
        self.assertGreater(ys[0], 0.5)
        self.assertLess(ys[1], -0.5)


if __name__ == "__main__":
    unittest.main()
