# -*- coding: utf-8 -*-
"""频谱分析单元测试：重点是 30Hz~16kHz 的边界准确性"""
import unittest

import numpy as np

from audio_visualizer.config import DB_FLOOR, MAX_F, MIN_F
from audio_visualizer.dsp import (SpectrumAnalyzer, fft_size_for, spectrum_edges,
                                  wave_size_for)

SR = 48000


def bars_of(signal, sample_rate=SR, n_bars=64, analyzer=None):
    an = analyzer or SpectrumAnalyzer(n_bars)
    return an.compute(signal, sample_rate, n_bars)


class TestFftSize(unittest.TestCase):
    def test_power_of_two_and_bounds(self):
        for sr in (8000, 44100, 48000, 96000, 192000):
            n = fft_size_for(sr)
            self.assertEqual(n & (n - 1), 0, "FFT 点数必须是 2 的幂")
            self.assertGreaterEqual(n, 1024)
            self.assertLessEqual(n, 16384)

    def test_tracks_sample_rate(self):
        self.assertEqual(fft_size_for(48000), 2048)
        self.assertEqual(fft_size_for(96000), 4096)
        self.assertEqual(fft_size_for(192000), 8192)

    def test_wave_size(self):
        self.assertEqual(wave_size_for(48000), 4320)
        self.assertGreaterEqual(wave_size_for(8000), 256)


class TestSpectrumEdges(unittest.TestCase):
    def setUp(self):
        self.n_bars = 64
        self.n_fft = fft_size_for(SR)
        self.edges, self.lo, self.hi = spectrum_edges(self.n_bars, SR, self.n_fft)
        self.freqs = np.fft.rfftfreq(self.n_fft, 1.0 / SR)

    def test_first_bar_excludes_dc_and_sub_30hz(self):
        """第一根柱子不能混进 DC 和 30Hz 以下的内容"""
        self.assertGreater(self.lo[0], 0)
        self.assertGreaterEqual(self.freqs[self.lo[0]], MIN_F)
        self.assertLess(self.freqs[self.lo[0] - 1], MIN_F)

    def test_last_bar_excludes_above_16khz(self):
        """最后一根柱子不能一路吃到 Nyquist"""
        self.assertLess(self.hi[-1], self.freqs.size)
        self.assertLessEqual(self.freqs[self.hi[-1] - 1], MAX_F)
        self.assertGreater(self.freqs[self.hi[-1]], MAX_F)

    def test_no_overlap_and_no_gap(self):
        """柱子必须非空，且 [lo[0], hi[-1]) 之间不留空洞"""
        self.assertTrue(np.all(self.hi > self.lo))
        self.assertTrue(np.all(np.diff(self.lo) >= 0))
        self.assertTrue(np.all(self.hi[:-1] >= self.lo[1:]))
        self.assertTrue(np.all(np.diff(self.hi) >= 0))

    def test_edges_are_log_distributed(self):
        self.assertAlmostEqual(float(self.edges[0]), MIN_F)
        self.assertAlmostEqual(float(self.edges[-1]), MAX_F)
        ratios = self.edges[1:] / self.edges[:-1]
        self.assertLess(float(ratios.max() - ratios.min()), 1e-6)

    def test_band_coverage_is_within_declared_range(self):
        lo_f = self.freqs[self.lo[0]]
        hi_f = self.freqs[self.hi[-1] - 1]
        self.assertGreaterEqual(lo_f, MIN_F)
        self.assertLessEqual(hi_f, MAX_F)

    def test_top_edge_never_exceeds_nyquist(self):
        for sr in (8000, 16000, 22050):
            edges, lo, hi = spectrum_edges(32, sr, fft_size_for(sr))
            self.assertLessEqual(float(edges[-1]), sr / 2.0)
            self.assertTrue(np.all(hi <= np.fft.rfftfreq(fft_size_for(sr),
                                                         1.0 / sr).size))


class TestSpectrumAnalyzer(unittest.TestCase):
    def test_insufficient_data_returns_none(self):
        """有效数据不足一个完整 FFT 窗时宁可不出图，也不给假频谱"""
        an = SpectrumAnalyzer(64)
        self.assertIsNone(an.compute(np.zeros(100, dtype=np.float32), SR, 64))
        self.assertIsNone(an.compute(np.empty(0, dtype=np.float32), SR, 64))
        self.assertIsNotNone(an.compute(np.zeros(2048, dtype=np.float32), SR, 64))

    def test_output_shape_and_range(self):
        bars = bars_of(np.zeros(4096, dtype=np.float32))
        self.assertEqual(bars.shape, (64,))
        self.assertTrue(np.all(bars >= DB_FLOOR))
        self.assertTrue(np.all(bars <= 0.0))

    def test_dc_and_low_freq_stay_at_floor(self):
        """纯直流信号不该点亮第一根柱子。

        旧实现把 idx[0] 强行设成 0，DC bin 会以 +6dB 直接点亮第一根柱子；
        现在第一根柱子从 >=30Hz 的 bin 开始，只剩窗函数旁瓣的残余（<-50dB）。
        """
        bars = bars_of(np.ones(4096, dtype=np.float32))
        self.assertLess(float(bars[0]), -50.0)
        self.assertLessEqual(float(bars.max()), -50.0)

    def test_tone_lands_in_expected_bar(self):
        edges, lo, hi = spectrum_edges(64, SR, fft_size_for(SR))
        freqs = np.fft.rfftfreq(fft_size_for(SR), 1.0 / SR)
        for f in (100.0, 440.0, 1000.0, 4000.0, 12000.0):
            with self.subTest(freq=f):
                x = 0.5 * np.sin(2 * np.pi * f * np.arange(4096) / SR)
                bars = bars_of(x.astype(np.float32))
                k = int(np.argmax(bars))
                # 该柱子测到的必须是这个音调最强 bin 的能量
                spec = np.abs(np.fft.rfft(x[-fft_size_for(SR):] * np.hanning(fft_size_for(SR))))
                peak_bin = int(np.argmax(spec))
                self.assertTrue(lo[k] <= peak_bin < hi[k],
                                "第 %d 根柱子的 bin 区间 [%d,%d) 没包住音调的峰值 bin %d"
                                % (k, lo[k], hi[k], peak_bin))
                expect = int(np.searchsorted(edges, f, side="right") - 1)
                tol = 1 if f >= 1000 else 4      # 低频段一个 bin 就有 23Hz，比对数柱还宽
                self.assertLessEqual(abs(k - expect), tol,
                                     "%.0fHz 应该在第 %d 根柱子附近，实际在第 %d 根"
                                     % (f, expect, k))
                self.assertGreater(float(bars[k]), -20.0)

    def test_tone_above_16khz_is_invisible(self):
        """20kHz 的内容不该出现在任何一根柱子上（旧实现会一路吃到 Nyquist）"""
        x = 0.5 * np.sin(2 * np.pi * 20000 * np.arange(4096) / SR).astype(np.float32)
        bars = bars_of(x)
        self.assertLessEqual(float(bars.max()), DB_FLOOR + 1.0)

    def test_analyzer_handles_sample_rate_change(self):
        an = SpectrumAnalyzer(48)
        for sr in (44100, 48000, 96000, 192000):
            x = np.sin(2 * np.pi * 1000 * np.arange(32768) / sr).astype(np.float32)
            bars = an.compute(x, sr, 48)
            self.assertEqual(bars.shape, (48,))
            self.assertLessEqual(an.top_f, MAX_F)
            self.assertGreater(float(bars.max()), -20.0)

    def test_bar_count_change_rebuilds_cache(self):
        an = SpectrumAnalyzer(32)
        x = np.zeros(8192, dtype=np.float32)
        self.assertEqual(an.compute(x, SR, 32).size, 32)
        self.assertEqual(an.compute(x, SR, 64).size, 64)
        self.assertEqual(an.compute(x, SR, 32).size, 32)

    def test_lower_sample_rate_limits_top_frequency(self):
        edges, lo, hi = spectrum_edges(32, 8000, fft_size_for(8000))
        self.assertLessEqual(float(edges[-1]), 4000.0)


if __name__ == "__main__":
    unittest.main()
