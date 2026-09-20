# -*- coding: utf-8 -*-
"""DSP 工作线程单元测试：只保留最新帧、数据不足不出图、异常不拖垮线程"""
import time
import unittest

import numpy as np

from audio_visualizer.dsp import DspWorker, fft_size_for


def reader(data):
    data = np.asarray(data, dtype=np.float32)

    def _read(n):
        n = int(n)
        return data[-n:].copy() if n < data.size else data.copy()
    return _read


def make_worker(data, sr=48000, cols=200, n_bars=32, interval=0.005, read=None):
    return DspWorker(read_samples=read or reader(data),
                     sample_rate=lambda: sr,
                     wave_cols=lambda: cols,
                     n_bars=n_bars, interval=interval)


class TestComputeOnce(unittest.TestCase):
    def test_produces_bars_and_wave(self):
        t = np.arange(16384) / 48000.0
        w = make_worker(0.5 * np.sin(2 * np.pi * 1000 * t))
        f = w.compute_once()
        self.assertEqual(f.bars.shape, (32,))
        self.assertEqual(f.wave.shape, (200,))
        self.assertEqual(f.sample_rate, 48000)
        self.assertEqual(f.valid, 4320)      # 一次只取 max(FFT 窗, 波形窗)

    def test_bars_skipped_when_not_enough_samples(self):
        """FFT 窗都没填满时不出频谱，避免启动瞬间的假数据"""
        w = make_worker(np.ones(300, dtype=np.float32))
        f = w.compute_once()
        self.assertIsNone(f.bars)
        self.assertIsNotNone(f.wave)          # 波形门槛更低，可以先画

    def test_nothing_at_all_for_tiny_buffer(self):
        w = make_worker(np.ones(64, dtype=np.float32))
        f = w.compute_once()
        self.assertIsNone(f.bars)
        self.assertIsNone(f.wave)

    def test_zero_sample_rate_falls_back(self):
        w = make_worker(np.zeros(8192, dtype=np.float32), sr=0)
        f = w.compute_once()
        self.assertEqual(f.sample_rate, 48000)
        self.assertEqual(f.bars.size, 32)

    def test_n_bars_change_is_picked_up(self):
        w = make_worker(np.zeros(8192, dtype=np.float32), n_bars=32)
        self.assertEqual(w.compute_once().bars.size, 32)
        w.n_bars = 64
        self.assertEqual(w.compute_once().bars.size, 64)

    def test_fft_size_matches_sample_rate(self):
        for sr in (44100, 48000, 96000):
            w = make_worker(np.zeros(32768, dtype=np.float32), sr=sr)
            f = w.compute_once()
            self.assertEqual(f.bars.size, 32)
            self.assertEqual(f.sample_rate, sr)
            self.assertLessEqual(fft_size_for(sr), 16384)


class TestThreadLifecycle(unittest.TestCase):
    def test_latest_is_none_before_start(self):
        w = make_worker(np.zeros(8192, dtype=np.float32))
        self.assertIsNone(w.latest())

    def test_runs_and_stops(self):
        t = np.arange(32768) / 48000.0
        w = make_worker(0.3 * np.sin(2 * np.pi * 440 * t))
        w.start()
        deadline = time.time() + 2.0
        while w.latest() is None and time.time() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(w.latest(), "DSP 线程没有产出任何帧")
        w.stop()
        self.assertFalse(w.is_alive())

    def test_keeps_only_latest_frame(self):
        w = make_worker(np.zeros(8192, dtype=np.float32))
        w.start()
        time.sleep(0.1)
        f1 = w.latest()
        time.sleep(0.1)
        f2 = w.latest()
        w.stop()
        self.assertIsNotNone(f2)
        self.assertGreater(f2.seq, f1.seq)
        self.assertIs(w.latest(), f2)         # 只留最新一帧，没有积压

    def test_read_error_does_not_kill_thread(self):
        def boom(_n):
            raise RuntimeError("环形缓冲区炸了")
        w = make_worker(None, read=boom, interval=0.005)
        w.start()
        time.sleep(0.1)
        self.assertTrue(w.is_alive(), "DSP 线程不应该因为异常退出")
        self.assertIsNone(w.latest())
        w.stop()
        self.assertFalse(w.is_alive())

    def test_reset_drops_pending_frame(self):
        w = make_worker(np.zeros(8192, dtype=np.float32))
        w.start()
        time.sleep(0.05)
        self.assertIsNotNone(w.latest())
        w.stop()
        w.reset()
        self.assertIsNone(w.latest())


if __name__ == "__main__":
    unittest.main()
