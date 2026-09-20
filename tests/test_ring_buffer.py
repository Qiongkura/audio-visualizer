# -*- coding: utf-8 -*-
"""环形缓冲区单元测试（不需要音频设备）"""
import unittest

import numpy as np

from audio_visualizer.dsp import RingBuffer, stereo_to_mono


class TestRingBuffer(unittest.TestCase):
    def test_empty_buffer_returns_nothing(self):
        rb = RingBuffer(16)
        self.assertEqual(rb.valid, 0)
        self.assertEqual(rb.read_last(8).size, 0)

    def test_returns_only_valid_samples(self):
        """启动瞬间不能把初始化零值当成有效音频"""
        rb = RingBuffer(1024)
        rb.write(np.ones(10, dtype=np.float32))
        self.assertEqual(rb.valid, 10)
        out = rb.read_last(1024)
        self.assertEqual(out.size, 10)
        self.assertTrue(np.all(out == 1.0))

    def test_chronological_order(self):
        rb = RingBuffer(64)
        rb.write(np.arange(20, dtype=np.float32))
        out = rb.read_last(20)
        np.testing.assert_array_equal(out, np.arange(20))

    def test_wrap_around_keeps_latest(self):
        rb = RingBuffer(32)
        rb.write(np.arange(100, dtype=np.float32))
        self.assertEqual(rb.valid, 32)
        out = rb.read_last(32)
        np.testing.assert_array_equal(out, np.arange(68, 100))

    def test_partial_read_after_wrap(self):
        rb = RingBuffer(32)
        rb.write(np.arange(100, dtype=np.float32))
        np.testing.assert_array_equal(rb.read_last(5), np.arange(95, 100))
        np.testing.assert_array_equal(rb.read_last(40), np.arange(68, 100))

    def test_chunked_writes_are_contiguous(self):
        rb = RingBuffer(16)
        for i in range(0, 40, 4):
            rb.write(np.arange(i, i + 4, dtype=np.float32))
        np.testing.assert_array_equal(rb.read_last(16), np.arange(24, 40))

    def test_write_larger_than_capacity(self):
        rb = RingBuffer(8)
        rb.write(np.arange(50, dtype=np.float32))
        self.assertEqual(rb.valid, 8)
        np.testing.assert_array_equal(rb.read_last(8), np.arange(42, 50))

    def test_clear_resets_valid(self):
        rb = RingBuffer(64)
        rb.write(np.arange(64, dtype=np.float32))
        rb.clear()
        self.assertEqual(rb.valid, 0)
        self.assertEqual(rb.wpos, 0)
        self.assertEqual(rb.read_last(64).size, 0)

    def test_read_returns_copy(self):
        rb = RingBuffer(64)
        rb.write(np.arange(32, dtype=np.float32))
        out = rb.read_last(32)
        out[:] = -1
        np.testing.assert_array_equal(rb.read_last(32), np.arange(32))

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            RingBuffer(0)


class TestStereoToMono(unittest.TestCase):
    def test_single_channel_passthrough(self):
        x = np.arange(8, dtype=np.float32).reshape(-1, 1)
        np.testing.assert_array_equal(stereo_to_mono(x), np.arange(8))

    def test_in_phase_keeps_amplitude(self):
        s = np.sin(np.linspace(0, 6, 64)).astype(np.float32)
        frames = np.column_stack([s, s])
        out = stereo_to_mono(frames)
        np.testing.assert_allclose(out, s, atol=1e-6)

    def test_anti_phase_does_not_cancel(self):
        """左右反相时简单平均会接近静音，这里必须保住音量"""
        s = np.sin(np.linspace(0, 6, 64)).astype(np.float32)
        frames = np.column_stack([s, -s])
        out = stereo_to_mono(frames)
        self.assertGreater(float(np.abs(out).max()), 0.9)
        avg = frames.mean(axis=1)
        self.assertLess(float(np.abs(avg).max()), 0.01)   # 对照：平均确实会抵消

    def test_silent_channel_falls_back_to_loud_one(self):
        s = np.sin(np.linspace(0, 6, 64)).astype(np.float32)
        frames = np.column_stack([np.zeros_like(s), s])
        np.testing.assert_allclose(stereo_to_mono(frames), s, atol=1e-6)

    def test_multichannel(self):
        s = np.linspace(-1, 1, 32).astype(np.float32)
        frames = np.column_stack([np.zeros_like(s), s, np.zeros_like(s),
                                  np.zeros_like(s)])
        np.testing.assert_allclose(stereo_to_mono(frames), s, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
