# -*- coding: utf-8 -*-
"""采集层单元测试：设备枚举、打开失败、回调 status、回调异常、切换设备清空缓冲"""
import unittest
from unittest import mock

import numpy as np

from audio_visualizer import audio_engine
from audio_visualizer.audio_engine import AudioEngine, EngineError
from tests.fakes import FakePaModule, device


def make_engine(devices=None, default_index=None, open_raises=None,
                enum_raises=None, ring_size=256):
    mod = FakePaModule(devices=devices if devices is not None else [device(1)],
                       default_index=default_index,
                       open_raises=open_raises, enum_raises=enum_raises)
    return AudioEngine(pa_module=mod, ring_size=ring_size), mod


class TestDeviceEnumeration(unittest.TestCase):
    def test_list_loopbacks_returns_device_info(self):
        eng, _ = make_engine([device(1, "扬声器 A", 48000, 2),
                              device(2, "耳机 B", 96000, 2)])
        devs = eng.list_loopbacks()
        self.assertEqual([d.name for d in devs], ["扬声器 A", "耳机 B"])
        self.assertEqual(devs[1].rate, 96000)
        self.assertIn("耳机 B|96000|2", devs[1].uid)

    def test_uid_separates_same_name_different_format(self):
        a = device(1, "扬声器 A", 48000, 2)
        b = device(2, "扬声器 A", 96000, 2)
        eng, _ = make_engine([a, b])
        uids = {d.uid for d in eng.list_loopbacks()}
        self.assertEqual(len(uids), 2, "同名不同采样率必须是两个设备")

    def test_enumeration_failure_raises(self):
        eng, _ = make_engine(enum_raises=OSError("PortAudio 挂了"))
        with self.assertRaises(EngineError):
            eng.list_loopbacks()

    def test_default_loopback_direct(self):
        eng, _ = make_engine([device(1, "扬声器 A"), device(2, "耳机 B")],
                             default_index=2)
        self.assertEqual(eng.default_loopback().name, "耳机 B")

    def test_default_loopback_by_name_match(self):
        eng, _ = make_engine([
            device(1, "扬声器 A", 48000, 2, loopback=False),
            device(9, "扬声器 A [Loopback]", 48000, 2, loopback=True),
        ], default_index=1)
        d = eng.default_loopback()
        self.assertEqual(d.index, 9)

    def test_default_loopback_missing(self):
        eng, _ = make_engine([device(1, "扬声器 A")], default_index=None)
        self.assertIsNone(eng.default_loopback())


class TestStartStop(unittest.TestCase):
    def test_start_opens_default_device(self):
        eng, mod = make_engine([device(1, "扬声器 A", 48000, 2)], default_index=1)
        self.assertTrue(eng.start(None))
        kw = mod.instances[0].opened[0]
        self.assertEqual(kw["rate"], 48000)
        self.assertEqual(kw["channels"], 2)
        self.assertEqual(kw["input_device_index"], 1)
        self.assertTrue(eng.is_healthy)

    def test_start_without_any_device(self):
        eng, _ = make_engine([], default_index=None)
        self.assertFalse(eng.start(None))
        self.assertIn("未找到可用的音频输出设备", eng.error)
        self.assertFalse(eng.is_healthy)

    def test_open_failure_is_reported(self):
        eng, _ = make_engine([device(1)], default_index=1,
                             open_raises=OSError("设备被占用"))
        self.assertFalse(eng.start(None))
        self.assertIn("设备被占用", eng.error)
        self.assertFalse(eng.is_healthy)

    def test_start_clears_ring(self):
        """切换设备后不能读到上一台设备的残留音频"""
        eng, _ = make_engine([device(1), device(2)], default_index=1)
        eng.start(None)
        eng.ring.write(np.ones(100, dtype=np.float32))
        self.assertEqual(eng.ring.valid, 100)
        eng.start(None)
        self.assertEqual(eng.ring.valid, 0)
        self.assertEqual(eng.read_last(256).size, 0)

    def test_unhealthy_when_stream_inactive(self):
        eng, mod = make_engine([device(1)], default_index=1)
        eng.start(None)
        mod.instances[0].last_stream.active = False
        self.assertFalse(eng.is_healthy)

    def test_stop_and_close_release_stream(self):
        eng, mod = make_engine([device(1)], default_index=1)
        eng.start(None)
        stream = mod.instances[0].last_stream
        eng.close()
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)
        self.assertTrue(mod.instances[0].terminated)
        self.assertFalse(eng.is_healthy)


class TestCallback(unittest.TestCase):
    def setUp(self):
        self.eng, _ = make_engine([device(1, "扬声器 A", 48000, 2)],
                                  default_index=1, ring_size=64)
        self.eng.start(None)

    def feed(self, samples, channels=2, status=0):
        """把 float32 数据喂给回调，返回回调的返回值"""
        data = np.asarray(samples, dtype=np.float32).tobytes()
        return self.eng._callback(data, len(samples) // channels, {}, status)

    def test_mono_write_and_read_order(self):
        s = np.arange(10, dtype=np.float32)
        ret = self.feed(np.column_stack([s, s]).ravel())
        self.assertEqual(ret, (None, self.eng._pa.paContinue))
        np.testing.assert_allclose(self.eng.read_last(10), s)

    def test_read_never_exceeds_valid(self):
        self.feed(np.ones(8, dtype=np.float32))
        self.assertEqual(self.eng.read_last(4096).size, 4)

    def test_anti_phase_stereo_not_cancelled(self):
        s = np.sin(np.linspace(0, 20, 32)).astype(np.float32)
        self.feed(np.column_stack([s, -s]).ravel())
        self.assertGreater(float(np.abs(self.eng.read_last(32)).max()), 0.5)

    def test_status_flags_are_recorded(self):
        pa = self.eng._pa
        self.feed(np.zeros(4, dtype=np.float32), status=pa.paInputOverflow)
        self.feed(np.zeros(4, dtype=np.float32), status=pa.paInputOverflow)
        self.feed(np.zeros(4, dtype=np.float32), status=pa.paOutputUnderflow)
        self.assertEqual(self.eng.overflow_count, 2)
        self.assertEqual(self.eng.underflow_count, 1)
        self.assertIn("溢出 2", self.eng.status_snapshot())
        self.assertIn("欠载 1", self.eng.status_snapshot())

    def test_status_counters_reset_on_restart(self):
        self.feed(np.zeros(4, dtype=np.float32),
                  status=self.eng._pa.paInputOverflow)
        self.eng.start(None)
        self.assertEqual(self.eng.overflow_count, 0)

    def test_callback_exception_aborts_and_is_reported(self):
        with mock.patch.object(audio_engine, "stereo_to_mono",
                               side_effect=RuntimeError("boom")), \
                mock.patch.object(self.eng, "log"):
            ret = self.feed(np.zeros(4, dtype=np.float32))
        self.assertEqual(ret, (None, self.eng._pa.paAbort))
        self.assertFalse(self.eng.alive)
        self.assertIn("boom", self.eng.error)

    def test_mono_device_passthrough(self):
        eng, _ = make_engine([device(1, "麦克风", 44100, 1)], default_index=1,
                             ring_size=64)
        eng.start(None)
        self.assertEqual(eng.channels, 1)
        s = np.arange(8, dtype=np.float32)
        eng._callback(s.tobytes(), 8, {}, 0)
        np.testing.assert_allclose(eng.read_last(8), s)

    def test_multichannel_loopback(self):
        eng, _ = make_engine([device(1, "环绕声", 48000, 6)], default_index=1,
                             ring_size=64)
        eng.start(None)
        self.assertEqual(eng.channels, 6)
        frames = np.zeros((4, 6), dtype=np.float32)
        frames[:, 4] = 0.5                    # 只有第 5 个声道有声音
        eng._callback(frames.ravel().tobytes(), 4, {}, 0)
        np.testing.assert_allclose(eng.read_last(4), 0.5)

    def test_full_buffer_write(self):
        """一次写入量正好填满缓冲区时也要按时间顺序读出来"""
        s = np.arange(64, dtype=np.float32)
        self.feed(np.column_stack([s, s]).ravel())
        self.assertEqual(self.eng.ring.valid, 64)
        np.testing.assert_allclose(self.eng.read_last(64), s)


if __name__ == "__main__":
    unittest.main()
