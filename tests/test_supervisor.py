# -*- coding: utf-8 -*-
"""设备异常恢复状态机单元测试（纯逻辑，不碰硬件）"""
import unittest

from audio_visualizer.audio_engine import DeviceSupervisor
from tests.fakes import StubEngine, dev


class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.speaker = dev(1, "扬声器 A", 48000, 2)
        self.headset = dev(2, "耳机 B", 48000, 2)

    def sup(self, **kw):
        self.engine = StubEngine(**kw)
        return DeviceSupervisor(self.engine, retry_delay=1.5, enum_backoff=3.0)

    def test_paused_does_nothing(self):
        sup = self.sup(healthy=False)
        d = sup.tick(paused=True, follow_default=True, now=100.0)
        self.assertEqual(d.action, "none")

    def test_healthy_follow_default_same_device(self):
        sup = self.sup(healthy=True, device=self.speaker, default=self.speaker)
        self.assertEqual(sup.tick(follow_default=True, now=100.0).action, "none")

    def test_switches_when_default_output_changes(self):
        sup = self.sup(healthy=True, device=self.speaker, default=self.headset)
        d = sup.tick(follow_default=True, now=100.0)
        self.assertEqual(d.action, "switch")
        self.assertEqual(d.device.uid, self.headset.uid)

    def test_detects_same_name_different_rate(self):
        """设备名没变但采样率变了，也要跟着切"""
        other = dev(1, "扬声器 A", 96000, 2)
        sup = self.sup(healthy=True, device=self.speaker, default=other)
        self.assertEqual(sup.tick(follow_default=True, now=100.0).action, "switch")

    def test_reconnect_when_following_default(self):
        sup = self.sup(healthy=False, device=self.speaker, error="设备被占用")
        d = sup.tick(follow_default=True, now=100.0)
        self.assertEqual(d.action, "reconnect")
        self.assertIsNone(d.device)             # None = 跟随默认输出设备

    def test_reconnect_respects_retry_delay(self):
        sup = self.sup(healthy=False, error="x")
        self.assertEqual(sup.tick(follow_default=True, now=100.0).action, "reconnect")
        self.assertEqual(sup.tick(follow_default=True, now=100.5).action, "none")
        self.assertEqual(sup.tick(follow_default=True, now=101.6).action, "reconnect")

    def test_manual_device_reconnects_when_still_present(self):
        sup = self.sup(healthy=False, devices=[self.speaker], error="x")
        d = sup.tick(follow_default=False, manual_device=self.speaker, now=100.0)
        self.assertEqual(d.action, "reconnect")
        self.assertEqual(d.device.uid, self.speaker.uid)

    def test_manual_device_lost_is_reported_not_silently_switched(self):
        """手动选的设备拔了，必须明确报断开，不能偷偷切到别的设备"""
        sup = self.sup(healthy=False, devices=[self.headset], error="x")
        d = sup.tick(follow_default=False, manual_device=self.speaker, now=100.0)
        self.assertEqual(d.action, "lost")
        self.assertIn("扬声器 A", d.message)
        self.assertIsNone(d.device)

    def test_lost_is_reported_only_once(self):
        sup = self.sup(healthy=False, devices=[], error="x")
        sup.tick(follow_default=False, manual_device=self.speaker, now=100.0)
        self.assertEqual(sup.tick(follow_default=False,
                                  manual_device=self.speaker, now=101.0).action, "none")
        self.assertEqual(sup.tick(follow_default=False,
                                  manual_device=self.speaker, now=200.0).action, "none")

    def test_lost_device_reappearing_triggers_reconnect(self):
        sup = self.sup(healthy=False, devices=[], error="x")
        sup.tick(follow_default=False, manual_device=self.speaker, now=100.0)
        sup.engine.devices = [self.speaker]
        sup.engine.healthy = False
        d = sup.tick(follow_default=False, manual_device=self.speaker, now=102.0)
        self.assertEqual(d.action, "reconnect")
        self.assertIsNone(sup.lost_uid)

    def test_enumeration_failure_is_reported_and_throttled(self):
        sup = self.sup(healthy=False, enum_raises="PortAudio 枚举失败", error="x")
        d = sup.tick(follow_default=False, manual_device=self.speaker, now=100.0)
        self.assertEqual(d.action, "error")
        self.assertIn("枚举失败", d.message)
        # 退避窗口内不再重复报错
        self.assertEqual(sup.tick(follow_default=False,
                                  manual_device=self.speaker, now=101.0).action, "none")
        self.assertEqual(sup.tick(follow_default=False,
                                  manual_device=self.speaker, now=104.0).action, "error")

    def test_note_started_resets_retry_timer(self):
        sup = self.sup(healthy=False, error="x")
        sup.tick(follow_default=True, now=100.0)
        sup.note_started(True, now=100.0)
        self.assertEqual(sup.retry_at, 0.0)
        self.assertEqual(sup.lost_uid, None)

    def test_note_started_failure_schedules_retry(self):
        sup = self.sup(healthy=False, error="x")
        sup.note_started(False, now=100.0)
        self.assertAlmostEqual(sup.retry_at, 101.5)
        self.assertEqual(sup.tick(follow_default=True, now=101.0).action, "none")

    def test_recovered_engine_stops_reconnecting(self):
        sup = self.sup(healthy=False, error="x")
        sup.tick(follow_default=True, now=100.0)
        sup.engine.healthy = True
        sup.engine.error = ""
        sup.engine.device = self.speaker
        sup.engine.default = self.speaker
        self.assertEqual(sup.tick(follow_default=True, now=102.0).action, "none")


if __name__ == "__main__":
    unittest.main()
