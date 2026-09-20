# -*- coding: utf-8 -*-
"""音频采集层：WASAPI 环回采集 + 设备异常恢复状态机。

设计要点：
- ``DeviceInfo`` 有稳定的内部 ID（名称 + 采样率 + 声道数），不再只靠显示名比较；
- 回调里的 PortAudio ``status`` 会被记录（overflow / underflow），不再被忽略；
- 手动指定的设备消失时明确报“设备已断开”，不静默退回默认设备；
- ``DeviceSupervisor`` 是不依赖 Tk 的纯逻辑状态机，可被单元测试覆盖。
"""
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import (CALLBACK_FRAMES, ENUM_BACKOFF, RETRY_DELAY, RING_SIZE,
                     STATUS_LOG_THROTTLE)
from .dsp import RingBuffer, stereo_to_mono

log = logging.getLogger(__name__)

try:                                  # 允许在没有声卡/没装 pyaudiowpatch 的环境 import
    import pyaudiowpatch as _pyaudio
except Exception:                     # pragma: no cover - 取决于运行环境
    _pyaudio = None


class EngineError(RuntimeError):
    """设备枚举 / 打开失败"""


@dataclass(frozen=True)
class DeviceInfo:
    """一个可采集的环回设备。

    ``uid`` 是稳定的内部标识：设备名 + 采样率 + 声道数。
    单看名称无法区分同一设备在不同采样率下的两个条目，
    也无法发现“设备换了但名字没变”的情况。
    """
    index: int
    name: str
    rate: int
    channels: int
    is_loopback: bool = True

    @property
    def uid(self):
        return "%s|%d|%d" % (self.name, self.rate, self.channels)

    @property
    def label(self):
        return "%s   @%dHz · %dch" % (self.name, self.rate, self.channels)

    @classmethod
    def from_pa(cls, d):
        return cls(
            index=int(d["index"]),
            name=str(d["name"]),
            rate=int(round(float(d.get("defaultSampleRate") or 48000))),
            channels=max(1, int(d.get("maxInputChannels") or 2)),
            is_loopback=bool(d.get("isLoopbackDevice", True)),
        )


@dataclass
class Decision:
    """看门狗的一次决策"""
    action: str                     # none | reconnect | switch | lost | error
    device: object = None           # DeviceInfo | None
    message: str = ""


class AudioEngine:
    """WASAPI 环回采集：把系统正在播放的声音写进环形缓冲区"""

    def __init__(self, pa_module=None, ring_size=RING_SIZE, logger=None):
        self._pa = _pyaudio if pa_module is None else pa_module
        if self._pa is None:
            raise EngineError("未安装 pyaudiowpatch，无法进行音频采集")
        self.log = logger or log
        self.pa = self._pa.PyAudio()
        self.stream = None
        self.device = None                  # DeviceInfo
        self.channels = 0
        self.sample_rate = 0
        self.ring = RingBuffer(ring_size)
        self.lock = threading.Lock()
        self.alive = False
        self.error = ""
        self.overflow_count = 0
        self.underflow_count = 0
        self.status_last = 0
        self._status_log_at = 0.0
        self._cb_buf = np.empty(0, dtype=np.float32)

    # ---------------- 设备枚举 ----------------
    def list_loopbacks(self):
        """列出所有可采集的输出设备；失败抛 EngineError（不再静默返回空表）"""
        try:
            out = [DeviceInfo.from_pa(d)
                   for d in self.pa.get_loopback_device_info_generator()]
        except Exception as e:
            raise EngineError("设备枚举失败：%s: %s" % (type(e).__name__, e)) from e
        return out

    def default_loopback(self):
        """系统默认输出设备对应的环回设备，找不到返回 None"""
        try:
            wasapi = self.pa.get_host_api_info_by_type(self._pa.paWASAPI)
            spk = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        except Exception as e:
            self.log.warning("获取默认输出设备失败: %s: %s", type(e).__name__, e)
            return None
        try:
            if spk.get("isLoopbackDevice"):
                return DeviceInfo.from_pa(spk)
            for lb in self.pa.get_loopback_device_info_generator():
                if spk["name"] in lb["name"]:
                    return DeviceInfo.from_pa(lb)
        except Exception as e:
            self.log.warning("匹配默认输出设备失败: %s: %s", type(e).__name__, e)
        return None

    # ---------------- 采集控制 ----------------
    def start(self, device=None):
        """打开采集流。device 为 None 表示跟随系统默认输出设备。"""
        self.stop()
        self.error = ""
        try:
            if device is None:
                device = self.default_loopback()
            if device is None:
                self.error = "未找到可用的音频输出设备"
                self.alive = False
                return False
            if not isinstance(device, DeviceInfo):
                device = DeviceInfo.from_pa(device)

            self.device = device
            self.channels = max(1, int(device.channels))
            self.sample_rate = int(device.rate or 48000)
            self.ring.clear()               # 清掉上一次设备的残留数据
            self.overflow_count = 0
            self.underflow_count = 0
            self.status_last = 0
            self.stream = self.pa.open(
                format=self._pa.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=CALLBACK_FRAMES,
                input=True,
                input_device_index=device.index,
                stream_callback=self._callback,
            )
            self.alive = True
            self.log.info("开始采集: %s @%dHz %dch", device.name,
                          self.sample_rate, self.channels)
            return True
        except Exception as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            self.alive = False
            self.log.error("打开采集流失败 (%s): %s", getattr(device, "name", device), e)
            return False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                self.log.warning("关闭采集流异常: %s: %s", type(e).__name__, e)
            self.stream = None
        self.alive = False

    def close(self):
        self.stop()
        try:
            self.pa.terminate()
        except Exception as e:
            self.log.warning("释放 PortAudio 异常: %s: %s", type(e).__name__, e)

    @property
    def is_healthy(self):
        if not self.alive or self.stream is None or self.error:
            return False
        try:
            return bool(self.stream.is_active())
        except Exception:
            return False

    def status_snapshot(self):
        """给状态栏用的一句话；没有异常事件时返回空串"""
        parts = []
        if self.overflow_count:
            parts.append("溢出 %d" % self.overflow_count)
        if self.underflow_count:
            parts.append("欠载 %d" % self.underflow_count)
        return " · ".join(parts)

    # ---------------- 回调 ----------------
    def _record_status(self, status):
        """记录 PortAudio 状态位，并节流打日志（原来这里直接忽略了）"""
        status = int(status)
        self.status_last = status
        pa_overflow = getattr(self._pa, "paInputOverflow", 0x1)
        pa_underflow = getattr(self._pa, "paOutputUnderflow", 0x2)
        if status & pa_overflow:
            self.overflow_count += 1
        if status & pa_underflow:
            self.underflow_count += 1
        now = time.time()
        if now - self._status_log_at >= STATUS_LOG_THROTTLE:
            self._status_log_at = now
            self.log.warning("PortAudio 状态异常 status=%d（溢出 %d 次 / 欠载 %d 次）",
                             status, self.overflow_count, self.underflow_count)

    def _callback(self, in_data, frame_count, time_info, status):
        if status:
            self._record_status(status)
        try:
            data = np.frombuffer(in_data, dtype=np.float32)
            if self.channels > 1 and data.size % self.channels == 0:
                frames = data.reshape(-1, self.channels)
                n = frames.shape[0]
                if self._cb_buf.size != n:
                    self._cb_buf = np.empty(n, dtype=np.float32)
                np.copyto(self._cb_buf, stereo_to_mono(frames))
                mono = self._cb_buf
            else:
                mono = data
            with self.lock:
                self.ring.write(mono)
            return (None, self._pa.paContinue)
        except Exception as e:
            self.error = "音频回调异常：%s: %s" % (type(e).__name__, e)
            self.alive = False
            self.log.exception("音频回调异常，已中止采集")
            return (None, self._pa.paAbort)

    def read_last(self, n):
        with self.lock:
            return self.ring.read_last(n)


# ============================================================ 设备状态机
class DeviceSupervisor:
    """决定“现在该不该重连、切设备、还是报断开”。

    纯逻辑，不碰 Tk 也不碰音频库，方便单测：
    - 采集流不健康时按 ``retry_delay`` 重连；
    - 手动指定的设备消失 → 报 ``lost``（只报一次），绝不偷偷切到别的设备；
    - 跟随默认设备时，默认输出设备换了（含采样率/声道变化）→ ``switch``。
    """

    def __init__(self, engine, retry_delay=RETRY_DELAY, enum_backoff=ENUM_BACKOFF):
        self.engine = engine
        self.retry_delay = float(retry_delay)
        self.enum_backoff = float(enum_backoff)
        self.retry_at = 0.0
        self.lost_uid = None
        self._enum_fail_at = 0.0
        self.last_error = ""

    def note_started(self, ok, now=None):
        """采集尝试结束后由调用方回报结果"""
        now = time.time() if now is None else now
        if ok:
            self.retry_at = 0.0
            self.lost_uid = None
            self.last_error = ""
        else:
            self.retry_at = now + self.retry_delay
            self.last_error = self.engine.error

    def tick(self, paused=False, follow_default=True, manual_device=None, now=None):
        now = time.time() if now is None else now
        if paused:
            return Decision("none")

        if not self.engine.is_healthy:
            if not follow_default and manual_device is not None:
                try:
                    devices = self.engine.list_loopbacks()
                except EngineError as e:
                    if now - self._enum_fail_at >= self.enum_backoff:
                        self._enum_fail_at = now
                        return Decision("error", message=str(e))
                    return Decision("none")
                if manual_device.uid not in {d.uid for d in devices}:
                    if self.lost_uid != manual_device.uid:
                        self.lost_uid = manual_device.uid
                        return Decision("lost",
                                        message="设备已断开：%s" % manual_device.name)
                    return Decision("none")     # 已提示过，等用户处理
                self.lost_uid = None
            if now < self.retry_at:
                return Decision("none")
            self.retry_at = now + self.retry_delay
            return Decision("reconnect",
                            device=None if follow_default else manual_device)

        self.lost_uid = None
        if follow_default:
            d = self.engine.default_loopback()
            if d is not None and (self.engine.device is None
                                  or d.uid != self.engine.device.uid):
                return Decision("switch", device=d)
        return Decision("none")
