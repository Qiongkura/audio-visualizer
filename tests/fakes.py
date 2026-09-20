# -*- coding: utf-8 -*-
"""测试替身：假 PortAudio 模块 / 假引擎 / 假画布。

有了这些，环形缓冲区、FFT 边界、设备状态机、渲染缓冲都能在没有声卡、
没有窗口的环境里被完整覆盖（CI 友好）。
"""
import numpy as np

from audio_visualizer.audio_engine import DeviceInfo, EngineError


def device(index=1, name="扬声器 (Fake)", rate=48000, channels=2, loopback=True):
    return {"index": index, "name": name, "defaultSampleRate": float(rate),
            "maxInputChannels": channels, "isLoopbackDevice": loopback}


class FakeStream:
    def __init__(self, active=True):
        self.active = bool(active)
        self.stopped = False
        self.closed = False

    def stop_stream(self):
        self.stopped = True
        self.active = False

    def close(self):
        self.closed = True

    def is_active(self):
        return self.active


class FakePyAudio:
    def __init__(self, module):
        self._m = module
        self.devices = list(module.devices)
        self.default_index = module.default_index
        self.open_raises = module.open_raises
        self.opened = []
        self.terminated = False
        self.last_stream = None

    def get_loopback_device_info_generator(self):
        if self._m.enum_raises is not None:
            raise self._m.enum_raises
        # 真机上这个生成器只会吐出环回设备
        return iter([dict(d) for d in self.devices
                     if d.get("isLoopbackDevice", True)])

    def get_host_api_info_by_type(self, _t):
        if self.default_index is None:
            raise RuntimeError("no default device")
        return {"defaultOutputDevice": self.default_index}

    def get_device_info_by_index(self, index):
        for d in self.devices:
            if d["index"] == index:
                return dict(d)
        raise RuntimeError("device %s not found" % index)

    def open(self, **kw):
        self.opened.append(kw)
        if self.open_raises is not None:
            raise self.open_raises
        self.last_stream = FakeStream()
        return self.last_stream

    def terminate(self):
        self.terminated = True


class FakePaModule:
    """假装成 pyaudiowpatch 模块"""
    paFloat32 = 1
    paContinue = 0
    paComplete = 1
    paAbort = 2
    paInputOverflow = 1
    paOutputUnderflow = 2
    paWASAPI = 13

    def __init__(self, devices=None, default_index=None, open_raises=None,
                 enum_raises=None):
        self.devices = list(devices or [])
        self.default_index = default_index
        self.open_raises = open_raises
        self.enum_raises = enum_raises
        self.instances = []

    def PyAudio(self):
        pa = FakePyAudio(self)
        self.instances.append(pa)
        return pa


class StubEngine:
    """给 DeviceSupervisor 用的最小引擎替身"""

    def __init__(self, healthy=True, device=None, error="",
                 devices=None, default=None, enum_raises=None):
        self.healthy = healthy
        self.device = device
        self.error = error
        self.devices = list(devices or [])
        self.default = default
        self.enum_raises = enum_raises
        self.enum_calls = 0

    @property
    def is_healthy(self):
        return self.healthy

    def list_loopbacks(self):
        self.enum_calls += 1
        if self.enum_raises is not None:
            raise EngineError(self.enum_raises)
        return list(self.devices)

    def default_loopback(self):
        return self.default


def dev(index=1, name="Speakers", rate=48000, channels=2):
    return DeviceInfo(index=index, name=name, rate=rate, channels=channels)


def tone(freq, sample_rate=48000, n=4096, amp=1.0):
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeCanvas:
    """够用的 Tk Canvas 替身，只记录调用"""

    def __init__(self, w=800, h=400):
        self.w, self.h = w, h
        self.images = []
        self.texts = []
        self.conf = {}
        self.deleted = 0

    def delete(self, *a):
        self.deleted += 1
        self.images.clear()
        self.texts.clear()

    def create_image(self, *a, **k):
        self.images.append(k)
        return 7

    def create_text(self, *a, **k):
        self.texts.append((a, k))
        return len(self.texts)

    def itemconfigure(self, *a, **k):
        self.conf.update(k)

    def winfo_width(self):
        return self.w

    def winfo_height(self):
        return self.h
