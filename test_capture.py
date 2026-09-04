# -*- coding: utf-8 -*-
"""
环回采集自检脚本（无需 GUI）：
1. 列出所有可采集的输出设备
2. 用系统默认输出播放 440Hz 测试音
3. 通过 WASAPI 环回采集 3 秒
4. 校验采集到的音量与主频是否正确

运行: venv\\Scripts\\python.exe test_capture.py
"""
import os
import sys
import tempfile
import time
import wave

import numpy as np
import pyaudiowpatch as pyaudio

DUR = 3.0
FREQ = 440.0


def make_tone_wav(path):
    """生成 1 秒 440Hz 测试音 wav"""
    sr = 48000
    t = np.arange(int(sr * 1.0)) / sr
    x = 0.35 * np.sin(2 * np.pi * FREQ * t)
    n = len(x)
    env = np.ones(n)
    env[:2000] = np.linspace(0, 1, 2000)
    env[-2000:] = np.linspace(1, 0, 2000)
    x = (x * env * 32767).astype(np.int16)
    stereo = np.column_stack([x, x])
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())


def main():
    p = pyaudio.PyAudio()
    loopbacks = list(p.get_loopback_device_info_generator())
    print(f"发现 {len(loopbacks)} 个可采集的输出设备:")
    for d in loopbacks:
        print(f"  [{d['index']}] {d['name']}  @{int(d['defaultSampleRate'])}Hz  "
              f"{d['maxInputChannels']}ch")
    if not loopbacks:
        print("FAIL: 没有任何音频输出设备，无法环回采集")
        return 1

    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    except Exception as e:
        print(f"FAIL: 无法获取默认输出设备: {e}")
        return 1

    target = spk if spk.get("isLoopbackDevice") else None
    if target is None:
        for lb in loopbacks:
            if spk["name"] in lb["name"]:
                target = lb
                break
    if target is None:
        print(f"FAIL: 找不到默认输出 '{spk['name']}' 对应的环回设备")
        return 1
    print(f"目标设备: {target['name']}")

    # 播放测试音（系统默认输出）
    tmp = os.path.join(tempfile.gettempdir(), "av_test_tone.wav")
    make_tone_wav(tmp)
    import winsound
    winsound.PlaySound(tmp, winsound.SND_ASYNC | winsound.SND_LOOP)

    ch = max(1, int(target["maxInputChannels"]))
    rate = int(target["defaultSampleRate"])
    frames = []

    def cb(in_data, frame_count, time_info, status):
        frames.append(np.frombuffer(in_data, np.float32).copy())
        return (None, pyaudio.paContinue)

    try:
        stream = p.open(format=pyaudio.paFloat32, channels=ch, rate=rate,
                        frames_per_buffer=512, input=True,
                        input_device_index=target["index"], stream_callback=cb)
    except Exception as e:
        winsound.PlaySound(None, winsound.SND_PURGE)
        print(f"FAIL: 打开环回流失败: {e}")
        return 1

    time.sleep(DUR)
    stream.stop_stream()
    stream.close()
    winsound.PlaySound(None, winsound.SND_PURGE)

    x = np.concatenate(frames)
    if x.size % ch:
        x = x[: x.size - x.size % ch]
    x = x.reshape(-1, ch).mean(axis=1)

    rms = float(np.sqrt((x ** 2).mean()))
    peak = float(np.abs(x).max())
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win))
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    dom = float(f[int(np.argmax(spec))])

    print(f"采集 {len(x)} 采样 ({len(x) / rate:.2f}s @ {rate}Hz, {ch}ch)")
    print(f"RMS={rms:.4f}  Peak={peak:.4f}  主频={dom:.1f}Hz (期望 ~{FREQ:.0f}Hz)")
    ok = rms > 0.01 and abs(dom - FREQ) < 20
    print("PASS" if ok else "FAIL: 采集到的信号异常（静音或频率不符）")
    p.terminate()
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
