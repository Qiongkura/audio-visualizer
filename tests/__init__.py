# -*- coding: utf-8 -*-
"""单元测试包（全部不需要真实声卡）

运行: python -m unittest discover -s tests -t . -v
"""
import logging

# 测试期间不打印日志，保持输出干净（日志本身另有测试覆盖）
_log = logging.getLogger("audio_visualizer")
_log.addHandler(logging.NullHandler())
_log.propagate = False
_log.setLevel(logging.CRITICAL)
