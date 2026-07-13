"""
Tests package marker for audio-helper.

Empty on purpose — pytest discovers test modules by filename convention,
this file simply makes ``tests`` an importable package so relative
imports and shared fixtures work if the suite grows.

Usage Example
-------------
>>> #   pytest              # unit tests only
>>> #   pytest -m integration   # ffmpeg / network / Demucs tests

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""
