"""
Pure-Python unit tests for audio-helper.

These tests exercise the MFCC / Mel-scale primitives and the
``sound_resemblance`` similarity metric without touching ffmpeg or the
network, so they run in every default CI job. Integration-style tests
that shell out to ffmpeg live in :mod:`tests.test_local` under the
``integration`` marker.

Usage Example
-------------
>>> # Run only the unit tests (default):
>>> #   pytest tests/test_unit.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

import numpy as np
from scipy.io import wavfile

from audio_helper.main import (
    hz_to_mel,
    mel_to_hz,
    mel_filter_banks,
    mfcc,
    sound_resemblance,
)


def test_mel_roundtrip():
    for hz in [0.0, 100.0, 440.0, 1000.0, 8000.0]:
        assert np.isclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-6)


def test_mel_filter_banks_shape():
    num_filters = 26
    n_fft = 512
    sample_rate = 16000
    fb = mel_filter_banks(num_filters, n_fft, sample_rate, 0, sample_rate // 2)
    assert fb.shape == (num_filters, n_fft // 2 + 1)
    assert fb.min() >= 0.0
    assert fb.max() <= 1.0 + 1e-9


def test_mfcc_shape():
    sample_rate = 16000
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(sample_rate).astype(np.float32)
    coefs = mfcc(signal, sample_rate, num_mfcc=13, n_fft=512)
    assert coefs.ndim == 2
    assert coefs.shape[1] == 13


def _write_tone(path, freq_hz, sample_rate=24000, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    wavfile.write(str(path), sample_rate, signal)


def test_sound_resemblance_identity(tmp_path):
    f = tmp_path / "tone.wav"
    _write_tone(f, 440.0)
    assert sound_resemblance(str(f), str(f)) > 0.999


def test_sound_resemblance_bounded(tmp_path):
    f_a = tmp_path / "a.wav"
    f_b = tmp_path / "b.wav"
    _write_tone(f_a, 440.0)
    _write_tone(f_b, 1760.0)
    score = sound_resemblance(str(f_a), str(f_b))
    assert 0.0 <= score <= 1.0
