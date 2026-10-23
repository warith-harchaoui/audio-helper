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
    mel_filter_banks,
    mel_to_hz,
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


# --- band guards ------------------------------------------------------------
# A filter bank cannot describe energy the FFT does not represent. Asking for a
# band above Nyquist used to walk off the end of the fbank array and die on a
# raw IndexError raised inside the inner loop, naming neither the argument at
# fault nor the constraint it broke.


def test_mel_filter_banks_rejects_a_band_above_nyquist():
    import pytest

    with pytest.raises(AssertionError, match="Nyquist"):
        mel_filter_banks(26, 512, 16000, 0, 16000)


def test_mel_filter_banks_rejects_an_inverted_band():
    import pytest

    with pytest.raises(AssertionError, match="below high_freq"):
        mel_filter_banks(26, 512, 16000, 4000, 1000)


def test_mel_filter_banks_accepts_exactly_nyquist():
    """The boundary itself is legal -- it is what mfcc() defaults to."""
    fbank = mel_filter_banks(26, 512, 16000, 0, 8000)
    assert fbank.shape == (26, 257)
    assert np.isfinite(fbank).all()


def test_mel_filter_banks_warns_when_filters_collapse(caplog):
    """Floored bin edges can collapse, leaving filters that contribute nothing.

    The caller then gets fewer effective filters than requested. Silently
    returning a bank with dead rows is the failure mode worth catching.
    """
    fbank = mel_filter_banks(80, 256, 8000, 0, 4000)
    empty = int((fbank.sum(axis=1) == 0).sum())
    assert empty > 0, "fixture no longer reproduces collapsed bins"
    assert "empty" in caplog.text.lower()


# ── a destination that is also a source ───────────────────────────────────


def test_an_output_that_is_one_of_the_inputs_is_refused() -> None:
    """The guard behind four writers, tested where it actually decides.

    `_refuse_output_over_input` is pure path arithmetic and runs before ffmpeg
    is reached, so it needs no audio and no ffmpeg to test — which is why it
    lives in the suite that runs by default rather than behind the integration
    marker.
    """
    import pytest

    from audio_helper.main import _refuse_output_over_input

    # The ordinary case stays silent.
    assert _refuse_output_over_input("out.wav", ["a.wav", "b.wav"]) is None

    with pytest.raises(ValueError, match="both an input and the output"):
        _refuse_output_over_input("a.wav", ["a.wav", "b.wav"])

    # Last in the list, not just first.
    with pytest.raises(ValueError, match="both an input and the output"):
        _refuse_output_over_input("b.wav", ["a.wav", "b.wav"])


def test_the_same_file_spelled_differently_is_still_refused() -> None:
    """`a.wav` and `./a.wav` are one file, and a plain string compare misses it.

    Without this, the guard is walked around by accident rather than on
    purpose: `./` in front of a path is how plenty of shell pipelines spell it.
    """
    import pytest

    from audio_helper.main import _refuse_output_over_input

    for spelling in ("./a.wav", "sub/../a.wav"):
        with pytest.raises(ValueError, match="both an input and the output"):
            _refuse_output_over_input(spelling, ["a.wav"])
