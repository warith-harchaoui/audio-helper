"""
Local integration tests for audio-helper.

These exercise the ffmpeg-backed helpers using fixtures generated on the
fly (sine tones written with soundfile), so they require ffmpeg on PATH
but no network access. Marked ``integration`` so they stay off the default
``pytest`` run (which only collects pure-Python unit tests).

Usage Example
-------------
>>> # Run the integration tests explicitly:
>>> #   pytest -m integration tests/test_local.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from audio_helper import (
    audio_concatenation,
    extract_audio_chunk,
    generate_silent_audio,
    get_audio_duration,
    is_valid_audio_file,
    load_audio,
    mix_room_tone,
    save_audio,
    sound_converter,
    split_audio_regularly,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tone(
    path: Path,
    freq_hz: float = 440.0,
    sample_rate: int = 24000,
    duration: float = 1.0,
    amplitude: float = 0.5,
) -> Path:
    """Write a deterministic sine-wave WAV file and return its path."""
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    wavfile.write(str(path), sample_rate, signal)
    return path


# ---------------------------------------------------------------------------
# is_valid_audio_file
# ---------------------------------------------------------------------------


def test_is_valid_audio_file_valid_wav(tmp_path):
    tone = _write_tone(tmp_path / "tone.wav")
    assert is_valid_audio_file(str(tone)) is True


def test_is_valid_audio_file_text_with_audio_extension(tmp_path):
    """A text file with a fake .wav extension must be rejected by ffprobe."""
    fake = tmp_path / "not_audio.wav"
    fake.write_text("this is plain text, not audio")
    assert is_valid_audio_file(str(fake)) is False


def test_is_valid_audio_file_unknown_extension(tmp_path):
    """A valid WAV renamed to an unrecognized extension is rejected by the extension check."""
    tone = _write_tone(tmp_path / "tone.wav")
    renamed = tmp_path / "tone.xyz"
    tone.rename(renamed)
    assert is_valid_audio_file(str(renamed)) is False


def test_is_valid_audio_file_missing_path(tmp_path):
    assert is_valid_audio_file(str(tmp_path / "does_not_exist.wav")) is False


# ---------------------------------------------------------------------------
# get_audio_duration
# ---------------------------------------------------------------------------


def test_get_audio_duration_matches_tone(tmp_path):
    tone = _write_tone(tmp_path / "tone.wav", duration=2.0)
    duration = get_audio_duration(str(tone))
    assert abs(duration - 2.0) < 0.01


# ---------------------------------------------------------------------------
# sound_converter
# ---------------------------------------------------------------------------


def test_sound_converter_wav_to_wav_resample(tmp_path):
    src = _write_tone(tmp_path / "tone.wav", sample_rate=24000, duration=1.0)
    dst = tmp_path / "resampled.wav"
    sound_converter(str(src), str(dst), freq=16000, channels=1, overwrite=True)
    assert is_valid_audio_file(str(dst))
    _, sample_rate = load_audio(str(dst), to_numpy=True, to_mono=True)
    assert sample_rate == 16000


# ---------------------------------------------------------------------------
# extract_audio_chunk
# ---------------------------------------------------------------------------


def test_extract_audio_chunk_basic(tmp_path):
    src = _write_tone(tmp_path / "tone.wav", duration=3.0)
    out = tmp_path / "chunk.wav"
    extract_audio_chunk(str(src), 0.5, 2.0, str(out), overwrite=True)
    assert is_valid_audio_file(str(out))
    assert abs(get_audio_duration(str(out)) - 1.5) < 0.05


def test_extract_audio_chunk_bad_start_raises(tmp_path):
    src = _write_tone(tmp_path / "tone.wav", duration=1.0)
    with pytest.raises(AssertionError):
        extract_audio_chunk(str(src), 5.0, 6.0, str(tmp_path / "bad.wav"), overwrite=True)


def test_extract_audio_chunk_end_before_start_raises(tmp_path):
    src = _write_tone(tmp_path / "tone.wav", duration=2.0)
    with pytest.raises(AssertionError):
        extract_audio_chunk(str(src), 1.0, 0.5, str(tmp_path / "bad.wav"), overwrite=True)


# ---------------------------------------------------------------------------
# split_audio_regularly
# ---------------------------------------------------------------------------


def test_split_audio_regularly_three_chunks(tmp_path):
    src = _write_tone(tmp_path / "tone.wav", duration=3.0)
    out_dir = tmp_path / "splits"
    chunks = split_audio_regularly(
        str(src), str(out_dir), split_time=1.0, output_format="wav", overwrite=True
    )
    # Three full chunks; allow ±1 for boundary rounding.
    assert 2 <= len(chunks) <= 4
    for chunk in chunks:
        assert is_valid_audio_file(chunk)


# ---------------------------------------------------------------------------
# audio_concatenation
# ---------------------------------------------------------------------------


def test_audio_concatenation_two_tones(tmp_path):
    a = _write_tone(tmp_path / "a.wav", freq_hz=440.0, duration=1.0)
    b = _write_tone(tmp_path / "b.wav", freq_hz=880.0, duration=1.0)
    out = tmp_path / "concat.mp3"
    audio_concatenation([str(a), str(b)], str(out), overwrite=True)
    assert is_valid_audio_file(str(out))
    assert abs(get_audio_duration(str(out)) - 2.0) < 0.1


# ---------------------------------------------------------------------------
# generate_silent_audio
# ---------------------------------------------------------------------------


def test_generate_silent_audio_is_silent(tmp_path):
    out = tmp_path / "silence.wav"
    generate_silent_audio(2.0, str(out), sample_rate=16000, overwrite=True)
    assert is_valid_audio_file(str(out))
    audio, sr = load_audio(str(out), to_numpy=True, to_mono=True)
    assert sr == 16000
    assert float(np.max(np.abs(audio))) == 0.0
    assert abs(len(audio) - 2.0 * sr) <= 1


# ---------------------------------------------------------------------------
# mix_room_tone
# ---------------------------------------------------------------------------


def test_mix_room_tone_preserves_duration_and_adds_noise(tmp_path):
    src = _write_tone(tmp_path / "speech.wav", duration=1.5)
    out = tmp_path / "speech-rt.wav"
    mix_room_tone(str(src), str(out), noise_db=-30.0, color="pink", overwrite=True)
    assert is_valid_audio_file(str(out))
    # Duration must match the speech within a few ms.
    assert abs(get_audio_duration(str(out)) - 1.5) < 0.05
    # Output should not be silent — there's at least the original tone in it.
    audio, _ = load_audio(str(out), to_numpy=True, to_mono=True)
    assert float(np.max(np.abs(audio))) > 0.0


def test_mix_room_tone_rejects_bad_color(tmp_path):
    src = _write_tone(tmp_path / "speech.wav", duration=0.5)
    with pytest.raises(AssertionError):
        mix_room_tone(str(src), str(tmp_path / "out.wav"), color="rainbow", overwrite=True)


# ---------------------------------------------------------------------------
# save_audio
# ---------------------------------------------------------------------------


def test_save_audio_numpy_roundtrip(tmp_path):
    sr = 16000
    rng = np.random.default_rng(0)
    signal = (0.1 * rng.standard_normal(sr)).astype(np.float32)
    out = tmp_path / "out.wav"
    save_audio(signal, str(out), sample_rate=sr)
    assert is_valid_audio_file(str(out))
    loaded, loaded_sr = load_audio(str(out), to_numpy=True, to_mono=True)
    assert loaded_sr == sr
    assert abs(len(loaded) - len(signal)) <= 1
