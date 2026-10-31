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
from audio_helper.main import _source_bitrate

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


def test_sound_converter_mp3_honours_bitrate(tmp_path):
    """A requested bitrate must land in the file, and must change it.

    The no-bitrate call is measured alongside on purpose: ffmpeg picks
    32 kb/s for a 24 kHz mono stream, so a test that only asserted
    "some bitrate is present" would pass without the parameter doing
    anything.
    """
    src = _write_tone(tmp_path / "tone.wav", sample_rate=48000, duration=3.0)

    asked = tmp_path / "asked.mp3"
    sound_converter(str(src), str(asked), freq=24000, channels=1, bitrate="64k", overwrite=True)
    default = tmp_path / "default.mp3"
    sound_converter(str(src), str(default), freq=24000, channels=1, overwrite=True)

    assert is_valid_audio_file(str(asked))
    # 64 kb/s over three seconds is about 24 KiB; the ffmpeg default for this
    # stream is half that. Comparing sizes keeps the test free of ffprobe.
    assert asked.stat().st_size > default.stat().st_size * 1.5


def test_sound_converter_carries_the_input_bitrate_over(tmp_path):
    """Asked for no bitrate, a lossy input must keep its own.

    The caller named a container, not a quality. Before this, ffmpeg chose,
    and for a mono stream it chooses 32 kb/s: re-containering a 96 kb/s
    episode silently cost two thirds of it.
    """
    src = _write_tone(tmp_path / "tone.wav", sample_rate=44100, duration=4.0)
    source = tmp_path / "source.mp3"
    sound_converter(str(src), str(source), freq=44100, channels=1, bitrate="96k")

    inherited = tmp_path / "inherited.mp3"
    sound_converter(str(source), str(inherited), freq=44100, channels=1)

    # Same seconds at the same bitrate is the same size, within the slack of
    # a frame and an ID3 tag. Comparing sizes keeps the test off ffprobe.
    assert abs(inherited.stat().st_size - source.stat().st_size) < source.stat().st_size * 0.1


def test_sound_converter_lets_an_explicit_bitrate_beat_the_input(tmp_path):
    """Inheriting is the default, never an override of what was asked for."""
    src = _write_tone(tmp_path / "tone.wav", sample_rate=44100, duration=4.0)
    source = tmp_path / "source.mp3"
    sound_converter(str(src), str(source), freq=44100, channels=1, bitrate="96k")

    asked = tmp_path / "asked.mp3"
    sound_converter(str(source), str(asked), freq=44100, channels=1, bitrate="32k")
    assert asked.stat().st_size < source.stat().st_size / 2


def test_sound_converter_does_not_carry_a_pcm_bitrate_over(tmp_path):
    """An uncompressed input has no bitrate to carry, and must not pretend to.

    A 44.1 kHz 16-bit mono WAV reads 705 kb/s. Carrying that over would ask
    libmp3lame for its 320 kb/s ceiling and produce a file five times the
    size of a sane speech encode, for a source that holds no such detail.
    """
    src = _write_tone(tmp_path / "tone.wav", sample_rate=44100, duration=4.0)
    out = tmp_path / "out.mp3"
    sound_converter(str(src), str(out), freq=24000, channels=1)
    # 320 kb/s over four seconds would be about 156 KiB; anything near that
    # means the PCM figure leaked into the encoder.
    assert out.stat().st_size < 60 * 1024


def test_source_bitrate_reports_nothing_for_a_missing_file(tmp_path):
    """Choosing a default must not fail a conversion ffmpeg would complete."""
    assert _source_bitrate(str(tmp_path / "nowhere.mp3")) is None


def test_sound_converter_bitrate_ignored_for_wav(tmp_path):
    """A WAV output has no bitrate to set: the call must warn, not fail."""
    src = _write_tone(tmp_path / "tone.wav", sample_rate=24000, duration=1.0)
    dst = tmp_path / "out.wav"
    result = sound_converter(str(src), str(dst), freq=16000, channels=1, bitrate="64k")
    assert result == str(dst)
    assert is_valid_audio_file(str(dst))


def test_sound_converter_lossy_encoding_is_not_applied_twice(tmp_path):
    """Naming a lossy codec must encode once, not MP3-inside-WAV then MP3 again."""
    src = _write_tone(tmp_path / "tone.wav", sample_rate=48000, duration=2.0)
    once = tmp_path / "once.mp3"
    sound_converter(
        str(src), str(once), freq=24000, channels=1, encoding="libmp3lame", bitrate="64k"
    )
    assert is_valid_audio_file(str(once))
    _, sample_rate = load_audio(str(once), to_numpy=True, to_mono=True)
    assert sample_rate == 24000


def test_sound_converter_returns_output_path(tmp_path):
    """sound_converter must return the output path (both CLIs rely on it)."""
    src = _write_tone(tmp_path / "tone.wav", sample_rate=24000, duration=1.0)
    dst = tmp_path / "resampled.wav"
    result = sound_converter(str(src), str(dst), freq=16000, channels=1, overwrite=True)
    assert result == str(dst)


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


def test_extract_audio_chunk_no_overwrite_skips_recompute(tmp_path):
    # overwrite=False used to be a no-op (ffmpeg always re-ran); an existing
    # valid output should now be returned as-is, mirroring every sibling
    # function's _overwrite_audio_file skip behavior.
    src = _write_tone(tmp_path / "tone.wav", duration=3.0)
    out = tmp_path / "chunk.wav"
    extract_audio_chunk(str(src), 0.5, 2.0, str(out), overwrite=True)
    mtime_before = out.stat().st_mtime_ns

    # A bad range would normally raise, but overwrite=False should skip
    # validation entirely once a valid output already exists.
    result = extract_audio_chunk(str(src), 50.0, 60.0, str(out), overwrite=False)
    assert result == str(out)
    assert out.stat().st_mtime_ns == mtime_before


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


def test_split_audio_regularly_sub_one_second_file_still_returns_a_chunk(tmp_path):
    # Regression test: the loop used to gate its very first iteration on
    # `time_cursor < total_duration - 1`, meant to drop a trailing remainder
    # under 1s. Applied to the whole file (not just a remainder), any input
    # shorter than 1 second matched that same condition on iteration zero and
    # the function silently returned [] — the entire file dropped, not just a
    # remainder. The first chunk must now always be produced.
    src = _write_tone(tmp_path / "short.wav", duration=0.9)
    out_dir = tmp_path / "short_splits"
    chunks = split_audio_regularly(
        str(src), str(out_dir), split_time=5.0, output_format="wav", overwrite=True
    )
    assert len(chunks) == 1
    assert is_valid_audio_file(chunks[0])
    assert abs(get_audio_duration(chunks[0]) - 0.9) < 0.05


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


def test_generate_silent_audio_rejects_nonpositive_duration(tmp_path):
    # A negative/zero duration used to reach np.zeros(int(duration * sample_rate))
    # unguarded: negative -> a cryptic numpy ValueError, zero -> a silently
    # written zero-length "silent" file. Both should be a clear AssertionError.
    for bad_duration in (0.0, -1.0):
        with pytest.raises(AssertionError):
            generate_silent_audio(bad_duration, str(tmp_path / "bad.wav"), overwrite=True)


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


def test_save_audio_rejects_unsupported_type(tmp_path):
    # Neither a torch.Tensor nor a numpy.ndarray used to silently no-op (no
    # error, no file written) despite the docstring promising AssertionError.
    with pytest.raises(AssertionError):
        save_audio([0.1, 0.2, 0.3], str(tmp_path / "out.wav"), sample_rate=16000)
