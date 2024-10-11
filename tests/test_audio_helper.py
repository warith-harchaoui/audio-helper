import pytest
from pathlib import Path
import urllib.request
import os_helper as osh
import numpy as np
from audio_helper import (
    load_audio,
    sound_converter,
    extract_audio_chunk,
    split_audio_regularly,
    generate_silent_audio,
    get_audio_duration,
    is_valid_audio_file,
)

audio_url = "https://harchaoui.org/warith/never-gonna-give-you-up.mp3"
audio_filename = "never-gonna-give-you-up.mp3"



def test_audio():
    with osh.temporary_folder(prefix = "audio_tests") as temp_dir:
        # Downaload an audio file
        audio_file = osh.os_path_constructor([temp_dir, audio_filename])
        urllib.request.urlretrieve(audio_url, audio_file)

        assert is_valid_audio_file(audio_file), "Downloaded file should be a valid audio file"
        
        duration = get_audio_duration(audio_file)
        assert duration > 0, "Audio duration should be positive"
        
        audio, sample_rate = load_audio(audio_file, to_numpy=True, two_channels=False)
        assert isinstance(audio, np.ndarray), "Audio data should be a numpy array"
        assert audio.ndim == 1, "Audio data should be mono"
        assert sample_rate > 0, "Sample rate should be positive"

        wav = osh.os_path_constructor([temp_dir, "converted_audio.wav"])
        sound_converter(audio_file, wav, freq=44100, channels=2, overwrite=True)
        assert osh.is_valid_audio_file(wav) == True, "Converted audio file should be a valid audio file"

        chunk = osh.os_path_constructor([temp_dir, "audio_chunk.wav"])
        start = 5.0
        end = 10.0
        extract_audio_chunk(audio_file, start, end, chunk, overwrite=True)
        assert osh.is_valid_audio_file(chunk) == True, "Extracted audio chunk should be a valid audio file"
        duration = get_audio_duration(chunk)
        assert duration == end - start, f"Extracted audio chunk duration should match {end - start} vs {duration}"

        chunk_folder = osh.os_path_constructor([temp_dir, "splits"])
        split_time = 10.0
        duration = get_audio_duration(audio_file)
        chunks = split_audio_regularly(audio_file, chunk_folder, split_time, overwrite=True)
        nb_chunks = int(np.ceil(duration / split_time))
        assert len(chunks) == nb_chunks, f"Number of audio chunks should match {nb_chunks} vs {len(chunks)}"
        durations = [get_audio_duration(chunk) for chunk in chunks]
        assert all([d == split_time for d in durations[:-1]]), f"All audio chunks should have the same duration {split_time}"

        silent_audio = osh.os_path_constructor([temp_dir, "silent_audio.wav"])
        duration = 5.0
        generate_silent_audio(duration, silent_audio, sample_rate=44100, overwrite=True)
        audio, sample_rate = load_audio(silent_audio, to_numpy=True, to_mono=True)
        assert np.sum(np.abs(audio)) == 0, "Generated silent audio should be silent with all zeros"
        assert sample_rate == 44100, f"Sample rate of generated silent audio should match 44100 vs {sample_rate}"
        assert len(audio) == duration * sample_rate, f"Generated silent audio should have the correct duration {duration} vs {len(audio) / sample_rate}"


