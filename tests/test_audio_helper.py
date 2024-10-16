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
    separate_sources,
    audio_concatenation,
    sound_resemblance,
    save_audio,
)

audio_url = "https://harchaoui.org/warith/example.mp3"
audio_filename = "example.mp3"

osh.verbosity(0)

overwrite = True

def get_audio():
    folder = "audio_tests"
    audio_file = osh.os_path_constructor([folder, audio_filename])
    if not(osh.file_exists(audio_file)):
        osh.make_directory(folder)
        urllib.request.urlretrieve(audio_url, audio_file)
    return audio_file

def test_audio():
    audio_file = get_audio()
    folder, _, _ = osh.folder_name_ext(audio_file)

    assert is_valid_audio_file(audio_file), "Downloaded file should be a valid audio file"
    
    duration = get_audio_duration(audio_file)
    assert duration > 0, "Audio duration should be positive"
    
    audio, sample_rate = load_audio(audio_file, to_numpy=True, two_channels=False)
    assert isinstance(audio, np.ndarray), "Audio data should be a numpy array"
    assert audio.ndim == 1, "Audio data should be mono"
    assert sample_rate > 0, "Sample rate should be positive"

def test_audio_resemblance():
    audio_file = get_audio()
    folder, _, _ = osh.folder_name_ext(audio_file)

    audio, sample_rate = load_audio(audio_file, to_numpy=True, to_mono=True)

    # Resemblance with itself should be high
    resemblance = sound_resemblance(audio_file, audio_file)
    assert resemblance > 0.98, f"Audio file should have a resemblance of 1 with itself, got {round(100*resemblance)}%"

    # Add some slight noise to the audio file
    second_audio = np.array(audio)
    M = np.max(audio)
    sz = audio.shape
    second_audio += 0.05 * M * np.random.randn(*sz)
    second_audio_file = osh.os_path_constructor([folder, "second_audio.wav"])
    save_audio(second_audio, second_audio_file, sample_rate)

    resemblance = sound_resemblance(audio_file, second_audio_file)
    assert resemblance > 0.8, f"Audio file should have a high resemblance with a noisy version of itself, got {round(100*resemblance)}%"

def test_conversion():
    audio_file = get_audio()
    folder, _, _ = osh.folder_name_ext(audio_file)

    wav = osh.os_path_constructor([folder, "converted_audio.wav"])
    sound_converter(audio_file, wav, freq=44100, channels=2, overwrite=overwrite)
    assert is_valid_audio_file(wav) == True, "Converted audio file (mp3 to wav) should be a valid audio file"
    audio, sample_rate = load_audio(wav, to_numpy=True, to_mono=False)
    assert sample_rate == 44100, f"Sample rate of converted audio (mp3 to wav) should match 44100 vs {sample_rate}"

    ogg = osh.os_path_constructor([folder, "converted_audio.ogg"])
    sound_converter(audio_file, ogg, freq=44100, channels=2, overwrite=overwrite)
    assert is_valid_audio_file(ogg) == True, "Converted audio file (mp3 to ogg) should be a valid audio file"
    audio, sample_rate = load_audio(ogg, to_numpy=True, to_mono=False)
    assert sample_rate == 44100, f"Sample rate of converted audio (mp3 to ogg) should match 44100 vs {sample_rate}"

def test_chunk():
    audio_file = get_audio()
    folder, _, _ = osh.folder_name_ext(audio_file)

    chunk = osh.os_path_constructor([folder, "audio_chunk.wav"])
    start = 5.0
    end = 10.0
    extract_audio_chunk(audio_file, start, end, chunk, overwrite=overwrite)
    assert is_valid_audio_file(chunk) == True, "Extracted audio chunk should be a valid audio file"
    duration = get_audio_duration(chunk)
    assert duration == end - start, f"Extracted audio chunk duration should match {end - start} vs {duration}"

    chunk_folder = osh.os_path_constructor([folder, "splits"])
    split_time = 10.0
    duration = get_audio_duration(audio_file)
    chunks = split_audio_regularly(audio_file, chunk_folder, split_time, overwrite=overwrite)
    durations = [get_audio_duration(chunk) for chunk in chunks]
    total_duration = np.sum(durations)
    error = round(100.0*np.abs(total_duration - duration) / duration)
    assert error < 1.0, f"Total duration of audio chunks should match the original audio file {total_duration} vs {duration} (error: {error}%)"
    t = durations[:-1]
    mini = np.min(t)
    error1 = round(100.0*np.abs(mini - split_time) / split_time)
    maxi = np.max(t)
    error2 = round(100.0*np.abs(maxi - split_time) / split_time)
    error = error1 + error2
    assert error < 1.0, f"Duration of audio chunks should match the split time {split_time} vs min: {mini}, max:{maxi} (error: {error}%)"

    original_signal, sample_rate = load_audio(audio_file, to_numpy=True, to_mono=True)
    reconstruction = osh.os_path_constructor([folder, "concatenation_audio.mp3"])
    reconstruction = audio_concatenation(chunks, reconstruction, overwrite = overwrite)
    assert is_valid_audio_file(reconstruction) == True, "Concatenated audio file should be a valid audio file"
    duration = get_audio_duration(reconstruction)
    original_duration = get_audio_duration(audio_file)
    error = np.abs(duration - original_duration) / original_duration
    assert error < 0.01, f"Concatenated audio file should have the same duration as the original audio file {duration} vs {get_audio_duration(audio_file)} (error: {100*round(error)}%)"
    resemblance = sound_resemblance(reconstruction, audio_file)
    assert resemblance > 0.75, f"Concatenated audio file should have a high resemblance with the original audio file, got {round(100*resemblance)}%: {reconstruction} and {audio_file}"

def test_silent_audio():
    audio_file = get_audio()
    folder, _, _ = osh.folder_name_ext(audio_file)

    silent_audio = osh.os_path_constructor([folder, "silent_audio.wav"])
    duration = 5.0
    generate_silent_audio(duration, silent_audio, sample_rate=44100, overwrite=overwrite)
    audio, sample_rate = load_audio(silent_audio, to_numpy=True, to_mono=True)
    assert np.sum(np.abs(audio)) == 0, "Generated silent audio should be silent with all zeros"
    assert sample_rate == 44100, f"Sample rate of generated silent audio should match 44100 vs {sample_rate}"
    assert len(audio) == duration * sample_rate, f"Generated silent audio should have the correct duration {duration} vs {len(audio) / sample_rate}"

def test_separation():
    audio_file = get_audio()
    folder, _, _ = osh.folder_name_ext(audio_file)

    # separating sources
    sources_folder = osh.os_path_constructor([folder, "sources"])
    osh.make_directory(sources_folder)
    sources = separate_sources(audio_file, sources_folder, overwrite=overwrite)
    assert len(sources) == 4, f"Number of separated sources should be 4 vs {len(sources)}"
    keys_set = set(sources.keys())
    normal_keys_set = {"vocals", "drums", "bass", "other"}
    assert keys_set == normal_keys_set, f"Separated sources should have the correct keys {keys_set} vs {normal_keys_set}"
    original_signal, sample_rate = load_audio(audio_file, to_numpy=True, to_mono=True)
    reconstructed_signal = 0
    for k in sources:
        assert is_valid_audio_file(sources[k]), f"Separated source {k} should be a valid audio file"
        duration = get_audio_duration(sources[k])
        original_duration = get_audio_duration(audio_file)
        error = round(100.0*np.abs(duration - original_duration) / original_duration)
        assert error < 1.0, f"Separated source {k} should have the same duration as the original audio file {duration} vs {get_audio_duration(audio_file)}"
        source, _ = load_audio(sources[k], to_numpy=True, to_mono=True)
        reconstructed_signal += source

    mp3_audio_file = osh.os_path_constructor([sources_folder, "reconstructed_signal.mp3"])
    save_audio(reconstructed_signal, mp3_audio_file, sample_rate)

    resemblance = sound_resemblance(audio_file, mp3_audio_file)# , window_seconds = 0.2)
    assert resemblance > 0.75, f"Resemblance between original and reconstructed signals should be high, got {round(100*resemblance)}% audio_file: {audio_file} vs reconstructed: {mp3_audio_file}"
