from audio_helper import (
    load_audio,
    sound_converter,
    extract_audio_chunk,
    split_audio_regularly,
    generate_silent_audio,
    get_audio_duration,
    is_valid_audio_file,
    separate_sources,
    sound_resemblance,
    save_audio,audio_concatenation
)
import urllib.request
import os_helper as osh
import numpy as np
import scipy.io.wavfile


osh.verbosity(0)

audio_url = "https://harchaoui.org/warith/never-gonna-give-you-up.mp3"
audio_filename = "never-gonna-give-you-up.mp3"

folder = "audio_tests"

audio_file = osh.os_path_constructor([folder, audio_filename])
urllib.request.urlretrieve(audio_url, audio_file)

wav = audio_file.replace(".mp3", ".wav")
sound_converter(audio_file, wav)

# separate_sources
sources_folder = osh.os_path_constructor([folder, "sources"])
sources = separate_sources(wav, sources_folder, overwrite=True, nb_workers = 3)
assert len(sources) == 4, f"Number of separated sources should be 4 vs {len(sources)}"
keys_set = set(sources.keys())
normal_keys_set = {"vocals", "drums", "bass", "other"}
assert keys_set == normal_keys_set, f"Separated sources should have the correct keys {keys_set} vs {normal_keys_set}"
original_signal, sample_rate = load_audio(audio_file, to_numpy=True, to_mono=True)
reconstructed_signal = np.zeros_like(original_signal)
for k in sources:
    assert is_valid_audio_file(sources[k]), f"Separated source {k} should be a valid audio file"
    duration = get_audio_duration(sources[k])
    assert duration == get_audio_duration(audio_file), f"Separated source {k} should have the same duration as the original audio file {duration} vs {get_audio_duration(audio_file)}"
    source, _ = load_audio(sources[k], to_numpy=True, to_mono=True)
    reconstructed_signal += source

reconstructed_signal /= max(reconstructed_signal)
reconstructed_signal *= max(original_signal)
d = np.abs(original_signal - reconstructed_signal)
diff = np.mean(np.abs(original_signal - reconstructed_signal))
# assert diff < 1e-6, f"Reconstructed signal should match the original signal with a small difference {diff}"

wav_audio_file = osh.os_path_constructor([sources_folder, "reconstructed_signal.wav"])
scipy.io.wavfile.write(wav_audio_file, sample_rate, reconstructed_signal)
mp3_audio_file = osh.os_path_constructor([sources_folder, "reconstructed_signal.mp3"])
sound_converter(wav_audio_file, mp3_audio_file)

# measure correlation between signals reconstructed_signal and original_signal
from scipy.signal import correlate
energy_original = np.sum(original_signal**2)
energy_reconstructed = np.sum(reconstructed_signal**2)
max_correlation = np.max(correlate(original_signal, reconstructed_signal))
max_correlation /= np.sqrt(energy_original * energy_reconstructed)
print(f"Max intercorrelation between original and reconstructed signals: {max_correlation}")

audio, sample_rate = load_audio(audio_file, to_numpy=True, to_mono=True)

resemblance = sound_resemblance(audio_file, audio_file, window_seconds = 0.2)
assert resemblance > 0.98, f"Audio file should have a resemblance of 1 with itself, got {round(100*resemblance)}%"

second_audio = np.array(audio)
M = np.max(audio)
sz = audio.shape
second_audio += 0.05 * M * np.random.randn(*sz)
second_audio_file = osh.os_path_constructor([folder, "second_audio.wav"])
save_audio(second_audio, second_audio_file, sample_rate)

resemblance = sound_resemblance(audio_file, second_audio_file, window_seconds = 0.2)
assert resemblance > 0.94, f"Audio file should have a high resemblance with a noisy version of itself, got {round(100*resemblance)}%"

chunk = osh.os_path_constructor([folder, "audio_chunk.wav"])
start = 5.0
end = 10.0
extract_audio_chunk(audio_file, start, end, chunk, overwrite=False)
assert is_valid_audio_file(chunk) == True, "Extracted audio chunk should be a valid audio file"
duration = get_audio_duration(chunk)
assert duration == end - start, f"Extracted audio chunk duration should match {end - start} vs {duration}"


chunk_folder = osh.os_path_constructor([folder, "splits"])
split_time = 10.0
duration = get_audio_duration(audio_file)
chunks = split_audio_regularly(audio_file, chunk_folder, split_time, overwrite=False)
nb_chunks = int(np.ceil(duration / split_time))
assert len(chunks) == nb_chunks, f"Number of audio chunks should match {nb_chunks} vs {len(chunks)}"
durations = [get_audio_duration(chunk) for chunk in chunks]
assert all([d == split_time for d in durations[:-1]]), f"All audio chunks should have the same duration {split_time}"



original_signal, sample_rate = load_audio(audio_file, to_numpy=True, to_mono=True)
reconstruction = osh.os_path_constructor([folder, "concatenation_audio.wav"])
reconstruction = audio_concatenation(chunks, reconstruction, overwrite = True)
assert is_valid_audio_file(reconstruction) == True, "Concatenated audio file should be a valid audio file"
duration = get_audio_duration(reconstruction)
original_duration = get_audio_duration(audio_file)
error = np.abs(duration - original_duration) / original_duration
assert error < 0.01, f"Concatenated audio file should have the same duration as the original audio file {duration} vs {get_audio_duration(audio_file)} (error: {100*round(error)}%)"
reconstruction_signal = []
for chunk in chunks:
    signal, _ = load_audio(chunk, to_numpy=True, to_mono=True)
    reconstruction_signal.append(signal.ravel())
reconstruction_signal = np.hstack(reconstruction_signal)
resemblance = sound_resemblance(reconstruction, audio_file)
assert resemblance > 0.94, f"Concatenated audio file should have a high resemblance with the original audio file, got {round(100*resemblance)}%"
