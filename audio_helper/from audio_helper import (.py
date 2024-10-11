from audio_helper import (
    load_audio,
    sound_converter,
    extract_audio_chunk,
    split_audio_regularly,
    generate_silent_audio,
    get_audio_duration,
    is_valid_audio_file,
    separate_sources
)
import urllib.request
import os_helper as osh
import numpy as np
import scipy.io.wavfile


osh.verbosity(0)

audio_url = "https://harchaoui.org/warith/never-gonna-give-you-up.mp3"
audio_filename = "never-gonna-give-you-up.mp3"

temp_dir = "."

audio_file = osh.os_path_constructor([temp_dir, audio_filename])
urllib.request.urlretrieve(audio_url, audio_file)

wav = audio_file.replace(".mp3", ".wav")
sound_converter(audio_file, wav)

# separate_sources
sources_folder = osh.os_path_constructor([temp_dir, "sources"])
sources = separate_sources(wav, sources_folder, overwrite=True, nb_workers = -2)
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
d = np.abs(original_signal - reconstructed_signal
diff = np.mean(np.abs(original_signal - reconstructed_signal))
# assert diff < 1e-6, f"Reconstructed signal should match the original signal with a small difference {diff}"

wav_audio_file = osh.os_path_constructor([sources_folder, "reconstructed_signal.wav"])
scipy.io.wavfile.write(wav_audio_file, sample_rate, reconstructed_signal)
mp3_audio_file = osh.os_path_constructor([sources_folder, "reconstructed_signal.mp3"])
sound_converter(wav_audio_file, mp3_audio_file)

