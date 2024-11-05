# Audio Helper

`Audio Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🕸️ AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](logo.png)](https://harchaoui.org/warith/ai-helpers)

Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.

# Installation

## Install Package

We recommend using Python environments. Check this link if you're unfamiliar with setting one up:

[🥸 Tech tips](https://harchaoui.org/warith/4ml/#install)

## Install `ffmpeg` 
To install Audio Helper, you must install `ffmpeg`:

- For macos 🍎
  
  Get [brew](https://brew.sh)
  ```bash
  brew install ffmpeg
  ```
- For Ubuntu 🐧
  ```bash
  sudo apt install ffmpeg
  ```
- For Windows 🪟
  Go to the [FFmpeg website](https://ffmpeg.org/download.html) and follow the instructions for downloading FFmpeg. You'll need to manually add FFmpeg to your system PATH.
  
and finally we still discuss between different python package managers and try to support as much as possible



```bash
pip install --force-reinstall --no-cache-dir git+https://github.com/warith-harchaoui/audio-helper.git@v1.0.0
```

# Usage
Here’s an example of how to use Audio Helper to load, convert, and split an audio file:

(download [example.mp3](https://harchaoui.org/warith/example.mp3) )

```python
import audio_helper as ah

# Load an audio file
audio_file = "example.mp3"
audio, sample_rate = ah.load_audio(audio_file)

# Convert the audio file to a different format
output_audio = "audio_tests/example.wav"
ah.sound_converter(audio_file, output_audio)

# Split the audio file into chunks of 30 seconds
chunks = ah.split_audio_regularly(audio_file, "audio_tests/chunks_folder", split_time=30.0, overwrite = True)
# Concatenate the chunks back together
new_concatenated_audio = "audio_tests/concatenated.wav"
concatenated_audio = ah.audio_concatenation(chunks, output_audio_filename = new_concatenated_audio)
```

Another cool example is about source separation (DEMUCS from META) with AI separating one audio track into 4 tracks:
- vocals
- drums
- bass
- other

It works with speech and songs

```python
import audio_helper as ah

audio_path = "input_audio.m4a"

sources = ah.separate_sources(
    audio_path,
    output_folder="audio_tests",
    device = "cpu", # or "cuda" if GPU or nothing to let it decide
    nb_workers = 4, # ignored if not cpu
    output_format = "mp3",
)

print(separated_sources)
# {'vocals': 'audio_tests/vocals.mp3', 'drums': 'audio_tests/drums.mp3', 'bass': 'audio_tests/bass.mp3', 'other': 'audio_tests/other.mp3'}
```

# Features
- Audio Loading: Load audio files with optional resampling and mono conversion.
- Sound Conversion: Convert audio files to different formats using ffmpeg.
- Source Separation: Separate an audio file into vocals, drums, bass, and other stems using a pre-trained PyTorch model (Demucs).
- Audio Splitting: Split audio files into chunks based on duration.
- Concatenation: Concatenate multiple audio files into one.
- Silent Audio Generation: Create silent audio files of a specified duration.
- Chunk Extraction: Extract specific segments from an audio file.

# Authors
 - [Warith Harchaoui](https://harchaoui.org/warith)
 - [Mohamed Chelali](https://mchelali.github.io)
 - [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug)

