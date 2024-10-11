# Audio Helper

`Audio Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🕸️ AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](logo.png)](https://harchaoui.org/warith/ai-helpers)

Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.

# Installation

## Install Package

We recommend using Python environments. Check this link if you're unfamiliar with setting one up:

[🥸 Tech tips](https://harchaoui.org/warith/4ml/#install)

## Install `ffmpeg` and `sox`
To install Audio Helper, you must install `ffmpeg`:

- For macos 🍎
  
  Get [brew](https://brew.sh)
  ```bash
  brew install ffmpeg
  brew install sox
  ```
- For Ubuntu 🐧
  ```bash
  sudo apt install ffmpeg
  sudo apt install sox
  ```
- For Windows 🪟
  - **FFmpeg**: Go to the [FFmpeg website](https://ffmpeg.org/download.html) and follow the instructions for downloading FFmpeg. You'll need to manually add FFmpeg to your system PATH.
  
  - **SoX**: Download SoX from the [SoX official website](http://sox.sourceforge.net/) or [Gyan.dev SoX Windows builds](https://www.gyan.dev/ffmpeg/builds/). Similarly, add SoX to your system PATH.


and finally:

```bash
pip install --force-reinstall --no-cache-dir git+https://github.com/warith-harchaoui/audio-helper.git@main
```

# Usage
Here’s an example of how to use Audio Helper to load, convert, and split an audio file:

```python
import audio_helper as ah

# Load an audio file
audio_file = "example.mp3"
audio, sample_rate = ah.load_audio(audio_file)

# Convert the audio file to a different format
output_audio = "example.wav"
ah.sound_converter(audio_file, output_audio)

# Split the audio file into chunks of 30 seconds
chunks = ah.split_audio_regularly(audio_file, "chunks_folder", split_time=30.0)

# Concatenate the chunks back together
new_concatenated_audio = "concatenated.wav"
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

sources = ah.separate_sources(
    "input_audio.m4a",
    output_folder="output_folder",
    device = "cpu", # or "cuda" if GPU or nothing to let it decide
    nb_workers: int = 4, # if cpu
    output_format: str = "mp3",
)

print(separated_sources)
# {'vocals': 'output_folder/vocals.mp3', 'drums': 'output_folder/drums.mp3', 'bass': 'output_folder/bass.mp3', 'other': 'output_folder/other.mp3'}
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

