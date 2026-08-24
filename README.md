# Audio Helper

[🇫🇷](https://github.com/warith-harchaoui/audio-helper/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/audio-helper/blob/main/README.md)

[![CI](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/warith-harchaoui/audio-helper/blob/main/LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#) [![Local-first](https://img.shields.io/badge/local--first-ffmpeg%20%2B%20local%20Demucs-brightgreen.svg)](#the-promise)

`Audio Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/logo.png)](https://harchaoui.org/warith/ai-helpers)

Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.

Audio Helper is battle-tested: every change ships through automated tests and continuous integration before it reaches PyPI, versions follow [Semantic Versioning](https://semver.org/), and two other published packages in the suite, [`youtube-helper`](https://github.com/warith-harchaoui/youtube-helper) and [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper), build directly on top of it in production. Audio Helper itself depends on [`os-helper`](https://github.com/warith-harchaoui/os-helper), the shared foundation package every library in the suite uses for logging and file management. That is what "battle-tested" means here: not a claim, but a dependency graph other shipped code actually stands on, on both sides of this package.

## The Promise

Audio Helper is **local-first** by design. Three honest cases:

1. **Guaranteed local.** Every operation, including the browser GUI at
   `GET /gui`, runs on your machine via **ffmpeg** and **local Demucs**
   (Meta's AI model for splitting a mixed track into its component sounds:
   vocals, drums, bass, everything else). Your audio is **never uploaded**
   to any third party. There is **no telemetry, no account, no SaaS**
   dependency.
2. **The one caveat: model weights.** Source separation downloads the
   **Demucs** model weights **once**, on first run (a normal Hugging Face /
   PyTorch cache fetch). After that it is fully offline. Nothing else needs the
   network.
3. **Your decision.** Nothing here forces the cloud. If you ever want to run
   behind a proxy or in a container, the FastAPI surface makes that easy, but
   that is a choice you make, not a default we impose.

## Documentation

[💻 Documentation](https://harchaoui.org/warith/ai-helpers/docs/audio-helper-doc/)

[🗺️ Landscape](https://github.com/warith-harchaoui/audio-helper/blob/main/LANDSCAPE.md)

[📋 Examples](https://github.com/warith-harchaoui/audio-helper/blob/main/EXAMPLES.md)

## Features

- Audio Loading: load files with optional resampling and mono downmix.
- Sound Conversion: ffmpeg-backed format/sample-rate/channels conversion.
- Source Separation: vocals / drums / bass / other via Demucs (optional `[demucs]` extra).
- Audio Splitting: fixed-duration chunks and arbitrary `[start, end]` slices.
- Concatenation: head-to-tail join into any ffmpeg-supported container.
- Silent Audio Generation: write silence of a specified duration.
- Room-Tone Mixing: pink/white/brown ambient noise to mask edits between cuts.
- Similarity: `sound_resemblance` score for A/B comparison, based on
  Mel-Frequency Cepstral Coefficients (MFCCs), a standard way of boiling a
  clip's spectrum down to a compact set of numbers that captures how it
  sounds rather than its raw waveform, so two recordings of the same voice
  score close together even if the takes differ slightly.
- Feature Extraction: scipy-based Mel / MFCC primitives.

**Five surfaces, one toolkit.** Every operation above is reachable as:

- **Library**: `import audio_helper as ah`.
- **CLI ×2**: `audio-helper` (argparse, always installed) and
  `audio-helper-click` (click twin, `[cli]` extra) with identical flags.
- **HTTP API**: FastAPI app (`[api]` extra), OpenAPI docs at `/docs`.
- **MCP**: the same FastAPI app exposed through the Model Context Protocol
  (MCP), a standard that lets an AI agent call a program's functions as
  tools instead of a human clicking through a UI (`audio-helper-mcp`,
  `[mcp]` extra) for any MCP-aware agent host.
- **GUI**: a build-step-free browser **Recipe Canvas** served at `GET /gui`.
  Chain the eight verbs into a sequential pipeline, hear every intermediate step
  (WaveSurfer waveforms), bypass any step for instant A/B, use the ear-first
  before/after comparator (Space bar toggles), and export the pipeline as a
  committable `recipe.yaml`.
  See [GUI.md](https://github.com/warith-harchaoui/audio-helper/blob/main/GUI.md).

For the exhaustive trigger catalogue, see
[TRIGGERS.md](https://github.com/warith-harchaoui/audio-helper/blob/main/TRIGGERS.md).

## Installation

**Prerequisites**: **Python 3.10–3.13**, **git**, **ffmpeg**, cross-platform:

- 🍎 **macOS** ([Homebrew](https://brew.sh)): `brew install python git ffmpeg`
- 🐧 **Ubuntu/Debian**: `sudo apt update && sudo apt install -y python3 python3-pip git ffmpeg`
- 🪟 **Windows** (PowerShell): `winget install Python.Python.3.12 Git.Git Gyan.FFmpeg`

We recommend using Python environments. Check this link if you're unfamiliar with setting one up: [🥸 Tech tips](https://harchaoui.org/warith/4ml/#install).

### From PyPI (recommended)

```bash
# Core audio utilities only (load, convert, split, concatenate, silent audio, chunks)
pip install audio-helper

# Add source separation (pulls in torch + torchaudio, ~2 GB)
pip install "audio-helper[demucs]"

# Optional surfaces
pip install "audio-helper[cli]"       # click-based CLI twin
pip install "audio-helper[api]"       # FastAPI HTTP surface
```

### From source (no PyPI)

```bash
git clone https://github.com/warith-harchaoui/audio-helper.git
cd audio-helper
pip install -e .

# Add source separation (pulls in torch + torchaudio, ~2 GB)
pip install -e ".[demucs]"

# Optional surfaces
pip install -e ".[cli]"
pip install -e ".[api]"
```

If you call `separate_sources` without the `[demucs]` extra, the function raises an `ImportError` pointing you back here.

## Usage

For the full catalog of recipes, see [📋 EXAMPLES.md](https://github.com/warith-harchaoui/audio-helper/blob/main/EXAMPLES.md).

Here's an example of how to use Audio Helper to load, convert, and split an audio file:

(download [example.mp3](https://harchaoui.org/warith/example.mp3) )

It is part of a JFK speech that is badly recorded

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

print(sources)
# {'vocals': 'audio_tests/vocals.mp3', 'drums': 'audio_tests/drums.mp3', 'bass': 'audio_tests/bass.mp3', 'other': 'audio_tests/other.mp3'}
```

## Multi-surface exposure

`audio-helper` is not just a library: the same functions are exposed
as a CLI and a FastAPI HTTP surface:

```bash
# Python library (default)
import audio_helper as ah

# argparse-based CLI (installed automatically)
audio-helper convert --input in.mp3 --output out.wav --freq 44100
audio-helper split --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper separate --input mix.mp3 --output-dir stems/
audio-helper resemblance --a a.mp3 --b b.mp3

# click-based CLI twin (needs the [cli] extra)
pip install "audio-helper[cli]"
# or from source:
pip install "audio-helper[cli]"
audio-helper-click convert --input in.mp3 --output out.wav --freq 44100

# FastAPI HTTP surface (needs the [api] extra)
pip install "audio-helper[api]"
# or from source:
pip install "audio-helper[api]"
uvicorn audio_helper.api:app --port 8000
# → OpenAPI docs at http://localhost:8000/docs

# MCP tools for any MCP-aware agent host (needs the [mcp] extra)
pip install "audio-helper[mcp]"
audio-helper-mcp
# → same app + an /mcp endpoint (fastapi-mcp)
```

Docker image (light, without Demucs by default):

```bash
docker build -t audio-helper .
docker run --rm -p 8000:8000 audio-helper
# with Demucs:
docker build --build-arg WITH_DEMUCS=1 -t audio-helper:demucs .
```

The **Recipe Canvas** GUI ships today: it is served by the FastAPI app at
`GET /gui` (open `http://localhost:8000/gui` after starting the server), and
already covers the sequential pipeline builder, the ear-first before/after
comparator, and `recipe.yaml` export/import. Batch triage across many files
(a drop-zone contact sheet, an MFCC-cluster view) is the part still on the
roadmap. See [GUI.md](https://github.com/warith-harchaoui/audio-helper/blob/main/GUI.md)
for the full split between what ships and what is planned.

## Author

 - [Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)

## Acknowledgements

Special thanks to [Mohamed Chelali](https://mchelali.github.io) and [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug) for fruitful discussions.

## License

This project is licensed under the BSD-3-Clause License: see the [LICENSE](https://github.com/warith-harchaoui/audio-helper/blob/main/LICENSE) file for details.
