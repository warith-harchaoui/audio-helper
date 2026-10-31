# Audio Helper

[🇫🇷](https://github.com/warith-harchaoui/audio-helper/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/audio-helper/blob/main/README.md)

[![CI](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/warith-harchaoui/audio-helper/blob/main/LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#) [![Local-first](https://img.shields.io/badge/local--first-ffmpeg%20%2B%20local%20Demucs-brightgreen.svg)](#the-promise)

`Audio Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.

[🌍 AI Helpers](https://deraison.ai/ai-helpers)

[![logo](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/logo.png)](https://deraison.ai/ai-helpers)

Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.

Audio Helper is battle-tested. Every change ships through automated tests and continuous integration before it reaches PyPI, and versions follow [Semantic Versioning](https://semver.org/).

Two other published packages in the suite build directly on top of it in production: [`youtube-helper`](https://github.com/warith-harchaoui/youtube-helper) and [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper). Audio Helper itself depends on [`os-helper`](https://github.com/warith-harchaoui/os-helper), the shared foundation every library in the suite uses for logging and file management.

That is what "battle-tested" means here: not a claim, but a dependency graph other shipped code actually stands on, on both sides of this package.

## The Promise

Audio Helper is **local-first** by design. Three honest cases:

1. **Guaranteed local.** Every operation, including the browser GUI at
   `GET /gui`, runs on your machine via **ffmpeg** and **local Demucs**
   (Meta's AI model for splitting a mixed track into its component sounds:
   vocals, drums, bass, everything else). Your audio is **never uploaded**
   to any third party. There is **no telemetry, no account, no SaaS**
   dependency.
2. **The one caveat: model weights.** Source separation downloads the
   **Hybrid Demucs** weights **once**, on first run: `separate_sources` runs
   torchaudio's `HDEMUCS_HIGH_MUSDB_PLUS` pipeline, which fetches about
   335 MB from `download.pytorch.org` into the PyTorch Hub cache
   (`torch.hub.get_dir()`, `~/.cache/torch/hub` by default). After that it is
   fully offline. Nothing else needs the network.
3. **Your decision.** Nothing here forces the cloud. If you ever want to run
   behind a proxy or in a container, the FastAPI surface makes that easy, but
   that is a choice you make, not a default we impose.

## Documentation

[💻 Documentation](https://deraison.ai/ai-helpers/docs/audio-helper-doc/)

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
- **CLI**: `audio-helper` — one command line, built with click, always
  installed.
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

## How it works

![How audio-helper works: one audio file in, ffmpeg and a local Demucs between, stems and clips out](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/diagrams/how-it-works.png)

## Installation

**Prerequisites**: **Python 3.10–3.13**, **git**, **ffmpeg**, cross-platform:

- 🍎 **macOS** ([Homebrew](https://brew.sh)): `brew install python git ffmpeg`
- 🐧 **Ubuntu/Debian**: `sudo apt update && sudo apt install -y python3 python3-pip git ffmpeg`
- 🪟 **Windows** (PowerShell): `winget install Python.Python.3.12 Git.Git Gyan.FFmpeg`

We recommend using Python environments. Check this link if you're unfamiliar with setting one up: [🥸 Tech tips](https://deraison.ai/4ml/#install).

### From PyPI (recommended)

```bash
# Core audio utilities only (load, convert, split, concatenate, silent audio, chunks)
pip install audio-helper

# Add source separation (pulls in torch + torchaudio, ~2 GB)
pip install "audio-helper[demucs]"

# Optional surfaces
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
pip install -e ".[api]"
```

If you call `separate_sources` without the `[demucs]` extra, the function raises an `ImportError` pointing you back here.

## Quick start

For the full catalog of recipes, see [📋 EXAMPLES.md](https://github.com/warith-harchaoui/audio-helper/blob/main/EXAMPLES.md).

Download the test file — [example.mp3](https://deraison.ai/example.mp3), 60
seconds of a badly recorded JFK speech, MP3 at 48 kHz stereo — then load it,
re-encode it, cut it into windows and put it back together:

```python
import audio_helper as ah
import os_helper as osh

# No writer creates its destination folder: ffmpeg is handed a path it expects
# to already exist. os-helper, the suite's file-management foundation, makes it.
osh.make_directory("out")

# Load. `to_numpy=True` returns a numpy array; the default returns a torch
# tensor and therefore needs the optional `[demucs]` extra.
audio, sample_rate = ah.load_audio("example.mp3", to_numpy=True)
print(audio.shape, sample_rate)
# (2880000,) 48000

# Re-encode: container, sample rate and channel count in one call.
print(ah.sound_converter("example.mp3", "out/example.wav", freq=16000, channels=1))
# out/example.wav

# Cut into 25-second windows, then put them back head to tail.
chunks = ah.split_audio_regularly("example.mp3", "out/windows", split_time=25.0, overwrite=True)
print(len(chunks))
# 3
joined = ah.audio_concatenation(chunks, "out/rejoined.mp3", overwrite=True)
print(f"{ah.get_audio_duration(joined):.2f}")
# 60.00
```

The other headline operation is source separation: one mixed track in, four
stems out (`drums`, `bass`, `other`, `vocals` — the order torchaudio's Hybrid
Demucs bundle declares them in). It works on speech as well as on songs, and
it needs the `[demucs]` extra plus that one-time 335 MB weight download.

```python
import audio_helper as ah

sources = ah.separate_sources(
    "song.mp3",
    output_folder="stems",
    device="cpu",  # or "cuda"; pass None to let it decide
    nb_workers=4,  # ignored unless device == "cpu"
    output_format="mp3",
    overwrite=True,
)

for stem, path in sources.items():
    print(stem, path)
```

Each stem is written as `<output_folder>/<stem>.<output_format>` and is
**mono**: the model works in stereo, then `separate_sources` downmixes every
stem before writing it.

## Multi-surface exposure

`audio-helper` is not just a library: the same functions are exposed
as a CLI and a FastAPI HTTP surface:

```bash
# Python library (default)
import audio_helper as ah

# CLI (installed automatically)
audio-helper convert --input in.mp3 --output out.wav --freq 44100
audio-helper split --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper separate --input mix.mp3 --output-dir stems/
audio-helper resemblance --a a.mp3 --b b.mp3


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

The page speaks **French and English**. `GET /gui?lang=fr|en` picks one, a flag in the header toggles between them, and a code the locale file does not carry falls back to the default rather than failing.

Every string lives in [`audio_helper/locales/fr/LC_MESSAGES/audio_helper.po`](https://github.com/warith-harchaoui/audio-helper/blob/main/audio_helper/locales/fr/LC_MESSAGES/audio_helper.po), a standard `gettext` catalogue read through `os_helper.Catalog` — the loader shared across the suite — so translating the page never means opening a Python file.

Two things stay in one language on purpose, and the file says why:

- the per-verb parameter labels name the form fields the request actually carries;
- the comment header of an exported `recipe.yaml` belongs to a committable artifact rather than to the interface.

### Use it as an agent skill

`skills/audio-helper/` packages audio-helper as a **Claude Skill** *and* an **OpenCode
skill**, so an agent can reach for it on your behalf. See
[`skills/README.md`](https://github.com/warith-harchaoui/audio-helper/blob/main/skills/README.md) to install it (symlink into
`~/.claude/skills/` and `~/.opencode/skills/`), and
[`TRIGGERS.md`](https://github.com/warith-harchaoui/audio-helper/blob/main/TRIGGERS.md) for the exhaustive trigger catalogue —
`scripts/check_triggers.py` keeps the skill's description in sync with it, and
the test suite fails if a capability stops being named.

## Rust rewrite

A companion crate, [`audio-helper-rs`](https://github.com/warith-harchaoui/audio-helper-rs), rewrites the file-level surface — probe, convert, load-to-PCM, chunk, silence, concatenate, room tone, split — in idiomatic Rust.

It is not a port and not a replacement: `separate_sources` (Demucs) and `sound_resemblance` (MFCC) stay Python-only, disproportionate for a Rust crate to reimplement.

The payoff of reaching for it instead of this package is a dependency that compiles straight into your own Rust binary, with no Python interpreter, no `pip install` and no virtualenv on the target machine. Useful when audio-helper's job is a small piece of a larger Rust program, or when the deployment target cannot carry a Python runtime at all.

## Author

 - [Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)

## Acknowledgements

Special thanks to [Mohamed Chelali](https://mchelali.github.io) and [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug) for fruitful discussions.

## License

This project is licensed under the BSD-3-Clause License: see the [LICENSE](https://github.com/warith-harchaoui/audio-helper/blob/main/LICENSE) file for details.
