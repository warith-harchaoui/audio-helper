---
name: audio-helper
description: >-
  Transform audio files — and the audio track of video files — locally with
  ffmpeg and a local Demucs. Nine operations: convert / re-encode (format,
  sample rate, mono-stereo), measure duration, cut a [start, end] slice, split
  into fixed-length chunks, concatenate, generate silence, mix room tone over
  edits, separate stems (vocals / drums / bass / other), and score MFCC
  similarity between two clips. Use when the user says "convert this mp3 to
  wav", "make it mono 16 kHz", "resample", "extract the audio from this mp4",
  "trim / cut from 3s to 8s", "split into 30-second chunks", "merge these
  files", "generate silence", "add room tone", "isolate the vocals",
  "karaoke / acapella", "remove the music", "how similar are these two takes",
  "how long is this audio", or names `audio-helper`, `audio-helper-click`,
  `audio-helper-mcp`, `import audio_helper as ah`. Do NOT use for transcription
  or subtitles (vocal-helper), text-to-speech, music generation, downloading
  audio (youtube-helper / podcast-helper), or DAW-style mixing.
license: BSD-3-Clause
metadata:
  author: Warith Harchaoui
  package: audio-helper
  suite: ai-helpers
  homepage: https://deraison.ai/ai-helpers
---

# audio-helper — local ffmpeg audio transforms

`audio-helper` does one job: **transform audio files**. It never transcribes,
never synthesizes, never downloads. Everything runs on the machine through
ffmpeg, with a local Demucs for stem separation. The same nine operations are
reachable five ways (library, two CLIs, HTTP API, MCP, browser GUI).

## Before anything: verify it is installed

```bash
audio-helper --help               # argparse CLI (ships with the package)
python -c "import audio_helper"   # library import check
ffmpeg -version                   # hard system dependency
```

Install:

```bash
pip install audio-helper                 # core: the eight ffmpeg operations
pip install 'audio-helper[demucs]'       # + stem separation (torch, ~2 GB)
pip install 'audio-helper[cli]'          # + the click twin
pip install 'audio-helper[api]'          # + FastAPI HTTP surface and GUI
pip install 'audio-helper[mcp]'          # + MCP tools over the same app
```

ffmpeg must be on PATH — macOS 🍎 `brew install ffmpeg`, Ubuntu 🐧 `sudo apt
install ffmpeg`, Windows 🪟 `winget install Gyan.FFmpeg`.

## The nine operations

| Intent | CLI | Library | API |
|---|---|---|---|
| Re-encode format / rate / channels | `audio-helper convert` | `sound_converter` | `POST /convert` |
| Duration in seconds | `audio-helper duration` | `get_audio_duration` | `POST /duration` |
| Extract a `[start, end]` slice | `audio-helper chunk` | `extract_audio_chunk` | `POST /chunk` |
| Split into fixed-length chunks | `audio-helper split` | `split_audio_regularly` | `POST /split` |
| Concatenate files | `audio-helper concat` | `audio_concatenation` | `POST /concat` |
| Generate silence | `audio-helper silence` | `generate_silent_audio` | `POST /silence` |
| Mix room tone over edits | `audio-helper roomtone` | `mix_room_tone` | `POST /roomtone` |
| Separate stems (Demucs) | `audio-helper separate` | `separate_sources` | `POST /separate` |
| MFCC similarity score | `audio-helper resemblance` | `sound_resemblance` | `POST /resemblance` |

Quick examples:

```bash
audio-helper convert talk.mp3 talk.wav --sample-rate 16000 --channels 1
audio-helper chunk interview.wav out.wav --start 3 --end 8
audio-helper split lecture.m4a ./chunks --duration 30
audio-helper separate song.mp3 ./stems          # needs [demucs]
audio-helper resemblance take1.wav take2.wav    # 0..1 similarity
```

```python
import audio_helper as ah

ah.sound_converter("talk.mp3", "talk.wav", freq=16000, channels=1)
dur = ah.get_audio_duration("talk.wav")
stems = ah.separate_sources("song.mp3", output_folder="./stems")
```

**Video files are valid input.** ffmpeg decodes the audio track of
`.mp4/.mkv/.mov/.webm/.avi…` directly — do not pre-extract with another tool.

For the full flag matrix and the output contract, read
`references/cli-reference.md`. For the HTTP API, MCP tools and the Recipe
Canvas GUI, read `references/surfaces.md`. For the exhaustive trigger list,
read `references/triggers.md`.

## Rules of thumb

- **Pick the operation from the verb.** "convert/resample/transcode" →
  `convert`. "cut/trim/crop a range" → `chunk`. "chop into pieces" → `split`.
  "merge/join/stitch" → `concat`. "isolate vocals / karaoke" → `separate`.
- **16 kHz mono is the suite's speech convention.** When the output feeds a
  speech pipeline, `convert --sample-rate 16000 --channels 1` first.
- **`separate` is the only heavy operation.** It needs `[demucs]` and downloads
  model weights once on first run; after that it is fully offline. Without the
  extra it raises a clear `ImportError` — install it, do not fall back to a
  SaaS.
- **`resemblance` is MFCC distance, not identity.** It answers "are these the
  same take", not "is this the same speaker" — speaker identity is
  vocal-helper's job.
- **Room tone is for masking edits**, not for denoising; audio-helper has no
  EQ, compressor, or loudness normalizer, and adding one is out of scope.
- **After running, report the output path and the measured duration** — do not
  re-run to confirm.
