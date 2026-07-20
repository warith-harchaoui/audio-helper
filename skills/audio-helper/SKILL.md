---
name: audio-helper
description: >-
  Process audio (and the audio track of video) files with the `audio-helper`
  toolkit — convert / re-encode formats and sample rates, extract a time slice,
  split into fixed-duration chunks, concatenate files, generate silence, mix
  room-tone ambient noise, score MFCC similarity between two clips, and separate
  a mix into vocals/drums/bass/other stems via Demucs. Exposed as a Python
  library (`import audio_helper as ah`), two CLIs (`audio-helper` argparse and
  `audio-helper-click`), a FastAPI HTTP surface, an MCP tool set, and a minimal
  browser GUI at `/gui`. Local-first, ffmpeg-backed, no SaaS.

  TRIGGER — any of: the user names an audio operation on a file ("convert this
  mp3 to wav / 16 kHz / mono", "re-encode / transcode this audio", "cut/trim/
  extract seconds A to B from this clip", "split this recording into 30-second
  chunks", "join/concatenate/merge these audio files", "generate N seconds of
  silence", "add room tone / background hiss / ambient noise to mask cuts",
  "separate vocals / isolate the vocals / extract stems / remove the music /
  karaoke / acapella", "how similar are these two recordings", "MFCC / audio
  fingerprint similarity"); the user types or references a command
  (`audio-helper`, `audio-helper-click`, `audio-helper-mcp`, subcommands
  `convert|duration|chunk|silence|concat|roomtone|split|separate|resemblance`)
  or a library function (`load_audio`, `sound_converter`, `extract_audio_chunk`,
  `split_audio_regularly`, `audio_concatenation`, `generate_silent_audio`,
  `mix_room_tone`, `sound_resemblance`, `separate_sources`, `get_audio_duration`,
  `is_valid_audio_file`); the user points at an audio file (`.mp3 .wav .m4a .flac
  .ogg .opus .aac .wma .aiff`) or a video file whose audio track is the target
  (`.mp4 .mkv .mov .webm …`) and wants it transformed; the user wants the audio
  API/MCP server run, or the drag-and-drop GUI; the user asks to install/run
  audio-helper.

  SKIP when: the task is speech-to-text / transcription / captions / subtitles
  (use vocal-helper / a whisper skill); text-to-speech / voice cloning /
  synthesis; music generation; downloading audio from YouTube or a URL (use
  youtube-helper); DAW-style editing (EQ, compression, mixing automation,
  fades beyond room-tone); loudness normalization / mastering; or pure signal-
  processing math with no file in play. audio-helper transforms audio *files*;
  it does not transcribe, synthesize, or fetch them.
---

# audio-helper — audio file operations toolkit

`audio-helper` is a small, local-first Python toolkit for preparing audio for
AI and media pipelines. Everything is ffmpeg-backed and file-oriented: you give
it paths, it writes paths. The same functions are reachable five ways (library,
two CLIs, HTTP API, MCP, GUI) so an agent can pick whichever fits.

## Before anything: verify it is installed

```bash
audio-helper --version            # argparse CLI (always installed with the pkg)
python -c "import audio_helper"   # library import check
```

If missing, install it (ffmpeg is a hard system dependency):

```bash
pip install audio-helper                 # core (convert/chunk/split/concat/…)
pip install 'audio-helper[demucs]'       # + source separation (torch, ~2 GB)
pip install 'audio-helper[cli]'          # + click CLI twin
pip install 'audio-helper[api,mcp]'      # + FastAPI HTTP surface + MCP tools
```

ffmpeg must be on PATH:
- macOS 🍎 : `brew install ffmpeg` (install `brew` via [brew.sh](https://brew.sh/))
- Ubuntu 🐧 : `sudo apt install ffmpeg`
- Windows 🪟 : `winget install Gyan.FFmpeg`

## The nine operations

Same names across the library, both CLIs, the API, and the MCP tools:

| Operation | CLI | Library function |
|-----------|-----|------------------|
| Re-encode format/rate/channels | `audio-helper convert` | `sound_converter` |
| Probe duration (seconds) | `audio-helper duration` | `get_audio_duration` |
| Extract `[start, end]` slice | `audio-helper chunk` | `extract_audio_chunk` |
| Generate silence | `audio-helper silence` | `generate_silent_audio` |
| Concatenate files | `audio-helper concat` | `audio_concatenation` |
| Mix room-tone noise | `audio-helper roomtone` | `mix_room_tone` |
| Split into fixed chunks | `audio-helper split` | `split_audio_regularly` |
| Demucs stem separation | `audio-helper separate` | `separate_sources` |
| MFCC similarity score | `audio-helper resemblance` | `sound_resemblance` |

Quick examples:

```bash
audio-helper convert --input in.mp3 --output out.wav --freq 16000 --channels 1
audio-helper chunk   --input in.mp3 --start 3.0 --end 8.5 --output cut.mp3
audio-helper split   --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper concat  --inputs a.mp3 b.mp3 c.mp3 --output all.mp3
audio-helper roomtone --input speech.wav --output speech-rt.wav --db -42 --color pink
audio-helper separate --input mix.mp3 --output-dir stems/          # needs [demucs]
audio-helper resemblance --a take1.wav --b take2.wav
```

```python
import audio_helper as ah
audio, sr = ah.load_audio("in.mp3", to_numpy=True, to_mono=True)
ah.sound_converter("in.mp3", "out.wav", freq=16000, channels=1)
chunks = ah.split_audio_regularly("in.mp3", "chunks/", split_time=30.0)
stems = ah.separate_sources("mix.mp3", output_folder="stems/")  # needs [demucs]
```

For the full flag matrix and every option, read `references/cli-reference.md`.
For the API / MCP / GUI surfaces (endpoints, ports, the `/gui` bench), read
`references/surfaces.md`. For the exhaustive, auditable trigger list, read
`references/triggers.md`.

## Rules of thumb

- **Pick the operation from the intent, not the file type.** "make it mono
  16 kHz" → `convert`; "just the first minute" → `chunk`; "cut it into pieces"
  → `split`; "stitch these together" → `concat`; "isolate the singer" →
  `separate`.
- **Video files are valid inputs.** The audio track of `.mp4/.mkv/.mov/.webm`
  decodes through ffmpeg — no need to pre-extract.
- **`separate` needs the `[demucs]` extra.** Without it the function raises a
  clear `ImportError`; install `audio-helper[demucs]` (pulls torch, ~2 GB) or
  tell the user why it is unavailable.
- **Defaults are safe:** output paths are auto-derived when omitted; `overwrite`
  is conservative (False) for generating/splitting, True for convert/chunk.
- **After running, report the output path(s)** the tool printed, and hand them
  back — do not re-run unless something failed.
- **Local only.** No network except the one-time Demucs model download; never
  sends audio to a SaaS.
