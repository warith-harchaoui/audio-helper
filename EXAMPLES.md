# Audio Helper Examples

[🇫🇷 EXEMPLES.md](https://github.com/warith-harchaoui/audio-helper/blob/main/EXEMPLES.md) · 🇬🇧 English

Practical recipes for the public surface of `audio-helper`. Every snippet below
was run against one file — [example.mp3](https://deraison.ai/example.mp3), 60
seconds of a badly recorded JFK speech, MP3 at 48 kHz stereo — and the output
comments are the output that came back. Download it next to your script and the
whole page replays.

Two things every recipe assumes: `ffmpeg` is installed and on `PATH`, and the
imports in [Setup](#setup) have been run. Only
[recipe 7](#7-separate-a-mix-into-vocals-drums-bass-and-other) needs the
optional `demucs` extra.

---

## Table of Contents

1. [Setup](#setup)
2. [Validate a file and read its duration](#1-validate-a-file-and-read-its-duration)
3. [Load a recording and save one back](#2-load-a-recording-and-save-one-back)
4. [Convert format, rate and channels](#3-convert-format-rate-and-channels)
5. [Cut one highlight out of a long recording](#4-cut-one-highlight-out-of-a-long-recording)
6. [Split into fixed windows and rejoin them](#5-split-into-fixed-windows-and-rejoin-them)
7. [Pad with silence and hide the cuts under room tone](#6-pad-with-silence-and-hide-the-cuts-under-room-tone)
8. [Separate a mix into vocals, drums, bass and other](#7-separate-a-mix-into-vocals-drums-bass-and-other)
9. [Score how close two takes sound](#8-score-how-close-two-takes-sound)
10. [Build your own Mel and MFCC features](#9-build-your-own-mel-and-mfcc-features)
11. [Command line](#command-line)
12. [HTTP API, MCP and the browser GUI](#http-api-mcp-and-the-browser-gui)
13. [Troubleshooting](#troubleshooting)

---

## Setup

```bash
# Core: ffmpeg-backed I/O, convert, cut, split, concat, silence, room tone, MFCC
pip install audio-helper

# Adds source separation — torch + torchaudio, about 2 GB
pip install "audio-helper[demucs]"
```

Every later snippet starts from these imports:

```python
import os

import numpy as np

import audio_helper as ah
import os_helper as osh
```

`os_helper` is not decoration: `audio-helper` logs through it, and it owns the
file management the audio functions deliberately leave alone. Two calls are
worth knowing before anything else:

```python
# 1 = INFO, so audio-helper's own progress lines show up; 0 = WARNING (the
# default) also makes ffmpeg quiet, because the library passes quiet=True
# whenever verbosity() <= 0.
osh.verbosity(1)

# None of the writing functions create their destination folder. This one does.
osh.make_directory("out")
```

## 1. Validate a file and read its duration

`is_valid_audio_file` runs `ffprobe` *and* checks the extension is in the
recognised audio/video list — a text file renamed `.wav` fails the probe, a
valid WAV renamed `.xyz` fails the extension check, and both return `False`
rather than raising. `get_audio_duration` reads the duration of the first audio
stream.

```python
print(ah.is_valid_audio_file("example.mp3"))
# True

print(f"{ah.get_audio_duration('example.mp3'):.2f} s")
# 60.00 s
```

Video containers count as valid: every operation here works on the audio track
of an `.mp4` or a `.mkv`.

## 2. Load a recording and save one back

`load_audio` returns `(signal, sample_rate)`. Decoding always goes through
ffmpeg, never libsndfile, so `.m4a`, `.opus`, `.webm` and video containers load
like anything else.

```python
audio, sr = ah.load_audio(
    "example.mp3",
    target_sample_rate=16000,  # resample on the way out
    to_mono=True,  # downmix (this is the default)
    to_numpy=True,  # hand back a numpy array
)
print(audio.shape, sr, audio.dtype)
# (960000,) 16000 float32
```

**Mind the default.** `to_numpy=False` is the default, and it returns a
*torch* tensor — which means the plain `ah.load_audio(path)` call needs the
`[demucs]` extra and raises `ImportError` without it.

```python
tensor, native_sr = ah.load_audio("example.mp3")
print(type(tensor).__name__, tuple(tensor.shape), native_sr)
# Tensor (2880000,) 48000
```

Keeping both channels gives `(time, channels)` in numpy, and `(channels, time)`
in torch:

```python
stereo, sr = ah.load_audio("example.mp3", to_mono=False, two_channels=True, to_numpy=True)
print(stereo.shape, sr)
# (2880000, 2) 48000
```

`save_audio` writes a numpy array or a torch tensor; the extension picks the
container and codec.

```python
beep = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
ah.save_audio(beep, "beep.wav", sample_rate=16000)
print(f"{ah.get_audio_duration('beep.wav'):.2f} s")
# 1.00 s
```

## 3. Convert format, rate and channels

`sound_converter` is the one ffmpeg call that covers container, sample rate,
channel count and codec. It returns the output path, and unlike most of its
neighbours it defaults to `overwrite=True`.

```python
print(
    ah.sound_converter(
        "example.mp3", "speech.wav", freq=16000, channels=1, encoding="pcm_s16le", overwrite=True
    )
)
# speech.wav
```

Resampling is the part worth paying attention to: it happens inside ffmpeg's
`libswresample`, so a downsample is band-limited first and does not alias. That
is the same path `load_audio(target_sample_rate=...)` takes.

For a lossy container, `bitrate` states the one thing the arguments above
cannot. It takes bits per second (`64000`) or an ffmpeg-style string
(`"64k"`).

Left unset, **the input's own bitrate is carried over**: a 96 kb/s episode
re-containered to `.m4a` stays 96 kb/s, because the caller named a container
and not a quality. An uncompressed input has no compressed bitrate to carry,
and `audio-helper` will not invent one, so ffmpeg chooses there, and for a
24 kHz mono stream it chooses 32 kb/s. That is the case where naming the
bitrate is worth the keystrokes.

```python
print(
    ah.sound_converter(
        "narration.wav",
        "narration.mp3",
        freq=24000,
        channels=1,
        bitrate="64k",
    )
)
# narration.mp3
```

```python
# Re-containering keeps the quality it was handed.
print(ah.sound_converter("episode.mp3", "episode.m4a"))
# episode.m4a
```

Which stage each setting lands on is worth knowing, because the function
encodes twice. `freq` and `channels` are applied to the intermediate WAV, so
the final container inherits them. A lossy `encoding` is applied to the final
encode instead, with `bitrate` alongside it: naming a lossy codec on the
intermediate WAV would write compressed payload inside a WAV container, which
the last stage would then decode and re-encode. A WAV output carries no
bitrate, so the setting is ignored there with a warning.

## 4. Cut one highlight out of a long recording

`extract_audio_chunk(audio_file, start_time, end_time, output_audio_filename)`
cuts a time-bounded slice, in seconds, and validates what it wrote. Bounds
outside the file raise `AssertionError`.

```python
print(ah.extract_audio_chunk("example.mp3", 10.0, 25.0, "highlight.mp3", overwrite=True))
# highlight.mp3

print(f"{ah.get_audio_duration('highlight.mp3'):.2f} s")
# 15.00 s
```

Leave the output path out and the slice lands beside its input, named from the
bounds **in milliseconds** — `example.mp3` cut at `[0, 2]` becomes
`example_chunk-0-2000.mp3` — and the returned path is absolute.

## 5. Split into fixed windows and rejoin them

`split_audio_regularly` is the chunker for windowed inference. It is also one
of the two functions here that creates its own output folder.

```python
chunks = ah.split_audio_regularly(
    "example.mp3", "windows", split_time=25.0, output_format="mp3", overwrite=True
)
print(len(chunks))
# 3

print([os.path.basename(c) for c in chunks])
# ['chunk_0000_split.mp3', 'chunk_0001_split.mp3', 'chunk_0002_split.mp3']

print([round(ah.get_audio_duration(c), 2) for c in chunks])
# [25.0, 25.0, 10.0]
```

The last window is whatever is left over — 60 s at 25 s gives 25 + 25 + 10, not
three padded windows. Returned paths are absolute, and the `_split` part of the
name is the `suffix=` argument.

`audio_concatenation` joins files head to tail, in the order given; the output
extension dictates the container.

```python
joined = ah.audio_concatenation(chunks, "rejoined.mp3", overwrite=True)
print(joined, f"{ah.get_audio_duration(joined):.2f} s")
# rejoined.mp3 60.00 s
```

## 6. Pad with silence and hide the cuts under room tone

`generate_silent_audio` writes digital silence, for padding a montage to a
fixed length or spacing two takes apart.

```python
pad = ah.generate_silent_audio(
    duration=5.0, output_audio_filename="pad.wav", sample_rate=44100, overwrite=True
)
print(pad, f"{ah.get_audio_duration(pad):.2f} s")
# pad.wav 5.00 s
```

Digital silence between two cuts is exactly what makes an edit audible, which
is what `mix_room_tone` is for: a constant ambience about 40 dB below the
speech masks the boundary while staying under the conscious hearing threshold.

```python
rt = ah.mix_room_tone(
    "speech.wav", "speech-roomtone.wav", noise_db=-42.0, color="pink", overwrite=True
)
print(rt, f"{ah.get_audio_duration(rt):.2f} s")
# speech-roomtone.wav 60.00 s
```

`color` takes ffmpeg's `anoisesrc` palette — `white`, `pink`, `brown` (alias
`red`), `blue`, `violet`, `velvet`. Pink at −42 dB is the default; `brown` at
−38 dB reads warmer and more present. Omit the output path and the result lands
beside the input as `<name>-roomtone.<ext>`, so two calls on the same input
write to the same place.

## 7. Separate a mix into vocals, drums, bass and other

`separate_sources` runs **Hybrid Demucs**, reached through torchaudio's
`HDEMUCS_HIGH_MUSDB_PLUS` pipeline rather than through the standalone `demucs`
package. It needs the `[demucs]` extra, and the first call downloads about
335 MB of weights from `download.pytorch.org` into the PyTorch Hub cache.
**This is the one recipe on this page that was not executed** — it is verified
against the code only, which is also why no output is quoted below.

```python
sources = ah.separate_sources(
    "song.mp3",
    output_folder="stems",
    device="cpu",  # or "cuda"; None lets it decide
    nb_workers=4,  # thread count, ignored unless device == "cpu"
    output_format="mp3",
    overwrite=True,
)

for stem, path in sources.items():
    print(stem, path)
```

What the code guarantees:

- The returned dict is keyed `drums`, `bass`, `other`, `vocals`, in that order
  — the order the torchaudio bundle declares its sources in, neither
  alphabetical nor vocals-first.
- Each file is `<output_folder>/<stem>.<output_format>`. Pass
  `output_folder=None` and the stems land in the input's own folder.
- Every stem is written **mono**. The model separates in stereo; the writer
  downmixes. If you need stereo stems, drive upstream `demucs` yourself.
- Without `torch` installed the call raises `ImportError` carrying the install
  hint, before any file is touched.
- It works on speech as well as on music, which is the useful trick: `vocals`
  on a noisy interview is a de-noised voice track.

## 8. Score how close two takes sound

`sound_resemblance(a, b)` resamples both files to 24 kHz mono, zero-pads the
shorter one, computes MFCCs and returns the cosine similarity of the two
matrices, in `[0, 1]`.

```python
print(f"{ah.sound_resemblance('speech.wav', 'speech-roomtone.wav'):.3f}")
# 0.978

print(f"{ah.sound_resemblance('speech.wav', 'speech.wav'):.3f}")
# 1.000
```

The room tone is fresh random noise on every pass, so that first figure moves
in its fourth decimal from one run to the next; at three decimals it held
steady across six.

Read it as a **sensitivity instrument for near-identical takes**, not as a
same-or-different classifier. The same measurement on genuinely unrelated
material stays high:

```python
print(f"{ah.sound_resemblance('speech.wav', 'beep.wav'):.3f}")
# 0.895

print(f"{ah.sound_resemblance('speech.wav', 'pad.wav'):.3f}")
# 0.912
```

A 440 Hz sine scores 0.895 against speech, and five seconds of digital silence
scores 0.912 — higher than the sine. The reason is in the implementation: the
cosine similarity runs over the raw MFCC matrices with no mean removal, so
coefficient 0, which is overall log-energy, dominates every comparison. Use the
score to confirm that a re-encode, a room-tone pass or a round-trip did not
damage a file. Do not use it to decide whether two files are the same recording.

## 9. Build your own Mel and MFCC features

The scipy-based primitives live in `audio_helper.main` and are not re-exported
on the package root, so import them explicitly.

```python
from audio_helper.main import hz_to_mel, mel_to_hz, mel_filter_banks, mfcc

print(hz_to_mel(440.0), mel_to_hz(549.6386753811499))
# 549.6386753811499 440.0

fb = mel_filter_banks(num_filters=26, n_fft=512, sample_rate=16000, low_freq=0, high_freq=8000)
print(fb.shape)
# (26, 257)

# `audio` is the 16 kHz mono array from recipe 2 — 60 s of speech.
coefs = mfcc(audio, 16000, num_mfcc=13, n_fft=512)
print(coefs.shape)
# (5999, 13)
```

The framing is fixed at 25 ms windows with a 10 ms hop, which is where the
frame count comes from, and the pipeline is the textbook one: pre-emphasis at
0.97, Hamming window, power spectrum, Mel filter bank, dB, DCT-II, keep the
first `num_mfcc` coefficients.

## Command line

One command line, built with click, installed with the package — no extra
needed. Flags are named rather than positional, because these commands take
several paths and a bare order is easy to get backwards.

```bash
$ audio-helper --help          # -h works too; -V prints the version
$ audio-helper duration    --input example.mp3
60.000000

$ audio-helper convert     --input example.mp3 --output out.wav --freq 16000 --channels 1
out.wav

$ audio-helper chunk       --input example.mp3 --start 3.0 --end 8.5 --output cut.mp3
cut.mp3

$ audio-helper split       --input example.mp3 --output-dir chunks/ --seconds 20 --overwrite
/path/to/chunks/chunk_0000_split.mp3
/path/to/chunks/chunk_0001_split.mp3
/path/to/chunks/chunk_0002_split.mp3

$ audio-helper concat      --inputs chunks/chunk_0000_split.mp3 chunks/chunk_0001_split.mp3 \
                           --output all.mp3 --overwrite
all.mp3

$ audio-helper roomtone    --input out.wav --output out-rt.wav --db -42 --color pink --overwrite
out-rt.wav

$ audio-helper silence     --duration 2 --output pad.wav --overwrite
pad.wav

$ audio-helper resemblance --a out.wav --b cut.mp3
0.877206

$ audio-helper separate    --input mix.mp3 --output-dir stems/   # needs [demucs]
```

Single-file commands print the path, `split` prints one absolute path per line,
`separate` prints JSON — so each one pipes into the next. `concat` accepts both
spellings: the list above, or the flag repeated as
`--inputs a.mp3 --inputs b.mp3`. `separate` is the one subcommand not run for
this page, for the same weight-download reason as recipe 7; without `torch` it
exits 2 with a one-line `error:` instead of a traceback.

The `--overwrite` flags above are not noise: `split`, `concat`, `silence` and
`roomtone` default to `--no-overwrite`, while `convert` and `chunk` default to
`--overwrite`.

## HTTP API, MCP and the browser GUI

The same functions, over HTTP, behind the `[api]` extra:

```bash
pip install "audio-helper[api]"
uvicorn audio_helper.api:app --port 8000
```

```bash
$ curl -s http://localhost:8000/health
{"status":"ok"}

$ curl -s -F 'file=@example.mp3' http://localhost:8000/duration
{"duration_seconds":60.0}

# Binary in, binary out: the converted file is the response body.
$ curl -s -F 'file=@example.mp3' -F 'output_format=wav' -F 'freq=16000' \
       -o out.wav http://localhost:8000/convert
$ ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
          -of default=noprint_wrappers=1 out.wav
codec_name=pcm_s16le
sample_rate=16000
channels=1
```

Every library operation has a route: `/convert`, `/duration`, `/chunk`,
`/silence`, `/concat`, `/roomtone`, `/split` (which returns a zip of the
chunks), `/separate`, `/resemblance`. Uploads are multipart; OpenAPI docs are
at `/docs`.

Two more surfaces sit on that same app. `GET /gui` serves the **Recipe
Canvas** — chain the verbs into a pipeline, hear every intermediate step,
export the result as a committable `recipe.yaml`; see
[GUI.md](https://github.com/warith-harchaoui/audio-helper/blob/main/GUI.md).
And `audio-helper-mcp` (the `[mcp]` extra) republishes the app's routes as MCP
tools for an agent host:

```bash
pip install "audio-helper[mcp]"
audio-helper-mcp
# same app, plus an /mcp endpoint
```

## Troubleshooting

**`FileNotFoundError`, or an `ffmpeg.Error`, when writing.** The destination
folder does not exist. `split_audio_regularly` and `separate_sources` create
their output folder; `sound_converter`, `extract_audio_chunk`,
`generate_silent_audio`, `audio_concatenation`, `mix_room_tone` and
`save_audio` all expect it to be there already. Call `osh.make_directory(...)`
first.

**`ImportError: This feature requires the optional 'demucs' extra.`** Either
`separate_sources`, or `load_audio` without `to_numpy=True` — the default
return type is a torch tensor. Add `to_numpy=True` if all you wanted was an
array.

**Nothing happened and the file is unchanged.** `overwrite=False` does not mean
"never write", it means "skip if up to date": the output is reused only while
it is newer than every input. Touch an input and the next call rebuilds it. If
you want the work redone unconditionally, pass `overwrite=True`.

**ffmpeg is noisy, or silent when you want to see it.** Both are
`osh.verbosity()`. At `0` (the default, WARNING) the library runs ffmpeg with
`quiet=True`; at `1` or more you get audio-helper's own INFO lines, and ffmpeg
stops being muted.

**`ffmpeg is required to decode audio but was not found on PATH`.** The message
is literal. `brew install ffmpeg` or `apt install ffmpeg`, then retry.

See [TRIGGERS.md](https://github.com/warith-harchaoui/audio-helper/blob/main/TRIGGERS.md)
for the full catalogue of phrasings, commands and file types.
