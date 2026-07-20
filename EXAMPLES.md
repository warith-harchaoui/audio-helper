# Audio Helper Examples

Practical recipes for the public surface of `audio-helper`. Every snippet
assumes:

```python
import audio_helper as ah
import os_helper as osh
```

and that `ffmpeg` is installed and on `PATH`. The optional `demucs`
extra (Torch + torchaudio) is only required for [Source Separation](#source-separation).

---

## Table of Contents

1. [Setup](#setup)
2. [Validity & Duration](#validity--duration)
3. [Load & Save](#load--save)
4. [Convert Formats](#convert-formats)
5. [Chunks](#chunks)
   - [Extract One Chunk](#extract-one-chunk)
   - [Split Regularly](#split-regularly)
   - [Concatenate](#concatenate)
6. [Silent Audio & Room Tone](#silent-audio--room-tone)
   - [Generate Silence](#generate-silence)
   - [Mix Room Tone](#mix-room-tone)
7. [Source Separation](#source-separation)
8. [Similarity](#similarity)
9. [Feature Extraction (Mel / MFCC)](#feature-extraction-mel--mfcc)

---

## Setup

Install with pip (replace the tag with the version you want):

```bash
pip install --force-reinstall --no-cache-dir \
    git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0
```

To enable Demucs-based source separation:

```bash
pip install --force-reinstall --no-cache-dir \
    "audio-helper[demucs] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
```

## Validity & Duration

`is_valid_audio_file` runs `ffprobe` *and* checks the file extension is
in the recognized audio/video list. `get_audio_duration` reads the
duration from the first audio stream.

```python
if ah.is_valid_audio_file("interview.mp3"):
    seconds = ah.get_audio_duration("interview.mp3")
    print(f"Interview is {seconds:.2f}s long.")
    # Interview is 1834.27s long.
```

A `.wav` extension on a text file fails the probe and returns False; a
valid WAV renamed to `.xyz` fails the extension check and also returns
False.

## Load & Save

`load_audio` returns `(signal, sample_rate)` where `signal` is a numpy
array by default. Use `to_mono=True` for mono, `two_channels=True` to
preserve stereo, and `target_sample_rate` to resample on the fly.

```python
audio, sr = ah.load_audio(
    "recording.wav",
    target_sample_rate=16000,   # resample to 16 kHz
    to_mono=True,               # downmix
    to_numpy=True,              # return numpy.ndarray (default for non-torch installs)
)
print(audio.shape, sr)          # (n_samples,) 16000
```

`save_audio` writes a numpy array (or torch tensor) to disk; the
extension dictates the container/codec.

```python
import numpy as np
sr = 16000
signal = (0.1 * np.random.randn(sr)).astype(np.float32)  # 1 s of noise
ah.save_audio(signal, "noise.wav", sample_rate=sr)
```

## Convert Formats

`sound_converter` is an ffmpeg wrapper covering format, sample rate,
and channel count in one call.

```python
ah.sound_converter(
    "speech.m4a",
    "speech.wav",
    freq=44100,
    channels=1,
    encoding="pcm_s16le",
    overwrite=True,
)
```

## Chunks

### Extract One Chunk

`extract_audio_chunk(input, start_s, end_s, output, overwrite=...)`
cuts a time-bounded slice and validates the output. Out-of-range
bounds raise `AssertionError`.

```python
ah.extract_audio_chunk("podcast.mp3", 60.0, 75.0, "highlight.mp3", overwrite=True)
```

### Split Regularly

`split_audio_regularly(source, output_folder, split_time_s, output_format=...)`
slices the source into fixed-duration chunks and returns the list of
chunk paths.

```python
chunks = ah.split_audio_regularly(
    "lecture.mp3",
    "lecture-chunks",
    split_time=30.0,
    output_format="mp3",
    overwrite=True,
)
print(f"{len(chunks)} chunks written.")
# 42 chunks written.
```

### Concatenate

`audio_concatenation([files], output, overwrite=...)` joins multiple
files head-to-tail. The container is dictated by the output extension.

```python
ah.audio_concatenation(
    ["intro.wav", "body.wav", "outro.wav"],
    "episode.mp3",
    overwrite=True,
)
```

## Silent Audio & Room Tone

### Generate Silence

```python
ah.generate_silent_audio(
    duration=5.0,
    output_audio_filename="pad.wav",
    sample_rate=44100,
    overwrite=True,
)
```

### Mix Room Tone

Adds a constant low-level ambient noise on top of a speech track so
silent gaps between cuts don't feel jarring. Default is pink noise at
−42 dB (inaudible but present).

```python
ah.mix_room_tone(
    "narration.wav",
    "narration-rt.wav",
    noise_db=-42.0,
    color="pink",              # white / pink / brown / blue / violet / velvet
    overwrite=True,
)
```

Use `color="brown"` and `noise_db=-38` for a warmer, slightly more
present room tone.

## Source Separation

`separate_sources` runs Demucs to produce vocals / drums / bass / other
stems. Requires the `[demucs]` extra (Torch + torchaudio).

```python
sources = ah.separate_sources(
    "song.mp3",
    output_folder="stems",
    device="cuda",        # or "cpu"; pass None to let it decide
    nb_workers=4,         # ignored when device != "cpu"
    output_format="mp3",
    overwrite=True,
)
print(sources)
# {'vocals': 'stems/vocals.mp3', 'drums': 'stems/drums.mp3',
#  'bass': 'stems/bass.mp3',   'other': 'stems/other.mp3'}
```

If Torch is not installed, the call raises an `ImportError` with the
install hint.

## Similarity

`sound_resemblance(a, b)` returns a score in `[0, 1]` based on MFCC
similarity. A file vs itself is ≈1.0; unrelated tones drop sharply.

```python
score = ah.sound_resemblance("original.wav", "reconstructed.mp3")
print(f"resemblance = {score:.3f}")
# resemblance = 0.974
```

## Feature Extraction (Mel / MFCC)

Low-level scipy-based helpers for building your own feature pipelines.

```python
import numpy as np
from audio_helper.main import hz_to_mel, mel_to_hz, mel_filter_banks, mfcc

# Hz <-> Mel
print(hz_to_mel(440.0), mel_to_hz(549.6))
# 549.6386500664797 440.00057651...

# Mel filter bank
sample_rate = 16000
fb = mel_filter_banks(num_filters=26, n_fft=512,
                      sample_rate=sample_rate, low_freq=0,
                      high_freq=sample_rate // 2)
print(fb.shape)   # (26, 257)

# MFCCs from a raw 1-D signal
signal = np.random.randn(sample_rate).astype(np.float32)
coefs = mfcc(signal, sample_rate, num_mfcc=13, n_fft=512)
print(coefs.shape)  # (n_frames, 13)
```

---

## Surfaces (CLI / API / MCP / GUI)

The same operations are reachable without writing Python.

**argparse CLI (always installed):**

```bash
audio-helper convert --input in.mp3 --output out.wav --freq 16000 --channels 1
audio-helper chunk   --input in.mp3 --start 3.0 --end 8.5 --output cut.mp3
audio-helper split   --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper concat  --inputs a.mp3 b.mp3 c.mp3 --output all.mp3
audio-helper roomtone --input speech.wav --output speech-rt.wav --db -42 --color pink
audio-helper separate --input mix.mp3 --output-dir stems/          # needs [demucs]
audio-helper resemblance --a take1.wav --b take2.wav
```

**click CLI twin (`[cli]` extra)** — identical flags:

```bash
pip install "audio-helper[cli]"
audio-helper-click convert --input in.mp3 --output out.wav --freq 16000
```

**FastAPI HTTP surface + GUI (`[api]` extra):**

```bash
pip install "audio-helper[api]"
uvicorn audio_helper.api:app --port 8000

# Convert over HTTP (multipart upload):
curl -F 'file=@in.mp3' -F 'output_format=wav' -F 'freq=16000' \
     -o out.wav http://localhost:8000/convert

# OpenAPI docs:  http://localhost:8000/docs
# Browser GUI:   http://localhost:8000/gui   (drop a file, pick an op, A/B the result)
```

**MCP tools (`[api,mcp]` extras):**

```bash
pip install "audio-helper[api,mcp]"
audio-helper-mcp                 # FastAPI + MCP on :8000
```

See [TRIGGERS.md](TRIGGERS.md) for the full catalogue of phrasings, commands,
and file types, and [skills/README.md](skills/README.md) to install `audio-helper`
as a Claude / OpenCode agent skill.
