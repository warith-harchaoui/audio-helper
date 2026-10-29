# Landscape

[🇫🇷 PAYSAGE.md](https://github.com/warith-harchaoui/audio-helper/blob/main/PAYSAGE.md) · 🇬🇧 English

Related and competing Python libraries in the "manipulate audio files"
space, benchmarked against `audio-helper`. Ratings are ⭐ (1) to
⭐⭐⭐⭐⭐ (5), scored on `audio-helper`'s intended job: everyday audio
handling for AI pipelines (load, convert, split, concat, silence,
room-tone, source separation, MFCC similarity). A library optimised for
a very different job (e.g. music-information retrieval, real-time DSP)
is not penalised; the score just reflects fit to *this* niche.

## At a glance

<!-- TABLE:START -->
| Audio Wrangling | Multi-format I/O | Format conversion | Split / concat / silence | Spectral / MFCC features | Source separation | Correct resampling | Light install | AI-pipeline ergonomics |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **audio-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| pydub | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| librosa | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| soundfile | ⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| torchaudio | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| ffmpeg-python | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| sox / pysox | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| audioread | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| scipy.io.wavfile | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Demucs | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Spleeter | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
<!-- TABLE:END -->

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/landscape.png)

The map is a 2-D summary of the eight criteria, so read it as a shape, not a scoreboard. `audio-helper` is at the top-right corner. The axes read **Horizontal: Adaptable ↔ Ingenious** and **Vertical: Resourceful ↔ Flexible**.
<!-- FIGURE:END -->

## Positioning

`audio-helper` deliberately sits at the intersection of **pydub-level ergonomics**
(one-line load / convert / split / concat / silence) and **AI-pipeline
needs** (source separation on demand, MFCC similarity for A/B
comparisons). It intentionally does *not* try to compete with `librosa`
on the analysis side, and it keeps `torch` **optional**:
you only pay the ~2 GB torch/torchaudio cost if you actually call
`separate_sources` (Demucs ships behind the `[demucs]` extra). That
trade-off is the main differentiator against `torchaudio` (torch is
mandatory) and against `librosa` (analysis-first, thin on format
conversion and everyday I/O).

The nuance behind each rating is worth spelling out. `audio-helper`'s
multi-format I/O goes through ffmpeg for every decode, so it reads and
writes essentially anything ffmpeg understands, where `soundfile` is
bounded by libsndfile's own format list (WAV / FLAC / OGG / AIFF / CAF
and, since libsndfile 1.1, MP3), `audioread` only decodes, and
`scipy.io.wavfile` handles WAV alone. On features, `librosa` is the
MIR-grade reference and earns five stars; `audio-helper` exposes MFCC
similarity for A/B comparison, not a full analysis suite, hence its
middle rating.

Correct resampling splits the field into three groups, not two. The
libraries that own a real polyphase or windowed-sinc resampler rate
highest: `audio-helper` and `ffmpeg-python` through libswresample,
`sox` / `pysox` through SoX, `librosa` through `soxr`, `torchaudio`
through its own `Resample` transform. The libraries that simply do not
resample — `soundfile`, `audioread`, `scipy.io.wavfile` — score one
star, because the job comes back to you whole. `pydub` sits in between,
and it is the interesting case: `AudioSegment.set_frame_rate` calls the
standard library's `audioop.ratecv`, which interpolates between samples
instead of applying a designed anti-aliasing filter, so a downsample
aliases rather than refusing to.

On separation, Hybrid Demucs is the state of the art, and `audio-helper`
does not reach it through the `demucs` package: `separate_sources` runs
torchaudio's `HDEMUCS_HIGH_MUSDB_PLUS` bundle — the same architecture,
trained by Meta, shipped as a torchaudio pipeline. That is the whole
reason the `[demucs]` extra pulls `torch` and `torchaudio` and nothing
else. `Spleeter` is the other well-known option, and it has aged
differently: release 2.4.2 pins `tensorflow==2.12.1` together with
`numpy<2.0.0`, and declares `requires-python >=3.8,<3.12` — so it
installs neither next to a current NumPy nor on the Python versions
`audio-helper` supports.

## When to pick what

- **`audio-helper`**: audio prep for AI pipelines: batch conversions,
  chunking for windowed inference, silence and room-tone for
  post-production, MFCC-based similarity, Demucs on demand.
- **`librosa`**: analysis-heavy work (onset detection, beat tracking,
  chroma) that does not need arbitrary format conversion. Mind the
  floor: `librosa` 1.0.0 requires Python 3.12 or newer, so on 3.10 and
  3.11 you get the 0.11 line instead.
- **`torchaudio`**: you are already tensor-native and want zero-copy
  between audio I/O and your model.
- **`pydub`**: quick scripting, DJ-style cut/paste, no MFCC or
  separation needed. It has not had a release since March 2021, and the
  `audioop` module it is built on left the standard library in Python
  3.13, so on 3.13 it needs the `audioop-lts` back-port.
- **`soundfile`**: plain, fast read/write of what libsndfile supports,
  when you need neither conversion nor analysis.
- **`ffmpeg-python` / `sox` / `pysox`**: heavy format conversion and
  transcoding, driving the ffmpeg or SoX CLI yourself.
- **`Demucs` / `Spleeter`**: production source separation with your own
  wrapper around the underlying model. Prefer the upstream `demucs`
  package over this one when you need the stereo stems, a different
  Demucs checkpoint, or the six-stem model: `audio-helper` fixes one
  bundle and downmixes every stem to mono.

## Sources and as-of date

**As of 2026-10-27.** The stars above are a judgement call about fit to
`audio-helper`'s job, not a benchmark: nothing in the table was measured on a
shared corpus, and a library built for a different job is not marked down for
it. What *is* sourced is every factual claim in the prose.

Versions and dates were read from each project's PyPI metadata
(`https://pypi.org/pypi/<name>/json`) on 2026-10-27:

| Project | Latest release | Released |
| --- | --- | --- |
| `pydub` | 0.25.1 | 2021-03-10 |
| `librosa` | 1.0.0 | 2026-08-11 |
| `soundfile` | 0.14.0 | 2026-06-06 |
| `torchaudio` | 2.11.0 | 2026-03-23 |
| `ffmpeg-python` | 0.2.0 | 2019-07-06 |
| `sox` (pysox) | 1.5.0 | 2024-03-20 |
| `audioread` | 3.1.0 | 2025-10-26 |
| `scipy` | 1.18.1 | 2026-08-21 |
| `spleeter` | 2.4.2 | 2025-04-03 |
| `demucs` | 4.1.0 | 2026-07-11 |

Where the other claims come from:

- **Pins and Python support**, from the same metadata: `spleeter` 2.4.2
  declares `tensorflow==2.12.1`, `numpy<2.0.0` and
  `requires-python >=3.8,<3.12`; `librosa` 1.0.0 declares
  `requires-python >=3.12` and depends on `soxr>=1.0.0`.
- **`pydub`'s resampling**, from the installed source of
  `pydub.AudioSegment.set_frame_rate` (0.25.1), which calls
  `audioop.ratecv`. `audioop` was removed from the standard library in
  Python 3.13 by PEP 594, which is why the `audioop-lts` back-port exists.
- **`soundfile`'s format list**, from `soundfile.available_formats()` on
  soundfile 0.13.1 / libsndfile 1.2.2, where `MP3` is present.
- **The separation model**, from `audio_helper/main.py`
  (`separate_sources` → `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`) and
  from torchaudio's `pipelines/_source_separation_pipeline.py`. The weights
  are fetched by `torchaudio.utils.download_asset` from
  `https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt`;
  a HEAD request on that URL answers 334 697 255 bytes, which is the ~335 MB
  the README quotes.
- **`audio-helper`'s own ratings**, from this repository at version 2.5.0.

Deliberately left as opinion rather than sourced: the *AI-pipeline
ergonomics* column, and the order of the recommendations above.

## Ideas worth stealing

Researched rather than guessed: what each neighbour does better, and what this library would have to change to match it. Nothing here is a commitment — it is the reading list the next version starts from.

- **`librosa` — the spectral feature vocabulary** — `sound_resemblance` here computes an MFCC similarity, but librosa's framing — mel filter banks, hop and window as explicit parameters, a documented feature set — is the reference every reader already knows. Naming the parameters the same way costs nothing and removes a translation step.

- **`pydub` — an API that reads like the edit you are describing** — Slicing a segment with `[]` and concatenating with `+` reads better than a function call for ad-hoc work. The function-call surface here is right for pipelines; a thin fluent wrapper over it would serve the notebook case without changing the core.

- **`soundfile` — ask libsndfile what it supports, do not hardcode it** — `available_formats()` reports the formats actually compiled into the installed libsndfile. The equivalent here is asking the installed ffmpeg rather than shipping a list that drifts from whatever the user has.

- **`sox` — effects as a composable chain** — Normalise, compand, dither and fade are separate steps you stack. Room tone is already this shape; the rest of the post-production verbs would compose the same way.
