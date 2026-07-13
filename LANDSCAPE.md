# LANDSCAPE

Related and competing Python libraries in the "manipulate audio files"
space, benchmarked against `audio-helper`. Ratings are `⭐️` (1) to
`⭐️⭐️⭐️⭐️⭐️` (5), scored on `audio-helper`'s intended job — everyday audio
handling for AI pipelines (load, convert, split, concat, silence,
room-tone, source separation, MFCC similarity). A library optimised for
a very different job (e.g. music-information retrieval, real-time DSP)
is not penalised — the score just reflects fit to *this* niche.

## At a glance

| Library / project | Multi-format I/O (ffmpeg fallback) | Format conversion | Split / concat / silence / room-tone | MFCC / spectral features | Source separation (Demucs/Spleeter) | Light install (no torch by default) | AI-pipeline ergonomics (`dict` return, path-based API) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **audio-helper** *(this project)* | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ (Demucs via `[demucs]` extra) | ⭐️⭐️⭐️⭐️⭐️ (torch optional) | ⭐️⭐️⭐️⭐️⭐️ |
| librosa | ⭐️⭐️⭐️ (via `audioread`/`soundfile`) | ⭐️⭐️ (writes limited) | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ (not built-in) | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ |
| torchaudio | ⭐️⭐️⭐️⭐️ (soundfile / ffmpeg backends) | ⭐️⭐️⭐️ | ⭐️⭐️ (slice tensors, no concat helper) | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ (HDEMUCS via `pipelines`) | ⭐️ (torch is required) | ⭐️⭐️⭐️ (tensor-native, not path-native) |
| pydub | ⭐️⭐️⭐️⭐️ (ffmpeg-backed) | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ (chainable segments) | ⭐️ | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ |
| soundfile | ⭐️⭐️ (WAV / FLAC / OGG only, no ffmpeg) | ⭐️ | ⭐️ (raw I/O primitive) | ⭐️ | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ |
| pyAudioAnalysis | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️ |
| essentia | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ (MIR-grade) | ⭐️⭐️ | ⭐️⭐️ (C++ build) | ⭐️⭐️ |
| madmom | ⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ (beat / onset) | ⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️ |
| Demucs (standalone CLI) | ⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️ | ⭐️ | ⭐️⭐️⭐️⭐️⭐️ (state-of-the-art) | ⭐️ (torch mandatory) | ⭐️⭐️ (CLI-first) |
| Spleeter | ⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️ | ⭐️ | ⭐️⭐️⭐️⭐️ (TF-based, unmaintained) | ⭐️ (TensorFlow mandatory) | ⭐️⭐️ |

## Positioning

`audio-helper` deliberately sits at the intersection of **pydub-level ergonomics**
(one-line load / convert / split / concat / silence) and **AI-pipeline
needs** (Demucs source separation on demand, MFCC similarity for A/B
comparisons). It intentionally does *not* try to compete with `librosa`
or `essentia` on the analysis side, and it keeps `torch` **optional** —
you only pay the ~2 GB torch/torchaudio cost if you actually call
`separate_sources`. That trade-off is the main differentiator against
`torchaudio` (torch is mandatory) and against `librosa` (no source
separation).

## When to pick what

- **`audio-helper`** — audio prep for AI pipelines: batch conversions,
  chunking for windowed inference, silence and room-tone for
  post-production, MFCC-based similarity, Demucs on demand.
- **`librosa`** — analysis-heavy work (onset detection, beat tracking,
  chroma) that does not need arbitrary format conversion.
- **`torchaudio`** — you are already tensor-native and want zero-copy
  between audio I/O and your model.
- **`pydub`** — quick scripting, DJ-style cut/paste, no MFCC or
  separation needed.
- **`Demucs` / `Spleeter`** — production source separation with your own
  wrapper around the underlying model.
- **`essentia` / `madmom`** — music-information retrieval,
  beat/downbeat/tempo estimation.
