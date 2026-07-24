# Landscape

[🇫🇷 PAYSAGE.md](https://github.com/warith-harchaoui/audio-helper/blob/main/PAYSAGE.md) · 🇬🇧 English

Related and competing Python libraries in the "manipulate audio files"
space, benchmarked against `audio-helper`. Ratings are ⭐ (1) to
⭐⭐⭐⭐⭐ (5), scored on `audio-helper`'s intended job — everyday audio
handling for AI pipelines (load, convert, split, concat, silence,
room-tone, source separation, MFCC similarity). A library optimised for
a very different job (e.g. music-information retrieval, real-time DSP)
is not penalised — the score just reflects fit to *this* niche.

## At a glance

<!-- TABLE:START -->
| Audio Wrangling | Multi-format I/O | Format conversion | Split / concat / silence | MFCC / spectral features | Source separation | Light install | AI-pipeline ergonomics |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **audio-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| librosa | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| torchaudio | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| pydub | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| soundfile | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| pyAudioAnalysis | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| essentia | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| madmom | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Demucs | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
| Spleeter | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ |
<!-- TABLE:END -->

## Positioning map

<!-- FIGURE:START -->
2D representation of the table above.

![Positioning map](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/landscape.png)

The map is a 2-D summary of the seven criteria, so read it as a shape, not a scoreboard. `audio-helper` is at the top-right corner. The axes read **Horizontal — Clarity ↔ Versatile** and **Vertical — Efficient ↔ Comprehensive**.
<!-- FIGURE:END -->

## Positioning

`audio-helper` deliberately sits at the intersection of **pydub-level ergonomics**
(one-line load / convert / split / concat / silence) and **AI-pipeline
needs** (source separation on demand, MFCC similarity for A/B
comparisons). It intentionally does *not* try to compete with `librosa`
or `essentia` on the analysis side, and it keeps `torch` **optional** —
you only pay the ~2 GB torch/torchaudio cost if you actually call
`separate_sources` (Demucs ships behind the `[demucs]` extra). That
trade-off is the main differentiator against `torchaudio` (torch is
mandatory) and against `librosa` (no source separation).

The nuance behind each rating is worth spelling out. `audio-helper`'s
multi-format I/O leans on an ffmpeg fallback, so it reads and writes
essentially anything ffmpeg understands, where `soundfile` covers only
WAV / FLAC / OGG and `librosa`'s write path is limited. On features,
`librosa`, `essentia`, `pyAudioAnalysis` and `madmom` are MIR-grade and
earn five stars — `audio-helper` exposes MFCC similarity for A/B
comparison, not a full analysis suite, hence its middle rating. On
separation, `Demucs` is state-of-the-art and `audio-helper` wraps it
directly; `torchaudio` reaches similar quality through its HDEMUCS
pipeline, while `Spleeter` is TensorFlow-based and unmaintained.

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
- **`essentia` / `madmom` / `pyAudioAnalysis`** — music-information
  retrieval, beat/downbeat/tempo estimation.
</content>
</invoke>
