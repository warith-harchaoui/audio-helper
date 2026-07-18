# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.9] - 2026-07-18

### Documentation

- Complete Numpy-style docstrings and full type annotations across the
  package per CODING.md (API endpoints return types, CLI argparse
  handlers).

## [1.5.8] - 2026-07-15

### Documentation

- Harmonize README/LISEZMOI to the AI Helpers common structure (single
  H1, PyPI + source install paths, refreshed version pins to v1.5.8); no
  code changes.

## [1.5.7] - 2026-07-14

### Documentation

- Finalize suite wording: describe capabilities in plain language
  (Voice Activity Detection, Speech to Text, Speech Synthesis, source
  separation) instead of specific tool names, for consistency across the
  suite's descriptions and the documentation site.


## [1.5.6] - 2026-07-14

### Maintenance

- Apply the project coding standards across the package and `tests/`:
  Numpy-style docstrings on every function/class (including private and
  nested helpers), full type annotations with `from __future__ import
  annotations`, and comment density raised above the floor in every
  module. No public API or behavior changes.
- Route library logging through the os-helper logging surface
  (`osh.info/warning/error`) and adopt os-helper path/file utilities
  more widely; pin `os-helper>=1.5.0`.
- Refresh the project logo asset.


## [1.5.5] - 2026-07-08

### Documentation

- Cross-platform Install prerequisites (macOS / Ubuntu / Windows).

## [1.5.4] - 2026-07-07

### Fixed

- `load_audio` now decodes via ffmpeg (`sound_converter`) instead of
  libsndfile/`soundfile`, so AAC/`.m4a`, `.opus` and the audio track of
  video files load correctly (libsndfile could not open them). The
  intermediate float32 WAV is read with `scipy.io.wavfile`. `soundfile` is
  dropped as a dependency; `make_silence` writes via scipy too.
- Test fixtures now write WAVs via `scipy.io.wavfile` (no soundfile).

## [1.5.2] - 2026-07-07

### Documentation

- Establish suite-wide Python coding-style mandate in `CONTRIBUTING.md`:
  numpy-style docstrings on every function and class, module-level
  docstring header (with usage example + author), full type annotations,
  generous explanatory comments.
- `EXAMPLES.md` cookbook present at the repo root and linked from
  README + LISEZMOI.
- `print(...)` in docs (EXAMPLES.md / README / LISEZMOI) is followed by
  a `#`-comment showing the expected output (doctest / REPL style);
  library `.py` code uses `osh.info` / `osh.warning` / `osh.error`
  instead of bare `print`.
- Every `brew install <pkg>` mention is paired with a brew.sh hint when
  not already obvious from context.
- `.gitignore` updated to drop accidental `*config.json` commits while
  keeping `*config.json.example` templates tracked.

## [1.4.2] - 2026-06-29

### Changed

- Convert `pyproject.toml` from `[tool.poetry]` to PEP 621 `[project]`;
  switch build-backend to `setuptools`.
- Drop `setup.py` and `poetry.lock` (sole source of truth is
  `pyproject.toml`).
- Bump `os-helper` pin to `v1.3.0` (catches up; transitively exposes
  `verbosity()` setter and `profile_utils`).
- Trim `dev` extras to just `pytest>=8` (torch/torchaudio still
  available via `pip install -e .[demucs]`).
- Add GitHub Actions CI.

## [1.4.1] - 2026-06-28

### Added

- Comprehensive test suite + `EXAMPLES.md`.

## [1.4.0] - 2026-06-28

### Removed

- `audio_helper.streaming` module (moved to its own home in
  [`podcast-helper`](https://github.com/warith-harchaoui/podcast-helper)).
  Single owner for stream-to-PCM.

## [1.3.0] - 2026-06-28

### Changed

- Docstring cleanup; pin `os-helper` to `v1.2.0`.

## [1.2.0] - 2026-06-27

### Added

- `mix_room_tone` helper for natural-sounding silence insertion.

## [1.1.0] - 2026-06-23

### Added

- `streaming` module (Phase 6, later relocated to `podcast-helper`).

## [1.0.1] - 2026-05-22

### Removed

- Obsolete dependency configuration files.

## [1.0.0] - 2024-11-05

First tagged release.
