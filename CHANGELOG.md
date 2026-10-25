# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2026-10-25

### Changed

- **One command line, built with click.** `audio-helper` now runs the click
  implementation; the `audio-helper-click` twin and the argparse CLI are gone,
  together with the `[cli]` extra. A flag naming the parsing library described
  the implementation, not the interface, and two CLIs over the same nine
  operations only gave drift somewhere to hide.

  `cli_click.py` is now `cli.py`; `cli_argparse.py` is deleted. Nothing
  crossed the module boundary between them — both imported the library
  functions directly and nothing else — so the deletion took no helper with
  it. `console_entry()` is renamed `main()`, matching every other member's
  console entry point.

- `click` moved from the `[cli]` extra into the core `dependencies`: the CLI
  is not an optional surface any more, so its dependency cannot be optional
  either. It is also dropped from `[dev]`, where naming it a second time would
  only let the two pins drift.

- `os-helper` cap widened to `<4` (os-helper 3.0.0 dropped its own CLI).

### Fixed

- **`concat --inputs a.mp3 b.mp3 c.mp3` would have stopped working.** That
  spelling is the one `EXAMPLES.md` has always shown, and it is what the
  argparse CLI accepted (`nargs="+"`); a click `multiple=True` option binds
  exactly one value per occurrence, so the surviving CLI would have kept only
  `a.mp3` and rejected the rest as "unexpected extra arguments". The trailing
  files are now collected as variadic arguments and appended, so both the
  listed form and the repeated `--inputs a --inputs b` form work and produce
  the same ordered list. Found by running the command, not by reading it, and
  held by a test proven to fail without the fix.

- `--help` offered no `-h`, and `--version` no `-V`.

- **Two subcommands rendered RST markup in their own help text.** click prints
  the docstring verbatim, so `chunk` and `separate` advertised
  ``Extract a ``[start, end]`` slice`` and ``needs the ``[demucs]`` extra``,
  double backticks included. They are plain prose now. (No `\f` truncation is
  needed here: no command docstring carries a numpydoc `Parameters` block.)

- The dead `try: import click / except ImportError` guard — which pointed at
  the `[cli]` extra that no longer exists — is removed, as is the
  `pytest.importorskip("click")` at the top of `tests/test_cli.py`.

- **The skill's CLI reference documented a CLI that never existed.**
  `skills/audio-helper/references/cli-reference.md` showed positional
  arguments (`audio-helper convert IN OUT`) and four invented flags —
  `--sample-rate` for `convert` (it is `--freq`), `--duration` for `split`
  (`--seconds`), `--level` for `roomtone` (`--db`), and a `--stems` selector
  that exists on neither door — so an agent following it produced failing
  commands. Rewritten from each subcommand's own `--help` output.

### Migration

- `audio-helper-click <cmd>` → `audio-helper <cmd>`. Same subcommands, same
  flags; the only behavioural difference from the old argparse door is that a
  non-existent `--input` is now a parser-level usage error (exit 2, usage line,
  the bad value named) instead of a library error (exit 1, `Error: …`). A
  library failure still prints one `Error: …` line and exits 1.
- `pip install "audio-helper[cli]"` now fails on an unknown extra — drop the
  `[cli]`, `click` comes with the package.
- `python -m audio_helper.cli_click` / `cli_argparse` → `python -m
  audio_helper.cli`.

### Tests

`tests/test_cli.py` is rewritten against the single CLI: 19 cases, down from
24. The duplicated per-door scenarios collapse onto one door, and the one that
lived only on the argparse side — a library exception surfacing as a single
`Error: …` line and exit 1 — is carried over rather than deleted. It had to
change trigger: a missing path is a usage error under click and no longer
reaches the wrapper, so the exception is injected instead, which also moves
the guarantee out of the `integration` marker and into the default suite. The
real-ffmpeg version of it stays, as `integration`. New: `-h` and `-V` beside
their long forms, both `concat` arities, and the usage-error exit code.

## [2.4.0] - 2026-10-21

### Changed

- **The GUI's strings moved from `locales/i18n.yaml` to a `gettext`
  catalogue**, the suite-wide mechanism as of os-helper 2.6.0. The English is
  now the source, written in the markup itself as `{{_("…")}}`; the French
  lives in `audio_helper/locales/fr/LC_MESSAGES/audio_helper.po` and is read
  through `os_helper.Catalog`. At 69 entries this is the largest catalogue in
  the suite.

  A string the French catalogue has not caught up with renders as **readable
  English** instead of as a key. The cost is the mirror image — rewording the
  English orphans its French entry, and gettext would quietly serve the
  English — so `tests/test_gui_i18n.py` hands every source string to
  `Catalog.problems`, which names both halves of that fault.

- **The comparator badge is translated before it is interpolated.**
  `Hearing: {side}` is the one string here carrying a placeholder, and the
  order matters: fill it first and the interpolated sentence matches no
  catalogue entry, so the badge silently falls back to English. Two tests now
  cover it — one that the rendered badge names its side in each language, and
  one that no translation has dropped the `{side}` placeholder, which would
  otherwise fail only at render time. Both proven by breaking them.

  The language toggle is computed rather than translated: its three parts all
  mean "the other language", which is data, not copy.

### Fixed

- **The compiled catalogue would never have shipped.** The stock Python
  `.gitignore` carries a blanket `*.mo`, where a `.mo` is a build artifact;
  here it is the file the GUI reads at run time. Left ignored it would have
  been absent from the repository, therefore from the wheel, and every string
  would have rendered as English on an installed copy while looking perfect
  locally. Negated and held by a "tracked by git" test.

- `tests/test_api.py` asked the locale file for six strings by semantic key.
  It now asks the catalogue by English source, so those assertions name the
  copy rather than a key.

## [2.3.0] - 2026-10-21

### Added

- **The Recipe Canvas GUI speaks French and English.** Every string it shows
  now comes from `audio_helper/locales/i18n.yaml` through
  `os_helper.Locales`, the loader shared across the suite (CODING.md 21.3),
  instead of being written into the markup. `GET /gui?lang=fr|en` picks the
  language; anything the file does not carry — including a mistyped code and
  the empty default — falls back to the declared default locale, so a bad
  `?lang=` returns a page rather than a 500. A flag in the header toggles
  between the two.

  The template is tokenised (`{{TOKEN}}` in HTML, a `T` object for the
  script), so one piece of markup serves both languages and a translator never
  opens a Python file. The script's strings are injected as JSON rather than
  substituted as text: it is the encoding an apostrophe cannot break out of,
  and French copy is full of them. Sixteen of them carry named fields
  (`{index}`, `{total}`, `{op}`, `{count}`, `{name}`, `{score}`, `{status}`,
  `{error}`, `{side}`) rather than being glued together in JavaScript, so a
  translator can reorder the sentence.

  The comparator badge is pre-rendered server-side from the very template the
  script reuses on each toggle — one sentence, one key, rather than a second
  string that could drift from the first. This is the first use of
  `Locales.render()` outside its own tests.

  `GUI_HTML` still resolves, via a module-level `__getattr__` that renders on
  access — importers that predate this keep working, and the locale file is
  not read by callers that never serve the GUI.

  **Two families of string stay out of the catalogue on purpose**, and the
  locale file records the reasoning. The per-verb parameter labels in `OPS`
  (`output_format`, `freq (Hz)`, `db`, `color`, …) each name the form field
  the request actually carries, so the wire's own spelling is what makes a
  step card checkable against the API docs. And the comment header
  `recipeToYaml` writes into `recipe.yaml` belongs to a committable artifact:
  a comment that changed language with the browser would make the same recipe
  diff against itself.

  One deliberate simplification: the comparator's help paragraph lost its
  `<strong>` / `<em>` emphasis. Keeping it would have cut one sentence into
  six fragments, which a translator cannot reorder — the exact failure
  localisation exists to prevent. The before/after distinction is already
  carried by the two labelled selects directly below it.

### Changed

- **The declared `os-helper` floor rises to 2.4.0**, the version that
  introduced `Locales`. It was `>=2.0.0`, which the GUI can no longer honour:
  a user installing against os-helper 2.0.x would get an `AttributeError` out
  of the `/gui` route. Nothing would have caught it — pip resolves the newest
  os-helper, never the floor. `tests/test_gui_i18n.py` now asserts the floor
  against the feature it needs.

### Fixed

- **Four GUI assertions passed on code rather than on anything a user can
  see.** `test_gui_ships_recipe_canvas_features` looked for `"bypass"` and
  `"recipe canvas"` in the served HTML — both of which also appear in the
  script's own identifiers and comment banners, so the assertions would have
  survived the entire user-facing header being deleted. They now ask the
  locale file for the strings the page is supposed to show, which also makes
  them language-independent.

- **A CI/local skew in the ruff scope, latent here and active in
  podcast-helper.** `pyproject.toml` excluded `tests` while the workflow lint
  step named `tests` explicitly — and an explicitly-passed path overrides the
  exclusion, so CI was stricter than the `ruff check .` of the pre-push hook
  and of every developer. podcast-helper sat on a red CI for two days over
  exactly this. The workflow now runs `ruff check .` and `tests` is no longer
  excluded, so there is one definition of clean; it costs zero findings here,
  measured rather than assumed.

### Notes

- `[tool.setuptools.package-data]` did not exist in this project; the locale
  file is the first non-`.py` file the package ships, and it is declared there
  now. Verified the way the sibling defect was found: the built wheel was
  opened and the file found inside it.

## [2.2.0] - 2026-10-19

### Fixed

- **`mel_filter_banks` walked off the end of its own array on a band above
  Nyquist.** Bin edges are `floor((n_fft + 1) * hz / sample_rate)` and nothing
  checked `high_freq` against `sample_rate / 2`, so asking for the "full band" —
  `mel_filter_banks(26, 512, 16000, 0, 16000)`, a natural enough mistake —
  indexed `fbank[i-1, 257]` in an array of width 257 and died on:

      IndexError: index 257 is out of bounds for axis 1 with size 257

  That names neither the argument at fault nor the rule it broke, and the
  traceback points at an inner loop rather than at the caller's band. It was also
  out of character: every other precondition in this module is stated in the
  caller's terms.

  Four guards now cover it (`num_filters >= 1`, `low_freq >= 0`,
  `low_freq < high_freq`, `high_freq <= nyquist`), the Nyquist one explaining
  *why* such a band cannot exist rather than only that it is refused. Exactly
  Nyquist stays legal — that is what `mfcc` passes. `mfcc` itself was never
  exposed, since it defaults `high_freq` to `sample_rate / 2`.

### Added

- **`mel_filter_banks` now warns when filters collapse to nothing.** Because bin
  edges are floored, a grid finer than the FFT can resolve makes neighbouring
  edges land on the same bin; both inner loops then run over an empty range and
  the filter comes out entirely zero. `mel_filter_banks(80, 256, 8000, 0, 4000)`
  returns **7 of 80 filters empty**.

  Nothing raised and nothing warned, so the caller received MFCCs computed from
  73 effective filters while believing they had asked for 80 — a quiet loss of
  resolution that is very hard to trace back from a degraded downstream model.
  The empty rows are now counted and reported through `osh.warning`, naming the
  remedy (lower `num_filters` or raise `n_fft`).

  Deliberately a warning rather than an error: a few dead filters at the band
  edges is a legitimate operating point, just not one to discover by accident.

## [2.1.5] - 2026-09-03

### Fixed

- **`get_audio_duration` could raise a raw `KeyError` instead of a clear
  error.** Some containers among `audio_extensions` (raw/headerless
  formats, some ADTS/MP3 streams) omit `duration` on the audio stream
  itself and only report it at the container ("format") level —
  `is_valid_audio_file`'s own `except` clause already anticipates this
  (it catches `KeyError` alongside `ffmpeg.Error`), but `get_audio_duration`
  read `audio_stream["duration"]` unguarded. Now falls back to
  `probe["format"]["duration"]` and raises the module's usual
  `AssertionError` only if neither is present.

## [2.1.4] - 2026-08-21

### Fixed

- **`_separate_sources` never actually moved the mixture tensor to the GPU.**
  `Tensor.to(device)` returns a new tensor rather than moving in place, and
  the return value was discarded, so on a CUDA machine the model moved to
  the GPU while `mix` silently stayed on the CPU. The following
  `model.forward(chunk)` call then failed on the first cross-device tensor
  op. Fixed by reassigning `mix = mix.to(device)`.
- **`split_audio_regularly` ignored its own `overwrite` parameter.** Every
  chunk was extracted with `extract_audio_chunk(..., overwrite=True)`
  hard-coded, regardless of what the caller passed in, so `overwrite=False`
  silently re-extracted every chunk instead of skipping already-valid ones
  as documented. Now forwards the caller's `overwrite` value.
- **`split_audio_regularly` dropped the entire file for inputs under 1 second.**
  The loop's exit condition (`time_cursor < total_duration - 1`) was meant to
  drop a trailing remainder shorter than 1s after at least one full chunk had
  already been produced, but it also gated the very first iteration: a
  `sound_path` shorter than 1s never entered the loop at all, so the function
  silently returned an empty list instead of one chunk covering the whole
  file. The first chunk is now always produced.

## [2.1.3] - 2026-08-17

### Fixed

- **`save_audio` silently no-op'd on an unsupported input type** (anything
  neither a `torch.Tensor` nor a `numpy.ndarray`) instead of raising the
  `AssertionError` its own docstring promised — no file was written and no
  error surfaced. Now raises with a clear message.
- **`_separate_sources` mutated a `Fade` transform shared across concurrent
  chunks** (`ThreadPoolExecutor`, `nb_workers > 1`): the last, possibly
  truncated chunk disabled `fade_out_len` on the shared instance in place,
  racing against any other chunk's concurrent `fade(out)` call. Each chunk
  now builds its own `Fade` instance instead of mutating a shared one.
- **`generate_silent_audio` accepted a non-positive `duration`**, reaching
  an unguarded `np.zeros(...)` — a cryptic numpy `ValueError` for a negative
  duration, or a silently-written zero-length file for zero. Now raises a
  clear `AssertionError` up front.
- **`extract_audio_chunk`'s `overwrite=False` was a no-op** (ffmpeg always
  re-ran, contrary to its own docstring's "currently advisory" note and
  every CLI/API surface that already exposes the flag). Now honors it via
  the same `_overwrite_audio_file` skip-if-valid helper every sibling
  function in this module already uses.

## [2.1.2] - 2026-08-15

### Fixed

- **Every FastAPI action endpoint (`/convert`, `/chunk`, `/silence`,
  `/concat`, `/roomtone`, `/split`, `/separate`) leaked its request-scoped
  temp directory on any failure.** Cleanup was only ever scheduled via
  `background.add_task` on the success path, which never runs if the
  underlying library call raises first — a malformed upload, an
  out-of-range chunk time range, or any other ordinary client-input mistake
  permanently left its temp directory on disk. Each endpoint now cleans up
  synchronously on any exception before re-raising.
- **The HTTP API collapsed every library exception into a generic 500.**
  `AssertionError` (the library's own input validation), `ValueError`
  (`load_audio` on unreadable input), and `ffmpeg.Error` (ffmpeg itself
  rejecting a corrupt/non-audio upload) now map to HTTP 400 with the
  underlying message, instead of an opaque, undiagnosable 500.
- **Both CLI twins printed a raw Python traceback on an ordinary library
  error** instead of a clean one-line message. Both `audio-helper` and
  `audio-helper-click` now print `Error: ...` to stderr and exit 1;
  `audio-helper-click`'s console-script entry point now points at a new
  `cli_click.console_entry()` wrapper (was the bare `cli` group).

## [2.1.1] - 2026-08-13

### Fixed

- `sound_converter`'s return-type annotation said `-> None`; it has always
  returned the output path (both CLIs rely on it) — annotation corrected to
  `-> str`.
- `api.py`'s `/gui` docstring/redirect comment still called the shipped GUI
  the "audition bench"; it's the "Recipe Canvas".

### Changed

- Documented `split_audio_regularly`'s existing sub-1-second tail-drop
  behavior (was silent).
- Added a conda `environment.yaml`; the Dockerfile now installs
  `requirements.txt` as its own cached layer.
- `requirements-dev.txt` is now derived from `pyproject.toml`'s `[dev]`
  extra instead of maintained by hand.

## [2.1.0] - 2026-08-08

### Added

- **MCP surface** (`audio_helper.mcp`, `[mcp]` extra, entry point
  `audio-helper-mcp`): exposes the existing FastAPI app as MCP tools via
  `fastapi-mcp`, mirroring the pattern already shipped in `standpoint` /
  `vocal-helper` / `md2star` / `os-helper`. Closes the CLI/API/MCP surface
  gap for audio-helper flagged in `ai-helpers/.private/do.md` §7.

## [2.0.1] - 2026-08-02

Documentation-only follow-up. The install commands now point at PyPI
(`pip install audio-helper`) instead of a pinned git tag, so the rendered PyPI
project page never drifts to an old version.

### Fixed

- README / LISEZMOI / EXAMPLES install commands no longer self-pin to a git tag
  (`@v1.6.1`); they use `pip install audio-helper` (and the `[extra]` forms),
  which always resolves to the latest published release.

### Added

- `tests/test_readme_install_pin.py` guards against the stale git self-pin ever
  returning to any Markdown file.

## [2.0.0] - 2026-08-02

### Changed

- Adopt the os-helper 2.0.0 foundation: pin `os-helper>=2.0.0,<3`. Logging and
  file management continue to route through os-helper (`osh.info` / `osh.warning`
  / `osh.error`), now on the suite's stable 2.x contract. Major bump because the
  shared foundation crossed a major boundary.
- README and LISEZMOI now use absolute github.com / raw.githubusercontent URLs so
  images and links render on the PyPI project page (relative paths 404 there).
- CI trimmed to a super-light single-Python gate (ruff + pytest both blocking);
  the full multi-version sweep runs locally before push.

## [1.6.1] - 2026-08-01

### Removed

- **MCP surface dropped.** `fastapi-mcp`'s latest release is incompatible with
  the latest `mcp` SDK (`Server.__init__()` signature mismatch — both packages
  are at their own latest, mutually incompatible release with no version
  pairing available). `audio_helper/mcp.py`, the `audio-helper-mcp` entry
  point, the `mcp` extra, and `tests/test_mcp.py` are removed; the FastAPI
  surface (`[api]` extra) is unaffected. Docs/Dockerfile updated (four
  surfaces: library, CLI ×2, HTTP API, GUI).
- **Agent skill dropped from the public repo.** Without an MCP surface, the
  Claude/OpenCode skill (`skills/`) no longer earns its keep as public
  distribution — moved to the gitignored `.private/skills/` (kept locally as
  reference, never published). `TRIGGERS.md` stays public; its skill-specific
  framing and dead `skills/` links are removed.

## [1.6.0] - 2026-07-20

### Added

- **Browser GUI — "Recipe Canvas"** served by the FastAPI app at `GET /gui`
  (and `GET /` redirects there). A single self-contained HTML + Tailwind-CDN +
  vanilla-JS page (in `audio_helper/gui.py`), **no build step / no framework /
  no npm**, that chains the eight verbs (convert / chunk / silence / concat /
  roomtone / split / separate / resemblance) into a **sequential pipeline**:
  - Client-orchestrated runner: each step POSTs to the *existing* API endpoint
    and its output file is piped into the next step's input (the browser is the
    orchestrator — zero new server-side recipe logic).
  - Per step: a WaveSurfer.js waveform + an `<audio>` player of that step's
    output, an on-demand Vega-Lite spectrogram, and a **Bypass** toggle for
    instant A/B. `split` / `separate` zips are unzipped in-browser (fflate) so a
    chosen member can chain downstream; `resemblance` shows its score inline.
  - **Ear-first comparator**: pick a before/after clip, scrub two aligned
    WaveSurfer tracks together, and toggle which you hear at the same playhead
    with the **Space bar**; an optional difference view draws `after − before`.
  - **`recipe.yaml` export / import** (client-side) so a pipeline is a
    committable, human-diffable artifact.
  - CDN libraries only (WaveSurfer.js, Vega-Lite/Vega/vega-embed, fflate,
    Tailwind) — matches the AI Helpers house style.
- **Agent skill** `skills/audio-helper/` (Claude Skills + OpenCode standard):
  trigger-rich `SKILL.md` plus progressive-disclosure `references/`
  (`cli-reference.md`, `surfaces.md`, `triggers.md`) and a `skills/README.md`.
- **`TRIGGERS.md`** at the repo root — an exhaustive, auditable catalogue of
  invocation phrasings, commands, functions, and file types; linked from
  README and LISEZMOI.

### Changed

- FastAPI app `version` is now resolved from installed package metadata instead
  of a hard-coded string (it had drifted to `1.5.5`).
- Explicit `X | None` type hints on optional parameters across `main.py`
  (`load_audio`, `extract_audio_chunk`, `generate_silent_audio`,
  `audio_concatenation`, `mix_room_tone`, `separate_sources`, `_separate_sources`).

### Fixed

- `.gitignore` no longer ignores `*.json` / `*.yaml` wholesale (it silently
  swallowed skill metadata and any exported `openapi.json`); scoped to
  `audio_tests/`. Added `.private/` to the ignore list.

### Documentation

- Reworked `GUI.md`: "what ships today" now documents the full Recipe Canvas
  (runner, per-step bypass, ear-first comparator, `recipe.yaml` round-trip); the
  remaining surfaces (live-keystroke spectrogram, batch drop zone, MFCC cluster
  view) stay clearly labelled as roadmap.
- README and LISEZMOI: added a local-first badge and an honest three-case
  **Promise / La promesse** section (guaranteed-local · model-weights caveat ·
  your decision); added the GUI, agent-skill, and `TRIGGERS.md` sections;
  refreshed version pins to `v1.6.0`.

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
