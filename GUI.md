# GUI — Audio Helper

This document has two parts: **(1) what ships today** — a minimal, working
"audition bench" served by the FastAPI app — and **(2) the roadmap** — an
ambitious, opinionated visual product we may build later. The roadmap is
explicitly *not yet implemented*; do not mistake it for current behaviour.

---

## Part 1 — What ships today: the minimal audition bench

A single, self-contained HTML page (Tailwind via CDN + vanilla ES-module JS,
**no build step, no framework**) served by the existing FastAPI app.

- **Route**: `GET /gui` (and `GET /` redirects to it). Source:
  `audio_helper/gui.py` (`GUI_HTML`), wired in `audio_helper/api.py`.
- **Run it**:
  ```bash
  pip install 'audio-helper[api]'
  uvicorn audio_helper.api:app --port 8000
  # open http://localhost:8000/gui
  ```
- **Workflow**: drop or pick an audio file → choose one operation (`convert`,
  `chunk`, `silence`, `concat`, `roomtone`, `split`, `separate`,
  `resemblance`) → the page reveals only that operation's fields → **Run**
  POSTs `multipart/form-data` to the *same* API endpoints the CLI and MCP use
  → the result plays back in an `<audio>` element next to the input for A/B
  comparison, with a download link (a single file, or a `.zip` for `split` /
  `separate`; `resemblance` shows the score inline).
- **Deliberately minimal**: it adds *zero* server-side logic. It is a thin,
  friendly front door to the API — and the canonical template other AI Helpers
  repos copy (swap the operation list and per-op fields, keep the plumbing).

Everything below is a **future roadmap**, not shipped.

---

## Part 2 — Roadmap (aspirational, not implemented)

> A design plan, not a CLI mirror. The CLI already handles "one file at
> a time, one operation at a time". A richer GUI could go further. This
> section lays out an ambitious, opinionated visual product for the
> audio-prep-for-AI workflow — a target, not current behaviour.

## North star

> **A canvas where audio flows through named operations, side by side,
> and you can hear every intermediate step.**

Audio work is inherently sequential (load → clean → split → embed …)
but auditing the *effect* of each operation is what the CLI cannot
give you. The GUI's job is to make the **pipeline visible, auditable,
and A/B-comparable** — not to reproduce ffmpeg flags with checkboxes.

## Three surfaces, one product

### 1. Recipe Canvas *(primary surface)*

- Draggable node graph, left-to-right — nodes are the eight verbs
  (`convert`, `chunk`, `silence`, `concat`, `roomtone`, `split`,
  `separate`, `resemblance`).
- Each node shows a **live waveform + spectrogram of its output**
  updated on parameter change (debounced 300 ms).
- Edges carry audio *and metadata* (sample rate, channels, duration).
  Downstream nodes highlight in red when an upstream change would
  invalidate them (e.g. changing sample rate under a fixed-window MFCC).
- One toggle per node: **"Bypass"** — routes the input straight to the
  output. Makes A/B trivial: compare the graph with a node bypassed vs
  active.
- Right-click a node → **"Export snapshot as recipe.yaml"**. Recipes
  are shell-independent artifacts you can commit to a repo.

### 2. Ear-first Comparator

Two horizontally aligned waveform tracks that scrub together. Every
node with a "Compare against original" toggle sends the pre/post
waveforms here. Bindings:

- Space bar: toggle *before / after* on the same playhead position.
- `A / B` keys: 1-second dial between two versions — the mixer-style
  workflow producers actually use.
- Difference channel: renders `after - before` at the bottom.
  Massively helps tune room-tone dB, MFCC similarity thresholds,
  Demucs separation quality.

### 3. Batch Drop Zone

A single big rectangle. Drop any number of files → they enter the
canvas as a batch context. Every node processes the whole batch;
outputs sit in a **contact-sheet view** (thumbnail waveform per file,
sortable by any metadata column: duration, RMS, similarity score).
Right-click → *"open in Recipe Canvas"* to trace back the graph.

## Design principles

- **Nothing invisible.** Every operation shows its effect *on this
  file*, not a symbolic parameter. That is the entire point of a GUI.
- **Time is a first-class citizen.** Everything scrubs. Playhead is a
  singleton across the app.
- **Files, not memory blobs.** The recipe engine writes intermediates
  to a project folder (opt-out). The CLI outputs and the GUI outputs
  are byte-identical — no "GUI produces different files".
- **Explain the ML.** For nodes involving models (`separate`,
  `resemblance`): tooltip shows the model card, expected quality
  window, and a link to the underlying paper. No mystery buttons.
- **Keyboard first, mouse second.** Every node action has a shortcut.
  The comparator's `A/B` toggle is inspired by mixing consoles, not
  Photoshop.
- **Colorblind-safe by construction.** All state uses shape + color +
  text, never color alone (see companion `front-colors` audit skill).

## What we deliberately don't do

- **No timeline editor.** DAWs already exist (Reaper, Ardour). We are
  not competing with them. Cuts happen via `chunk` and `split` nodes,
  visualized but not manipulated on a linear ruler.
- **No effects rack.** No EQ, no compressor. Only the audio-prep-for-
  AI verbs. Scope discipline.
- **No cloud lock-in.** Everything runs on the same local FastAPI
  server the container already ships. GUI is a thin JS client.

## Stack

- Front end: TypeScript + Svelte 5 + WaveSurfer.js (waveforms) +
  Vega-Lite (spectrograms). No React — matches the `front-ui`
  companion skill's stack.
- Back end: the FastAPI app already exists (`audio_helper.api`) and
  covers 100 % of the operations. GUI is a client only.
- Recipe format: YAML, versioned, human-diffable.

## Milestones

| Milestone | What ships | Why first |
| --- | --- | --- |
| M0 | Recipe Canvas with 3 nodes: `convert`, `chunk`, `split`. Waveform preview. | Prove the canvas metaphor before scaling verbs. |
| M1 | All 8 verbs. Ear-first comparator. | Feature parity with the CLI. |
| M2 | Batch Drop Zone + contact sheet. | Where the GUI passes the CLI in productivity. |
| M3 | Recipe export/import + node validation graph. | Reproducibility story for pipelines shared across a team. |
| M4 | MFCC-embedding cluster view: drop 100 files, see them clustered by MFCC similarity, click a cluster to hear one representative. | The "we can only do this in a GUI" moment — dataset triage for AI training. |

## Non-goals (recorded so we do not drift)

- Not a full DAW.
- Not a hosted SaaS.
- Not a substitute for the CLI in CI (recipes emit CLI-equivalent
  YAML that CI can replay headless).

## Success metric

> A user who owns 500 clips and needs to prep a training set for a
> speech model does the whole job in one afternoon, in one window,
> and finishes with a committable `recipe.yaml`.

If we ship that, we win.
