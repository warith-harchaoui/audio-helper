# GUI — Audio Helper

This document has two parts: **(1) what ships today** — the **Recipe Canvas**, a
client-orchestrated sequential pipeline served by the FastAPI app — and **(2) the
roadmap** — the remaining, more ambitious surfaces we may build later. The
roadmap is explicitly *not yet implemented*; do not mistake it for current
behaviour.

The Recipe Canvas keeps the suite's house style: **vanilla JS + Tailwind (CDN),
no build step, no framework, no npm.** Waveforms come from WaveSurfer.js and
spectrograms from Vega-Lite — both loaded via CDN `<script>` tags.

---

## Part 1 — What ships today: the Recipe Canvas

A single, self-contained HTML page (Tailwind via CDN + vanilla ES-module JS,
**no build step**) served by the existing FastAPI app.

- **Route**: `GET /gui` (and `GET /` redirects to it). Source:
  `audio_helper/gui.py` (`GUI_HTML`), wired in `audio_helper/api.py`.
- **Run it**:
  ```bash
  pip install 'audio-helper[api]'
  uvicorn audio_helper.api:app --port 8000
  # open http://localhost:8000/gui
  ```

### 1. Recipe Canvas *(primary surface)*

- Drop or pick a **source** audio file → its waveform shows what feeds the
  recipe.
- **Chain the eight verbs** (`convert`, `chunk`, `silence`, `concat`,
  `roomtone`, `split`, `separate`, `resemblance`) into an ordered, **sequential**
  pipeline. Add steps, reorder them (↑ / ↓), and remove them.
- **Running is client-orchestrated**: the browser calls the *existing* API
  endpoints in order, feeding each step's output file straight into the next
  step's input. No new server logic runs a recipe — the page is the orchestrator.
- Per step: a **WaveSurfer waveform** + an `<audio>` player of that step's
  output, an on-demand **Vega-Lite spectrogram**, and a **Bypass** toggle that
  routes the input straight to the output for instant A/B.
- `split` / `separate` return a `.zip`; the page unzips it in the browser
  (fflate) and lets you pick which member chains downstream. `resemblance` is a
  terminal step and shows its MFCC similarity score inline.

### 2. Ear-first comparator

- Pick a **before** and an **after** clip (the source or any step's output).
- Two aligned WaveSurfer tracks scrub together; **Space bar** toggles which one
  you hear at the *same* playhead — the mixer-style A/B producers actually use.
- Optional **difference view**: renders a coarse `after − before` waveform so
  tuning room-tone dB, MFCC thresholds, or Demucs quality is *visible*, not just
  audible.

### 3. `recipe.yaml` export / import

- **Export snapshot as `recipe.yaml`** — a shell-independent, human-diffable
  artifact you can commit to a repo. Import replays it back into the canvas.
- Both the YAML emit and parse happen client-side.

### Stack (as shipped)

- Front end: **vanilla JS + Tailwind (CDN)**, no build step — matches the rest
  of the AI Helpers suite. Waveforms: **WaveSurfer.js**. Spectrograms:
  **Vega-Lite** (+ Vega / vega-embed). In-browser unzip: **fflate**.
- Back end: the FastAPI app already exists (`audio_helper.api`) and covers 100 %
  of the operations. The GUI is a client only — it adds **zero** server-side
  recipe logic.
- Recipe format: YAML, versioned, human-diffable.

---

## Part 2 — Roadmap (aspirational, not implemented)

> A design plan, not a CLI mirror. The Recipe Canvas above already makes the
> pipeline visible, auditable, and A/B-comparable. The surfaces below go
> further and are **not shipped yet**.

## North star

> **A canvas where audio flows through named operations, side by side,
> and you can hear every intermediate step.**

## Not-yet-shipped surfaces

### Live spectrogram on every keystroke

Today the spectrogram is rendered **on demand** (click "Spectrogram" on a step).
The roadmap version recomputes it live as parameters change (debounced ~300 ms)
so the frequency-domain effect of every tweak is instant. Deferred because a
per-keystroke DFT in the browser needs a Web Worker + a real FFT to stay smooth.

### Draggable node graph

The shipped canvas is a **linear, ordered list** of steps (which matches the
inherently sequential nature of audio prep). A future version could render a
left-to-right **node graph** with edges carrying audio *and* metadata (sample
rate, channels, duration), highlighting downstream nodes in red when an upstream
change would invalidate them.

### Batch Drop Zone + contact sheet

A single big rectangle. Drop any number of files → they enter the canvas as a
batch context. Every step processes the whole batch; outputs sit in a
**contact-sheet view** (thumbnail waveform per file, sortable by any metadata
column: duration, RMS, similarity score). Right-click → *"open in Recipe
Canvas"* to trace back the pipeline.

### MFCC-embedding cluster view

Drop 100 files, see them clustered by MFCC similarity, click a cluster to hear
one representative. The "we can only do this in a GUI" moment — dataset triage
for AI training.

## Design principles

- **Nothing invisible.** Every operation shows its effect *on this file*, not a
  symbolic parameter.
- **Time is a first-class citizen.** Everything scrubs; the comparator playhead
  is shared across before/after.
- **Files, not memory blobs.** The GUI outputs are byte-identical to the CLI
  outputs — same endpoints, same library, no "GUI produces different files".
- **Keyboard first, mouse second.** The comparator's Space-bar A/B toggle is
  inspired by mixing consoles, not Photoshop.
- **Colorblind-safe by construction.** State uses shape + label + text, never
  color alone (see companion `front-colors` audit skill).

## What we deliberately don't do

- **No timeline editor.** DAWs already exist (Reaper, Ardour). Cuts happen via
  `chunk` and `split` steps, visualized but not manipulated on a linear ruler.
- **No effects rack.** No EQ, no compressor. Only the audio-prep-for-AI verbs.
- **No cloud lock-in.** Everything runs on the same local FastAPI server the
  container already ships. The GUI is a thin JS client.

## Milestones

| Milestone | What ships | Status |
| --- | --- | --- |
| M0 | Recipe Canvas with `convert` / `chunk` / `split`, waveform preview. | ✅ shipped |
| M1 | All 8 verbs. Ear-first comparator. | ✅ shipped |
| M3 | Recipe export/import. | ✅ shipped (validation graph: roadmap) |
| M-live | Live spectrogram on every keystroke. | 🔭 roadmap |
| M2 | Batch Drop Zone + contact sheet. | 🔭 roadmap |
| M4 | MFCC-embedding cluster view. | 🔭 roadmap |

## Non-goals (recorded so we do not drift)

- Not a full DAW.
- Not a hosted SaaS.
- Not a substitute for the CLI in CI (recipes emit CLI-equivalent YAML that CI
  can replay headless).

## Success metric

> A user who owns 500 clips and needs to prep a training set for a speech model
> does the whole job in one afternoon, in one window, and finishes with a
> committable `recipe.yaml`.

If we ship that, we win.
