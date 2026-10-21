"""
Audio Helper — single-page GUI ("Recipe Canvas").

This module holds nothing but the self-contained HTML document served by the
FastAPI app at ``GET /gui`` (see :mod:`audio_helper.api`). It is deliberately
build-step-free: one string of HTML + Tailwind (via CDN) + vanilla ES-module
JavaScript. There is no bundler, no framework, no npm — the whole page is a
static asset the API returns verbatim.

Where the words live
--------------------
In the markup, in English. Each one is wrapped as ``{{_("…")}}`` in the HTML,
or listed in :data:`_JS_NAME_TO_SOURCE` for the script, and
:class:`os_helper.Catalog` translates it through the compiled ``gettext``
catalogues under ``locales/`` (``CODING.md`` 21.3).

English is the msgid, so it needs no catalogue and cannot fall out of sync with
one; a string the French catalogue has not caught up with renders as readable
English instead of as a key. The cost is the mirror image: rewording the
English here orphans its French entry, and gettext would quietly serve the
English. ``tests/test_gui_i18n.py`` exists to catch that — it hands every source
string to :meth:`os_helper.Catalog.problems`, so an orphaned translation is a
red build rather than a half-reverted page.

Two families of string stay out of the catalogue on purpose, and the locale
file says why: the per-verb parameter labels in ``OPS``, which name the form
fields the request actually carries, and the comment header written into an
exported ``recipe.yaml``, which is a committable artifact rather than UI.

:func:`render_gui` resolves a language and returns the finished page.

From bench to canvas
--------------------
The first shipped GUI was a minimal "audition bench": one file, one operation,
one A/B comparison. This module upgrades it to the **Recipe Canvas** described
in ``GUI.md`` — a *sequential* pipeline where the user chains the eight verbs
(``convert``, ``chunk``, ``silence``, ``concat``, ``roomtone``, ``split``,
``separate``, ``resemblance``) into a recipe and hears every intermediate step.

The whole thing is still client-only: running a recipe calls the very same
``/convert`` / ``/chunk`` / … endpoints the CLI uses, in order, feeding
each step's output file straight into the next step's input. No new server
logic is required to run a recipe — the browser is the orchestrator.

What the page does
------------------
- Drop / pick a local audio file (kept entirely client-side).
- Build a recipe: add steps (one per verb), reorder them, set per-step params.
- Run the recipe top-to-bottom; each step POSTs ``multipart/form-data`` to the
  existing API and the returned file becomes the next step's input.
- Per step: a WaveSurfer.js waveform + an ``<audio>`` player of that step's
  output, plus a **Bypass** toggle (routes input straight to output for an
  instant A/B).
- Ear-first comparator: pick any two produced clips, scrub them together, and
  toggle *before / after* at the same playhead with the space bar; an optional
  difference view highlights what changed.
- Export / import the recipe as ``recipe.yaml`` so a pipeline is a committable
  artifact (both the YAML emit and parse happen client-side).

CDN libraries (no build step)
-----------------------------
- Tailwind CSS — utility styling.
- WaveSurfer.js — waveform rendering + scrubbing.
- Vega-Lite / Vega-Embed — spectrogram of a step's output (roadmap-adjacent,
  rendered on demand rather than on every keystroke).
- fflate — tiny in-browser unzip so ``split`` / ``separate`` (which return a
  ``.zip``) can still feed a chosen member into a downstream step.

Why a separate module
---------------------
Keeping the (long) HTML out of :mod:`audio_helper.api` keeps the route
definitions readable and lets other repos in the AI Helpers suite copy this
file almost verbatim as their GUI template: swap the operation list and the
per-operation form fields, keep the plumbing.

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import functools
import html
import json
import re
from pathlib import Path

import os_helper as osh

# Inside the package, not beside it: a file one level up is not package data
# and does not travel in the wheel. best-engine shipped exactly that bug —
# `pip install` gave a package whose every GUI string raised FileNotFoundError.
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

#: The gettext domain, and therefore each ``.mo`` file's stem.
_DOMAIN = "audio_helper"

#: The language served when the request names one we cannot render.
_DEFAULT_LANGUAGE = "fr"


@functools.lru_cache(maxsize=1)
def _catalog() -> osh.Catalog:
    """Resolve the catalogue directory once, through the suite's shared loader."""
    return osh.Catalog.from_package(_LOCALES_DIR, _DOMAIN, default_language=_DEFAULT_LANGUAGE)


def supported_languages() -> tuple[str, ...]:
    """Language codes the GUI can render: English plus every compiled catalogue."""
    return _catalog().languages


def default_language() -> str:
    """The language an unknown or absent ``?lang=`` falls back to."""
    return _catalog().default_language


# The language toggle is computed, not translated. Its three parts all mean
# "the other language", which is data rather than copy: a msgid of
# ``/gui?lang=fr`` whose translation is ``/gui?lang=en`` would be nonsense, and
# `Catalog.problems` would rightly treat it as prose that had drifted.
#
#: Flag shown for a language — the flag of that language, not of the page.
_FLAGS: dict[str, str] = {"fr": "🇫🇷", "en": "🇬🇧"}

#: How each language names itself. Deliberately NOT translated: a French page
#: offering English says "English version" in English, which is what a reader
#: scanning for their own language actually looks for.
_ENDONYMS: dict[str, str] = {"fr": "Version française", "en": "English version"}


# The inline script's strings. The JS names are the contract between this
# module and the script below; the English beside each one is the msgid, so
# the script's copy lives in the same place as the markup's and is checked
# by the same test. Injected as JSON, which is the one encoding an
# apostrophe cannot break out of — and French copy is full of them.
_JS_NAME_TO_SOURCE: dict[str, str] = {
    "hearing": "Hearing: {side}",
    "beforeLabel": "Before",
    "afterLabel": "After",
    "emptyRecipe": "No steps yet. Pick a verb above and press “+ Add step”.",
    "moveUp": "Move step up",
    "moveDown": "Move step down",
    "bypass": "Bypass",
    "removeStep": "Remove step",
    "terminalScore": " · terminal (score)",
    "multiFileZip": " · multi-file (zip)",
    "concatHint": "extra file(s) to append after the piped input",
    "resemblanceHint": "second file (B) to compare the piped input (A) against",
    "scoreLine": "MFCC similarity score: {score} (terminal step)",
    "outputHeading": "Output — {name}",
    "download": "Download",
    "spectrogram": "Spectrogram",
    "notRun": "Not run yet.",
    "zipPick": "{count} files produced — pick which one chains downstream:",
    "needStep": "Add at least one step first.",
    "running": "Running recipe…",
    "stepProgress": "Step {index}/{total}: {op}…",
    "noInput": "No input available — the previous step produced nothing chainable.",
    "stoppedNoInput": "Recipe stopped: a step had no input.",
    "resemblanceNeedsB": "resemblance needs a second file (B).",
    "stopped": "Recipe stopped.",
    "httpError": "HTTP {status}: {body}",
    "stoppedAtStep": "Recipe stopped at step {index}.",
    "emptyArchive": "The step produced an empty archive.",
    "recipeComplete": "Recipe complete.",
    "recipeFailed": "Recipe failed: {error}",
    "nothingToExport": "Nothing to export — the recipe is empty.",
    "exported": "Exported recipe.yaml.",
    "imported": "Imported {count} step(s) from {name}.",
    "importFailed": "Could not import recipe: {error}",
    "computingSpectrogram": "Computing spectrogram…",
    "spectrogramUnavailable": "Spectrogram unavailable: {error}",
    "clipSource": "Source",
    "clipStep": "Step {index}: {op} ({name})",
    "diffUnavailable": "Difference unavailable: {error}",
}


#: One ``{{_("…")}}`` site in the template. Non-greedy and newline-free, so a
#: run of them on one line stays separate and an unterminated one is left
#: alone rather than swallowing the rest of the page.
_SOURCE_RE = re.compile(r'\{\{_\("(.*?)"\)\}\}')

_GUI_TEMPLATE: str = r"""<!doctype html>
<html lang="{{LANG}}" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{_("Audio Helper — Recipe Canvas")}}</title>
  <!-- Tailwind via CDN: keeps the page a single self-contained file, no build. -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- WaveSurfer.js: per-step waveforms + scrubbing for the ear-first comparator. -->
  <script src="https://unpkg.com/wavesurfer.js@7"></script>
  <!-- fflate: tiny in-browser unzip so split/separate (zip) can chain a member. -->
  <script src="https://cdn.jsdelivr.net/npm/fflate@0.8.2/umd/index.js"></script>
  <!-- Vega + Vega-Lite + embed: on-demand spectrogram of a step's output. -->
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    /* Respect users who ask for reduced motion (accessibility baseline). */
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  </style>
</head>
<body class="h-full bg-slate-50 text-slate-900 antialiased">
  <div class="mx-auto max-w-5xl px-4 py-8">
    <header class="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 class="text-2xl font-semibold tracking-tight">{{_("Audio Helper — Recipe Canvas")}}</h1>
        <p class="mt-1 text-sm text-slate-600">
          {{_("Chain the eight verbs into a sequential pipeline. Each step runs on the local API and feeds its output into the next — hear every intermediate step, bypass any step for instant A/B, and export the whole thing as a committable")}} <code>recipe.yaml</code> {{_("file.")}}
        </p>
        <p class="mt-1 text-xs text-slate-500">
          {{_("Local-first: audio is processed by your local API (ffmpeg + local Demucs). Nothing is uploaded to a third party; no telemetry.")}}
        </p>
      </div>
      <!-- Recipe-level actions: export / import a recipe.yaml artifact,
           plus the flag that links to the other locale of this page. -->
      <div class="flex flex-wrap items-center gap-2">
        <a href="{{LANG_HREF}}" aria-label="{{LANG_ARIA}}"
           class="text-xl hover:opacity-70">{{LANG_FLAG}}</a>
        <button id="export-yaml"
                class="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium
                       hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
          {{_("Export recipe.yaml")}}
        </button>
        <label class="cursor-pointer rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                      font-medium hover:bg-slate-100 focus-within:ring-2 focus-within:ring-blue-500">
          {{_("Import recipe.yaml")}}
          <input id="import-yaml" type="file" accept=".yaml,.yml,text/yaml" class="hidden" />
        </label>
      </div>
    </header>

    <!-- 1) Source file: drag-and-drop zone doubling as a file picker. -->
    <section class="mb-5">
      <label for="file" class="block text-sm font-medium mb-1">{{_("Source audio file")}}</label>
      <div id="drop" tabindex="0"
           class="flex flex-col items-center justify-center rounded-xl border-2 border-dashed
                  border-slate-300 bg-white px-4 py-6 text-center cursor-pointer
                  focus:outline-none focus:ring-2 focus:ring-blue-500 hover:border-blue-400">
        <p class="text-sm text-slate-500">{{_("Drop a file here, or click to choose")}}</p>
        <p id="filename" class="mt-2 text-sm font-medium text-slate-800"></p>
        <input id="file" type="file" accept="audio/*,video/*" class="hidden" />
      </div>
      <!-- Source waveform: the recipe's "step 0" that every first step reads from. -->
      <div class="mt-3 rounded-xl border border-slate-200 bg-white p-3">
        <div class="mb-1 flex items-center justify-between">
          <h2 class="text-xs font-medium text-slate-600">{{_("Source waveform")}}</h2>
          <audio id="src-audio" controls class="h-8"></audio>
        </div>
        <div id="src-wave" class="min-h-[48px]"></div>
      </div>
    </section>

    <!-- 2) Add-a-step controls: pick a verb, append it to the recipe. -->
    <section class="mb-4 flex flex-wrap items-end gap-3">
      <div>
        <label for="add-op" class="block text-sm font-medium mb-1">{{_("Add a step")}}</label>
        <select id="add-op"
                class="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-500">
          <option value="convert">{{_("convert — re-encode (freq / channels / codec)")}}</option>
          <option value="chunk">{{_("chunk — extract a [start, end] slice")}}</option>
          <option value="silence">{{_("silence — generate N seconds of silence")}}</option>
          <option value="concat">{{_("concat — append extra file(s) head-to-tail")}}</option>
          <option value="roomtone">{{_("roomtone — mix low-level ambient noise")}}</option>
          <option value="split">{{_("split — fixed-duration chunks (zip)")}}</option>
          <option value="separate">{{_("separate — Demucs stems (zip)")}}</option>
          <option value="resemblance">{{_("resemblance — MFCC similarity score")}}</option>
        </select>
      </div>
      <button id="add-step"
              class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white
                     hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
        {{_("+ Add step")}}
      </button>
      <button id="run-recipe"
              class="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white
                     hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500
                     disabled:opacity-50">
        {{_("▶ Run recipe")}}
      </button>
      <button id="clear-recipe"
              class="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium
                     hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
        {{_("Clear")}}
      </button>
      <span id="status" class="text-sm text-slate-600" role="status" aria-live="polite"></span>
    </section>

    <!-- 3) The recipe: an ordered list of step cards, rendered by JS. -->
    <section id="recipe" class="mb-8 space-y-4" aria-label="{{_("Recipe steps")}}"></section>

    <!-- 4) Ear-first comparator: align two produced clips and toggle before/after. -->
    <section class="mb-8 rounded-xl border border-slate-200 bg-white p-4">
      <h2 class="mb-2 text-sm font-semibold">{{_("Ear-first comparator")}}</h2>
      <p class="mb-3 text-xs text-slate-600">
        {{_("Pick a before clip and an after clip (the source, or any step's output). Press")}} <kbd class="rounded border px-1">{{_("Space")}}</kbd>
        {{_("to toggle which one you hear at the same playhead — the mixer-style A/B producers use.")}}
      </p>
      <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label for="cmp-a" class="block text-xs font-medium mb-1">{{_("Before")}}</label>
          <select id="cmp-a"
                  class="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500"></select>
        </div>
        <div>
          <label for="cmp-b" class="block text-xs font-medium mb-1">{{_("After")}}</label>
          <select id="cmp-b"
                  class="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500"></select>
        </div>
      </div>
      <div id="cmp-wave" class="mt-3 min-h-[64px]"></div>
      <div class="mt-3 flex flex-wrap items-center gap-3">
        <button id="cmp-play"
                class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white
                       hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
          {{_("Play / Pause")}}
        </button>
        <button id="cmp-toggle"
                class="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium
                       hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500">
          {{_("Toggle A/B (Space)")}}
        </button>
        <!-- Pre-rendered server-side from the same template the script uses,
             so the badge is already a sentence before the first toggle. -->
        <span id="cmp-which"
              class="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-white">
          {{CMP_HEARING_INIT}}
        </span>
        <label class="ml-auto flex items-center gap-2 text-xs text-slate-600">
          <input id="cmp-diff" type="checkbox" class="h-4 w-4" />
          {{_("Show difference view")}}
        </label>
      </div>
      <!-- Difference view: rendered on demand, colorblind-safe (shape + label, not color alone). -->
      <div id="cmp-diff-wrap" hidden class="mt-3">
        <p class="mb-1 text-xs font-medium text-slate-600">{{_("Difference (after − before)")}}</p>
        <canvas id="cmp-diff-canvas" class="w-full rounded border border-slate-200" height="80"></canvas>
      </div>
    </section>
  </div>

  <script type="module">
    // ===================================================================
    // Recipe Canvas — client-orchestrated sequential pipeline.
    //
    // The whole runner is browser-side: each step POSTs to the SAME API
    // endpoint the CLI uses, and the returned blob becomes the next
    // step's input file. No new server logic is needed to run a recipe.
    // ===================================================================

    // Every human-readable string the script writes comes from the locale
    // file, injected as JSON — the one encoding an apostrophe cannot break
    // out of, and French copy is full of them.
    const T = {{T_JSON}};

    // --- tiny DOM helpers -------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const status = (msg) => { $("status").textContent = msg; };

    // The source file the recipe reads from (kept client-side until Run).
    let sourceFile = null;

    // Per-verb metadata: which endpoint, what shape the response has, and
    // the parameter fields the step card should render. This is the single
    // source of truth the UI, the runner, and the YAML (de)serializer share.
    // kind: "file"  -> response is a single audio file (chainable)
    //       "zip"   -> response is a .zip of many files (chainable via a picked member)
    //       "score" -> response is JSON {score} (terminal, not chainable)
    const OPS = {
      convert:   { endpoint: "/convert",   kind: "file",
                   params: [
                     { name: "output_format", label: "output_format", type: "text",   value: "wav" },
                     { name: "freq",          label: "freq (Hz)",     type: "number", value: "44100" },
                     { name: "channels",      label: "channels",      type: "number", value: "1" },
                   ] },
      chunk:     { endpoint: "/chunk",     kind: "file",
                   params: [
                     { name: "start",         label: "start (s)",     type: "number", value: "0",  step: "0.01" },
                     { name: "end",           label: "end (s)",       type: "number", value: "5",  step: "0.01" },
                     { name: "output_format", label: "output_format", type: "text",   value: "wav" },
                   ] },
      silence:   { endpoint: "/silence",   kind: "file", noInput: true,
                   params: [
                     { name: "duration",      label: "duration (s)",  type: "number", value: "3",  step: "0.1" },
                     { name: "output_format", label: "output_format", type: "text",   value: "wav" },
                   ] },
      concat:    { endpoint: "/concat",    kind: "file", extraFiles: true,
                   params: [
                     { name: "output_format", label: "output_format", type: "text",   value: "wav" },
                   ] },
      roomtone:  { endpoint: "/roomtone",  kind: "file",
                   params: [
                     { name: "db",            label: "db",            type: "number", value: "-42", step: "1" },
                     { name: "color",         label: "color",         type: "text",   value: "pink" },
                     { name: "output_format", label: "output_format", type: "text",   value: "wav" },
                   ] },
      split:     { endpoint: "/split",     kind: "zip",
                   params: [
                     { name: "seconds",       label: "seconds / chunk", type: "number", value: "10", step: "0.1" },
                     { name: "output_format", label: "output_format",   type: "text",   value: "wav" },
                   ] },
      separate:  { endpoint: "/separate",  kind: "zip",
                   params: [
                     { name: "output_format", label: "output_format",   type: "text",   value: "mp3" },
                   ] },
      resemblance:{ endpoint: "/resemblance", kind: "score", extraFiles: true, single_extra: true,
                   params: [] },
    };

    // --- source file picker + drag-and-drop -------------------------------
    const drop = $("drop");
    const fileInput = $("file");
    // Clicking the drop zone opens the native picker.
    drop.addEventListener("click", () => fileInput.click());
    drop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });
    // Highlight while dragging over the zone.
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("border-blue-500"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("border-blue-500"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("border-blue-500");
      if (e.dataTransfer.files.length) setSource(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => { if (fileInput.files.length) setSource(fileInput.files[0]); });

    // A WaveSurfer instance for the source clip (lazily created once).
    let srcWave = null;

    // Register the source file: name, inline player, waveform, and comparator entry.
    function setSource(f) {
      sourceFile = f;
      $("filename").textContent = f.name;
      const url = URL.createObjectURL(f);
      $("src-audio").src = url;
      // (Re)draw the source waveform so the user sees what feeds the recipe.
      if (srcWave) srcWave.destroy();
      srcWave = WaveSurfer.create({
        container: "#src-wave", height: 48, waveColor: "#94a3b8", progressColor: "#2563eb", url,
      });
      // Refresh the comparator dropdowns: the source is always a valid clip.
      refreshComparatorChoices();
    }

    // ===================================================================
    // Recipe state + rendering
    // ===================================================================

    // The recipe is an ordered array of step objects. Each step holds its
    // verb, its parameter values, a bypass flag, any extra files (for concat
    // / resemblance), and — after a run — its produced output blob + URL.
    let recipe = [];
    // Monotonic id so we can key DOM nodes and WaveSurfer instances per step.
    let nextStepId = 1;
    // WaveSurfer instances keyed by step id, so we can tear them down on rerender.
    const stepWaves = new Map();

    // Build a fresh step object for a given verb with its default params.
    function makeStep(op) {
      const spec = OPS[op];
      // Copy the default param values so editing one step never mutates the spec.
      const values = {};
      for (const p of spec.params) values[p.name] = p.value;
      return {
        id: nextStepId++,
        op,
        values,
        bypass: false,
        extraFiles: [],     // File objects for concat / resemblance
        outBlob: null,      // produced blob after a run (file kind)
        outUrl: null,       // object URL for the produced blob
        outName: "",        // suggested download name
        score: null,        // for resemblance
        zipMembers: [],      // [{name, blob, url}] for split / separate
        chosenMember: 0,     // which zip member chains downstream
        error: null,
      };
    }

    // Append a step of the currently-selected verb to the recipe.
    $("add-step").addEventListener("click", () => {
      recipe.push(makeStep($("add-op").value));
      renderRecipe();
    });

    // Clear the whole recipe (keeps the source file loaded).
    $("clear-recipe").addEventListener("click", () => {
      recipe = [];
      renderRecipe();
    });

    // Render every step card from the current recipe state. We rebuild the
    // list wholesale (simple + correct for a handful of steps) and wire the
    // per-card controls each time.
    function renderRecipe() {
      // Tear down old waveforms so we don't leak WaveSurfer instances.
      for (const w of stepWaves.values()) { try { w.destroy(); } catch (e) {} }
      stepWaves.clear();

      const host = $("recipe");
      host.innerHTML = "";

      // Empty-state hint so the canvas is never a confusing blank slab.
      if (recipe.length === 0) {
        const empty = document.createElement("p");
        empty.className = "rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500";
        empty.textContent = T.emptyRecipe;
        host.appendChild(empty);
        refreshComparatorChoices();
        return;
      }

      // One card per step, in order.
      recipe.forEach((step, index) => {
        const spec = OPS[step.op];
        const card = document.createElement("article");
        card.className = "rounded-xl border border-slate-200 bg-white p-4 shadow-sm";

        // --- header: index, verb name, reorder / bypass / remove controls ---
        const header = document.createElement("div");
        header.className = "mb-3 flex flex-wrap items-center gap-3";
        header.innerHTML = `
          <span class="inline-flex h-6 w-6 items-center justify-center rounded-full bg-slate-800 text-xs font-bold text-white">${index + 1}</span>
          <span class="font-mono text-sm font-semibold">${step.op}</span>
          <span class="text-xs text-slate-500">${spec.endpoint}${spec.kind === "score" ? T.terminalScore : spec.kind === "zip" ? T.multiFileZip : ""}</span>
        `;
        // Reorder up.
        const up = mkBtn("↑", T.moveUp);
        up.disabled = index === 0;
        up.addEventListener("click", () => { swap(index, index - 1); });
        // Reorder down.
        const down = mkBtn("↓", T.moveDown);
        down.disabled = index === recipe.length - 1;
        down.addEventListener("click", () => { swap(index, index + 1); });
        // Bypass toggle: route input straight to output for instant A/B.
        const bypass = document.createElement("label");
        bypass.className = "ml-auto flex items-center gap-2 text-xs font-medium";
        bypass.innerHTML = `<input type="checkbox" class="h-4 w-4" ${step.bypass ? "checked" : ""} /> ${T.bypass}`;
        bypass.querySelector("input").addEventListener("change", (e) => {
          step.bypass = e.target.checked;
        });
        // Remove this step.
        const del = mkBtn("✕", T.removeStep);
        del.classList.add("text-red-600");
        del.addEventListener("click", () => { recipe.splice(index, 1); renderRecipe(); });
        header.append(up, down, bypass, del);
        card.appendChild(header);

        // --- parameter fields for this verb ---
        if (spec.params.length) {
          const grid = document.createElement("div");
          grid.className = "mb-3 grid grid-cols-2 gap-3 sm:grid-cols-3";
          for (const p of spec.params) {
            const wrap = document.createElement("div");
            const inputId = `step-${step.id}-${p.name}`;
            wrap.innerHTML = `
              <label for="${inputId}" class="block text-xs font-medium mb-1">${p.label}</label>
              <input id="${inputId}" type="${p.type}" ${p.step ? `step="${p.step}"` : ""}
                     class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm
                            focus:outline-none focus:ring-2 focus:ring-blue-500" />
            `;
            const input = wrap.querySelector("input");
            input.value = step.values[p.name] ?? "";
            // Persist edits back into the step's param values.
            input.addEventListener("input", () => { step.values[p.name] = input.value; });
            grid.appendChild(wrap);
          }
          card.appendChild(grid);
        }

        // --- extra files (concat needs more files; resemblance needs one B) ---
        if (spec.extraFiles) {
          const wrap = document.createElement("div");
          wrap.className = "mb-3";
          const inputId = `step-${step.id}-extra`;
          const hint = step.op === "concat" ? T.concatHint : T.resemblanceHint;
          wrap.innerHTML = `
            <label for="${inputId}" class="block text-xs font-medium mb-1">${hint}</label>
            <input id="${inputId}" type="file" accept="audio/*,video/*" ${spec.single_extra ? "" : "multiple"}
                   class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
          `;
          wrap.querySelector("input").addEventListener("change", (e) => {
            step.extraFiles = Array.from(e.target.files);
          });
          card.appendChild(wrap);
        }

        // --- output area: player + waveform + (zip member picker) + spectrogram ---
        const out = document.createElement("div");
        out.className = "rounded-lg border border-slate-100 bg-slate-50 p-3";
        if (step.error) {
          // Show a clear, colorblind-safe error (icon + text, not color alone).
          out.innerHTML = `<p class="text-sm text-red-700">⚠ ${step.error}</p>`;
        } else if (step.op === "resemblance" && step.score != null) {
          // The score is a field in the sentence, so a translation can place
          // it where that language puts it.
          out.innerHTML = `<p class="text-sm">${T.scoreLine.replace("{score}", step.score.toFixed(4))}</p>`;
        } else if (step.outUrl) {
          // Single-file (or chosen zip member) output: inline player + waveform.
          out.innerHTML = `
            <div class="mb-1 flex items-center justify-between">
              <h3 class="text-xs font-medium text-slate-600">${T.outputHeading.replace("{name}", step.outName)}</h3>
              <div class="flex items-center gap-2">
                <a class="text-xs font-medium text-blue-600 hover:underline" href="${step.outUrl}" download="${step.outName}">${T.download}</a>
                <button data-spec class="text-xs font-medium text-slate-600 hover:underline">${T.spectrogram}</button>
              </div>
            </div>
            <audio controls class="mb-2 h-8 w-full" src="${step.outUrl}"></audio>
            <div class="step-wave min-h-[40px]"></div>
            <div class="step-spec mt-2"></div>
          `;
          // Draw the step's output waveform.
          const w = WaveSurfer.create({
            container: out.querySelector(".step-wave"), height: 40,
            waveColor: "#a7f3d0", progressColor: "#059669", url: step.outUrl,
          });
          stepWaves.set(step.id, w);
          // Spectrogram is rendered on demand (not on every keystroke) — keeps
          // the "live spectrogram on keystroke" idea as roadmap, not default.
          out.querySelector("[data-spec]").addEventListener("click", () => {
            renderSpectrogram(out.querySelector(".step-spec"), step.outBlob);
          });
        } else {
          out.innerHTML = `<p class="text-xs text-slate-500">${T.notRun}</p>`;
        }

        // Zip member picker: split/separate produce many files; the user picks
        // which member continues downstream (and can play/download each).
        if (spec.kind === "zip" && step.zipMembers.length) {
          const picker = document.createElement("div");
          picker.className = "mt-3 border-t border-slate-200 pt-3";
          picker.innerHTML = `<p class="mb-1 text-xs font-medium text-slate-600">${T.zipPick.replace("{count}", step.zipMembers.length)}</p>`;
          const sel = document.createElement("select");
          sel.className = "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm";
          step.zipMembers.forEach((m, i) => {
            const o = document.createElement("option");
            o.value = String(i); o.textContent = m.name;
            if (i === step.chosenMember) o.selected = true;
            sel.appendChild(o);
          });
          sel.addEventListener("change", () => {
            step.chosenMember = Number(sel.value);
            // Repoint the step's chainable output to the chosen member.
            const m = step.zipMembers[step.chosenMember];
            step.outBlob = m.blob; step.outUrl = m.url; step.outName = m.name;
            renderRecipe();
            refreshComparatorChoices();
          });
          picker.appendChild(sel);
          out.appendChild(picker);
        }

        card.appendChild(out);
        host.appendChild(card);
      });

      // Keep the comparator's clip choices in sync with produced outputs.
      refreshComparatorChoices();
    }

    // Small helper: build a square icon button with an accessible label.
    function mkBtn(glyph, label) {
      const b = document.createElement("button");
      b.type = "button";
      b.title = label;
      b.setAttribute("aria-label", label);
      b.textContent = glyph;
      b.className = "inline-flex h-7 w-7 items-center justify-center rounded-lg border border-slate-300 " +
                    "bg-white text-sm hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-40";
      return b;
    }

    // Swap two steps and rerender (used by the ↑ / ↓ reorder buttons).
    function swap(i, j) {
      if (j < 0 || j >= recipe.length) return;
      [recipe[i], recipe[j]] = [recipe[j], recipe[i]];
      renderRecipe();
    }

    // ===================================================================
    // The runner: execute the recipe top-to-bottom, piping outputs forward.
    // ===================================================================

    $("run-recipe").addEventListener("click", runRecipe);

    async function runRecipe() {
      if (recipe.length === 0) { status(T.needStep); return; }
      // Reset any prior outputs so a re-run starts clean.
      for (const s of recipe) {
        s.outBlob = s.outUrl = null; s.outName = ""; s.score = null;
        s.zipMembers = []; s.chosenMember = 0; s.error = null;
      }

      $("run-recipe").disabled = true;
      status(T.running);
      // The "current" file flowing through the pipeline. Starts as the source.
      let current = sourceFile;

      try {
        for (let i = 0; i < recipe.length; i++) {
          const step = recipe[i];
          const spec = OPS[step.op];
          status(T.stepProgress
            .replace("{index}", i + 1)
            .replace("{total}", recipe.length)
            .replace("{op}", step.op));

          // Bypass: skip the API entirely, pass the input straight through.
          if (step.bypass) {
            if (current) {
              step.outBlob = current;
              step.outUrl = URL.createObjectURL(current);
              step.outName = current.name || `step${i + 1}.bin`;
            }
            continue;
          }

          // silence takes no input; every other verb needs one.
          if (!spec.noInput && !current) {
            step.error = T.noInput;
            renderRecipe();
            status(T.stoppedNoInput);
            return;
          }

          // Build the multipart body this endpoint expects.
          const fd = new FormData();
          if (step.op === "concat") {
            // concat needs >= 2 files: the piped input plus every extra chosen.
            fd.append("files", current, current.name || "input.wav");
            for (const f of step.extraFiles) fd.append("files", f);
          } else if (step.op === "resemblance") {
            // resemblance compares the piped input (A) against one extra (B).
            const b = step.extraFiles[0];
            if (!b) { step.error = T.resemblanceNeedsB; renderRecipe(); status(T.stopped); return; }
            fd.append("a", current, current.name || "a.wav");
            fd.append("b", b);
          } else if (!spec.noInput) {
            fd.append("file", current, current.name || "input.wav");
          }
          // Append the verb's scalar parameters.
          for (const [k, v] of Object.entries(step.values)) fd.append(k, v);

          // Fire the request against the same endpoint the CLI uses.
          const res = await fetch(spec.endpoint, { method: "POST", body: fd });
          if (!res.ok) {
            const txt = await res.text();
            step.error = T.httpError
              .replace("{status}", res.status)
              .replace("{body}", txt.slice(0, 200));
            renderRecipe();
            status(T.stoppedAtStep.replace("{index}", i + 1));
            return;
          }

          // Interpret the response per verb kind.
          if (spec.kind === "score") {
            // resemblance is terminal: record the score, stop chaining.
            const j = await res.json();
            step.score = Number(j.score);
            current = null;   // nothing chainable downstream
          } else if (spec.kind === "zip") {
            // split / separate: unzip in the browser, expose members, chain one.
            const buf = new Uint8Array(await res.arrayBuffer());
            const files = fflate.unzipSync(buf);
            step.zipMembers = Object.entries(files).map(([name, bytes]) => {
              const blob = new Blob([bytes]);
              return { name, blob, url: URL.createObjectURL(blob) };
            });
            if (step.zipMembers.length === 0) {
              step.error = T.emptyArchive;
              renderRecipe(); status(T.stopped); return;
            }
            // Default to the first member for chaining (user can repick later).
            const m = step.zipMembers[step.chosenMember] || step.zipMembers[0];
            step.outBlob = m.blob; step.outUrl = m.url; step.outName = m.name;
            current = new File([m.blob], m.name);
          } else {
            // Single-file output: wrap the blob and pipe it forward.
            const blob = await res.blob();
            const name = `step${i + 1}.${step.values.output_format || "wav"}`;
            step.outBlob = blob;
            step.outUrl = URL.createObjectURL(blob);
            step.outName = name;
            current = new File([blob], name);
          }
        }
        status(T.recipeComplete);
      } catch (err) {
        status(T.recipeFailed.replace("{error}", err));
      } finally {
        $("run-recipe").disabled = false;
        renderRecipe();
        refreshComparatorChoices();
      }
    }

    // ===================================================================
    // recipe.yaml export / import (client-side, no server round-trip)
    // ===================================================================

    // Serialize the recipe to a minimal, human-diffable YAML document. We
    // hand-roll the emitter (the recipe is a shallow, well-known shape) to
    // avoid pulling a YAML library just for this.
    function recipeToYaml() {
      const lines = [];
      lines.push("# audio-helper recipe — replay these verbs in order.");
      lines.push("# Each step maps 1:1 to a CLI subcommand / API endpoint.");
      lines.push("version: 1");
      lines.push("steps:");
      for (const step of recipe) {
        lines.push(`  - op: ${step.op}`);
        if (step.bypass) lines.push(`    bypass: true`);
        const keys = Object.keys(step.values);
        if (keys.length) {
          lines.push(`    params:`);
          for (const k of keys) {
            // Quote values that could be misread as YAML (leading -, spaces…).
            const v = String(step.values[k]);
            const needsQuote = /^[\s-]|[:#]/.test(v) || v === "";
            lines.push(`      ${k}: ${needsQuote ? JSON.stringify(v) : v}`);
          }
        }
      }
      return lines.join("\n") + "\n";
    }

    // Export: download the current recipe as recipe.yaml.
    $("export-yaml").addEventListener("click", () => {
      if (recipe.length === 0) { status(T.nothingToExport); return; }
      const blob = new Blob([recipeToYaml()], { type: "text/yaml" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "recipe.yaml";
      a.click();
      status(T.exported);
    });

    // Parse a recipe.yaml back into recipe state. This is a deliberately small
    // parser matching exactly the shape recipeToYaml emits — enough to round-
    // trip a committed recipe without a full YAML dependency.
    function yamlToRecipe(text) {
      const out = [];
      let cur = null;
      let inParams = false;
      for (const raw of text.split("\n")) {
        const line = raw.replace(/\r$/, "");
        if (!line.trim() || line.trim().startsWith("#")) continue;
        // A new step begins with "  - op: <verb>".
        const opMatch = line.match(/^\s*-\s*op:\s*(\S+)\s*$/);
        if (opMatch) {
          cur = makeStep(opMatch[1]);
          cur.values = {};          // start empty; fill from the file
          out.push(cur);
          inParams = false;
          continue;
        }
        if (!cur) continue;
        // "    bypass: true"
        if (/^\s*bypass:\s*true\s*$/.test(line)) { cur.bypass = true; continue; }
        // "    params:" opens the param block.
        if (/^\s*params:\s*$/.test(line)) { inParams = true; continue; }
        // "      key: value" inside the param block.
        const kv = line.match(/^\s{6}(\w+):\s*(.*)$/);
        if (inParams && kv) {
          let v = kv[2];
          // Unquote JSON-quoted strings the emitter may have produced.
          if (v.startsWith('"') && v.endsWith('"')) { try { v = JSON.parse(v); } catch (e) {} }
          cur.values[kv[1]] = v;
        }
      }
      return out;
    }

    // Import: replace the recipe with the parsed file, then rerender.
    $("import-yaml").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        recipe = yamlToRecipe(await f.text());
        renderRecipe();
        status(T.imported
          .replace("{count}", recipe.length)
          .replace("{name}", f.name));
      } catch (err) {
        status(T.importFailed.replace("{error}", err));
      }
      // Reset the input so importing the same file twice still fires change.
      e.target.value = "";
    });

    // ===================================================================
    // On-demand spectrogram (Vega-Lite) of a produced clip.
    // ===================================================================

    // Compute a coarse magnitude spectrogram with the WebAudio FFT and render
    // it as a Vega-Lite heatmap. Rendered only when the user clicks — the
    // "spectrogram on every keystroke" idea stays a labelled roadmap item.
    async function renderSpectrogram(container, blob) {
      container.innerHTML = `<p class="text-xs text-slate-500">${T.computingSpectrogram}</p>`;
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audio = await ctx.decodeAudioData(await blob.arrayBuffer());
        const data = audio.getChannelData(0);
        // Window the signal into frames and take a magnitude spectrum per frame.
        const fftSize = 512, hop = 256, frames = Math.min(120, Math.floor((data.length - fftSize) / hop));
        const rows = [];
        for (let t = 0; t < frames; t++) {
          const start = t * hop;
          // Cheap DFT over a small band of bins — enough for a visual cue.
          for (let k = 0; k < 48; k++) {
            let re = 0, im = 0;
            for (let n = 0; n < fftSize; n += 4) {  // stride to keep it cheap
              const ang = (-2 * Math.PI * k * n) / fftSize;
              const s = data[start + n] || 0;
              re += s * Math.cos(ang); im += s * Math.sin(ang);
            }
            const mag = Math.log10(1 + Math.sqrt(re * re + im * im));
            rows.push({ time: t, freq: k, mag });
          }
        }
        await vegaEmbed(container, {
          $schema: "https://vega.github.io/schema/vega-lite/v5.json",
          width: "container", height: 120, data: { values: rows },
          mark: "rect",
          encoding: {
            x: { field: "time", type: "ordinal", axis: { labels: false, title: "time" } },
            y: { field: "freq", type: "ordinal", sort: "descending", axis: { labels: false, title: "freq" } },
            color: { field: "mag", type: "quantitative", scale: { scheme: "viridis" }, legend: null },
          },
        }, { actions: false });
      } catch (err) {
        container.innerHTML = `<p class="text-xs text-red-700">${T.spectrogramUnavailable.replace("{error}", err)}</p>`;
      }
    }

    // ===================================================================
    // Ear-first comparator: align two clips, toggle before/after on Space.
    // ===================================================================

    // Two WaveSurfer instances share one visible track; we mute/unmute to A/B.
    let cmpA = null, cmpB = null, cmpHearing = "a";

    // Gather every playable clip (source + each step's chainable output) so
    // the comparator dropdowns can offer them.
    function comparatorClips() {
      const clips = [];
      if (sourceFile && $("src-audio").src) clips.push({ label: T.clipSource, url: $("src-audio").src, blob: sourceFile });
      recipe.forEach((s, i) => {
        if (s.outUrl) clips.push({
          label: T.clipStep
            .replace("{index}", i + 1)
            .replace("{op}", s.op)
            .replace("{name}", s.outName),
          url: s.outUrl, blob: s.outBlob,
        });
      });
      return clips;
    }

    // Refill the before/after dropdowns, preserving the user's selection when
    // possible so a rerender does not reset their comparison.
    function refreshComparatorChoices() {
      const clips = comparatorClips();
      for (const id of ["cmp-a", "cmp-b"]) {
        const sel = $(id);
        const prev = sel.value;
        sel.innerHTML = "";
        clips.forEach((c, i) => {
          const o = document.createElement("option");
          o.value = String(i); o.textContent = c.label;
          sel.appendChild(o);
        });
        // Default: A = first clip, B = last clip (source vs final result).
        if (prev && Number(prev) < clips.length) sel.value = prev;
        else if (id === "cmp-b" && clips.length > 1) sel.value = String(clips.length - 1);
      }
    }

    // (Re)load both comparator players from the current dropdown selections.
    function loadComparator() {
      const clips = comparatorClips();
      if (clips.length < 1) return;
      const a = clips[Number($("cmp-a").value)] || clips[0];
      const b = clips[Number($("cmp-b").value)] || clips[0];
      if (cmpA) cmpA.destroy();
      if (cmpB) cmpB.destroy();
      // A is visible; B is drawn transparently on top so playheads align.
      cmpA = WaveSurfer.create({ container: "#cmp-wave", height: 64, waveColor: "#94a3b8", progressColor: "#2563eb", url: a.url });
      cmpB = WaveSurfer.create({ container: "#cmp-wave", height: 64, waveColor: "rgba(0,0,0,0)", progressColor: "rgba(0,0,0,0)", url: b.url });
      cmpHearing = "a";
      applyHearing();
      updateDiff(a.blob, b.blob);
    }

    // Mute whichever side we are NOT hearing, and update the label.
    function applyHearing() {
      if (!cmpA || !cmpB) return;
      cmpA.setVolume(cmpHearing === "a" ? 1 : 0);
      cmpB.setVolume(cmpHearing === "b" ? 1 : 0);
      $("cmp-which").textContent = T.hearing
        .replace("{side}", cmpHearing === "a" ? T.beforeLabel : T.afterLabel);
    }

    // Reload the comparator when either selection changes.
    $("cmp-a").addEventListener("change", loadComparator);
    $("cmp-b").addEventListener("change", loadComparator);

    // Play/pause both tracks together so the playheads stay aligned.
    $("cmp-play").addEventListener("click", () => {
      if (!cmpA || !cmpB) loadComparator();
      if (!cmpA) return;
      if (cmpA.isPlaying()) { cmpA.pause(); cmpB.pause(); }
      else { cmpA.play(); cmpB.play(); }
    });

    // Toggle which side we hear (button + Space bar share this).
    function toggleAB() {
      cmpHearing = cmpHearing === "a" ? "b" : "a";
      applyHearing();
    }
    $("cmp-toggle").addEventListener("click", toggleAB);
    // Space bar toggles A/B unless the user is typing in a form field.
    document.addEventListener("keydown", (e) => {
      if (e.code !== "Space") return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      e.preventDefault();
      if (!cmpA) loadComparator();
      toggleAB();
    });

    // Difference view toggle: reveal the canvas and (re)draw the diff.
    $("cmp-diff").addEventListener("change", (e) => {
      $("cmp-diff-wrap").hidden = !e.target.checked;
      if (e.target.checked) {
        const clips = comparatorClips();
        const a = clips[Number($("cmp-a").value)];
        const b = clips[Number($("cmp-b").value)];
        if (a && b) updateDiff(a.blob, b.blob);
      }
    });

    // Draw a coarse (after − before) waveform so tuning room-tone dB / Demucs
    // quality is visible, not just audible. Colorblind-safe: a labelled line,
    // no reliance on hue.
    async function updateDiff(beforeBlob, afterBlob) {
      if ($("cmp-diff-wrap").hidden || !beforeBlob || !afterBlob) return;
      const canvas = $("cmp-diff-canvas");
      const g = canvas.getContext("2d");
      canvas.width = canvas.clientWidth || 600;
      g.clearRect(0, 0, canvas.width, canvas.height);
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const [ba, bb] = await Promise.all([
          ctx.decodeAudioData(await beforeBlob.arrayBuffer()),
          ctx.decodeAudioData(await afterBlob.arrayBuffer()),
        ]);
        const da = ba.getChannelData(0), db = bb.getChannelData(0);
        const n = Math.min(da.length, db.length);
        const mid = canvas.height / 2;
        g.strokeStyle = "#dc2626"; g.beginPath();
        // Downsample to the canvas width; plot the per-pixel max |after-before|.
        const step = Math.max(1, Math.floor(n / canvas.width));
        for (let x = 0; x < canvas.width; x++) {
          let peak = 0;
          for (let k = 0; k < step; k++) {
            const idx = x * step + k;
            if (idx >= n) break;
            peak = Math.max(peak, Math.abs(da[idx] - db[idx]));
          }
          const y = peak * mid;
          g.moveTo(x, mid - y); g.lineTo(x, mid + y);
        }
        g.stroke();
      } catch (err) {
        g.fillStyle = "#334155"; g.font = "12px sans-serif";
        g.fillText(T.diffUnavailable.replace("{error}", err), 8, mid);
      }
    }

    // Initial render so the page shows the empty-state hint immediately.
    renderRecipe();
  </script>
</body>
</html>
"""


def render_gui(lang: str = "") -> str:
    """Render the single-page GUI in the requested language.

    Parameters
    ----------
    lang : str
        Two-letter language code. Anything outside the locale file's
        a compiled catalogue — including the empty string — falls back to
        :data:`_DEFAULT_LANGUAGE`, so an unknown ``?lang=`` never 500s.

    Returns
    -------
    str
        The complete HTML document for that language.

    Examples
    --------
    >>> "<!doctype html>" in render_gui("en")
    True
    >>> render_gui("tlh") == render_gui(default_language())
    True
    """
    cat = _catalog()
    resolved = cat.resolve(lang)
    translate = cat.translator_for(resolved).gettext

    # The other language, for the toggle. With two languages this is "the one
    # that is not current"; with more it is the next one round, which keeps the
    # flag a cycle rather than a dead end.
    languages = cat.languages
    other = languages[(languages.index(resolved) + 1) % len(languages)]

    def _render_source(match: re.Match[str]) -> str:
        """Translate one ``{{_("…")}}`` site and escape it for HTML.

        Escaping is not optional: these land in attributes (placeholder, alt,
        aria, title) as readily as in text nodes, and a French apostrophe in an
        attribute would close it early.
        """
        return html.escape(translate(match.group(1)), quote=True)

    page = _SOURCE_RE.sub(_render_source, _GUI_TEMPLATE)
    page = page.replace("{{LANG}}", html.escape(resolved, quote=True))
    page = page.replace("{{LANG_HREF}}", html.escape(f"/gui?lang={other}", quote=True))
    page = page.replace("{{LANG_FLAG}}", _FLAGS.get(other, other.upper()))
    page = page.replace("{{LANG_ARIA}}", html.escape(_ENDONYMS.get(other, other), quote=True))

    # The comparator badge starts out already saying something, built from the
    # very source string the script reuses on every toggle — one sentence, one
    # catalogue entry, rather than a second string that could drift from the
    # first. The `{side}` placeholder is filled after translation, because an
    # already-interpolated sentence matches no catalogue entry.
    initial_badge = translate("Hearing: {side}").format(side=translate("Before"))
    page = page.replace("{{CMP_HEARING_INIT}}", html.escape(initial_badge, quote=True))

    # The script's strings go in as JSON rather than as substituted text: it is
    # the encoding that cannot be broken out of by an apostrophe, and French
    # copy is full of them.
    table = {name: translate(source) for name, source in _JS_NAME_TO_SOURCE.items()}
    return page.replace("{{T_JSON}}", json.dumps(table, ensure_ascii=False))


def __getattr__(name: str) -> str:
    """Serve ``GUI_HTML`` on first access rather than building it at import.

    The module-level constant used to BE the page, and importers still reach
    for it. Assigning it eagerly would read and parse the locale file on every
    import of this module, including in callers that never serve the GUI — and
    a module-level ``__getattr__`` only fires for names that are NOT otherwise
    defined, so the name is deliberately left unbound rather than set to a
    placeholder.
    """
    if name == "GUI_HTML":
        return render_gui()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
