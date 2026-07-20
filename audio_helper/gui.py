"""
Audio Helper — minimal single-page GUI ("audition bench").

This module holds nothing but the self-contained HTML document served by the
FastAPI app at ``GET /gui`` (see :mod:`audio_helper.api`). It is deliberately
build-step-free: one string of HTML + Tailwind (via CDN) + vanilla ES-module
JavaScript. There is no bundler, no framework, no npm — the whole page is a
static asset the API returns verbatim.

Why a separate module
----------------------
Keeping the (long) HTML out of :mod:`audio_helper.api` keeps the route
definitions readable and lets other repos in the AI Helpers suite copy this
file almost verbatim as their GUI template: swap the operation list and the
per-operation form fields, keep the plumbing.

What the page does
------------------
- Drop / pick a local audio file (kept entirely client-side).
- Choose one operation (convert / chunk / silence / concat / roomtone /
  split / separate / resemblance).
- Reveal only the fields that operation needs.
- POST a ``multipart/form-data`` request to the SAME FastAPI endpoints the
  CLI and MCP surfaces use — the GUI adds zero new server logic.
- Play the input and the output side by side in ``<audio>`` players and
  offer a download link for the result (single file, or a ``.zip`` for the
  multi-file ``split`` / ``separate`` operations).

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

# The entire GUI is this one HTML string. It is returned as-is by the
# ``/gui`` route. Tailwind is pulled from a CDN so there is no build step;
# the JavaScript is a single inline ES module talking to the existing API.
GUI_HTML: str = r"""<!doctype html>
<html lang="en" class="h-full">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audio Helper — Audition Bench</title>
  <!-- Tailwind via CDN: keeps the page a single self-contained file, no build. -->
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    /* Respect users who ask for reduced motion (accessibility baseline). */
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  </style>
</head>
<body class="h-full bg-slate-50 text-slate-900 antialiased">
  <div class="mx-auto max-w-3xl px-4 py-8">
    <header class="mb-6">
      <h1 class="text-2xl font-semibold tracking-tight">Audio Helper — Audition Bench</h1>
      <p class="mt-1 text-sm text-slate-600">
        Drop an audio file, pick an operation, run it on the local API,
        then compare input vs output and download the result.
      </p>
    </header>

    <!-- 1) File input: drag-and-drop zone doubling as a file picker. -->
    <section class="mb-5">
      <label for="file" class="block text-sm font-medium mb-1">Input audio file</label>
      <div id="drop" tabindex="0"
           class="flex flex-col items-center justify-center rounded-xl border-2 border-dashed
                  border-slate-300 bg-white px-4 py-8 text-center cursor-pointer
                  focus:outline-none focus:ring-2 focus:ring-blue-500 hover:border-blue-400">
        <p class="text-sm text-slate-500">Drop a file here, or click to choose</p>
        <p id="filename" class="mt-2 text-sm font-medium text-slate-800"></p>
        <input id="file" type="file" accept="audio/*,video/*" class="hidden" />
      </div>
    </section>

    <!-- 2) Operation selector. Changing it reveals only the relevant fields. -->
    <section class="mb-5">
      <label for="op" class="block text-sm font-medium mb-1">Operation</label>
      <select id="op"
              class="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-500">
        <option value="convert">convert — re-encode (freq / channels / codec)</option>
        <option value="chunk">chunk — extract a [start, end] slice</option>
        <option value="silence">silence — generate N seconds of silence</option>
        <option value="concat">concat — join multiple files head-to-tail</option>
        <option value="roomtone">roomtone — mix low-level ambient noise</option>
        <option value="split">split — fixed-duration chunks (zip)</option>
        <option value="separate">separate — Demucs stems (zip)</option>
        <option value="resemblance">resemblance — MFCC similarity score</option>
      </select>
    </section>

    <!-- 3) Per-operation parameter fields. Shown/hidden by data-ops list. -->
    <section id="params" class="mb-5 grid grid-cols-2 gap-3">
      <div data-ops="convert chunk silence split separate roomtone">
        <label class="block text-xs font-medium mb-1">output_format</label>
        <input id="output_format" value="wav"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="convert">
        <label class="block text-xs font-medium mb-1">freq (Hz)</label>
        <input id="freq" type="number" value="44100"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="convert">
        <label class="block text-xs font-medium mb-1">channels</label>
        <input id="channels" type="number" value="1"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="chunk">
        <label class="block text-xs font-medium mb-1">start (s)</label>
        <input id="start" type="number" step="0.01" value="0"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="chunk">
        <label class="block text-xs font-medium mb-1">end (s)</label>
        <input id="end" type="number" step="0.01" value="5"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="silence">
        <label class="block text-xs font-medium mb-1">duration (s)</label>
        <input id="duration" type="number" step="0.1" value="3"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="split">
        <label class="block text-xs font-medium mb-1">seconds / chunk</label>
        <input id="seconds" type="number" step="0.1" value="10"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="roomtone">
        <label class="block text-xs font-medium mb-1">db</label>
        <input id="db" type="number" step="1" value="-42"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="roomtone">
        <label class="block text-xs font-medium mb-1">color</label>
        <input id="color" value="pink"
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
      <div data-ops="concat resemblance" class="col-span-2">
        <label class="block text-xs font-medium mb-1">
          extra file(s) — this operation needs a second (concat: more) file
        </label>
        <input id="extra" type="file" accept="audio/*,video/*" multiple
               class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
      </div>
    </section>

    <!-- 4) Run button + status line. -->
    <section class="mb-6">
      <button id="run"
              class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white
                     hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500
                     disabled:opacity-50">
        Run
      </button>
      <span id="status" class="ml-3 text-sm text-slate-600" role="status" aria-live="polite"></span>
    </section>

    <!-- 5) Players + result. Input on the left, output on the right. -->
    <section class="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <div class="rounded-xl border border-slate-200 bg-white p-4">
        <h2 class="mb-2 text-sm font-medium">Input</h2>
        <audio id="in-audio" controls class="w-full"></audio>
      </div>
      <div class="rounded-xl border border-slate-200 bg-white p-4">
        <h2 class="mb-2 text-sm font-medium">Output</h2>
        <audio id="out-audio" controls class="w-full"></audio>
        <div id="out-extra" class="mt-2 text-sm"></div>
        <a id="download" class="mt-2 inline-block text-sm font-medium text-blue-600 hover:underline"
           hidden download>Download result</a>
      </div>
    </section>
  </div>

  <script type="module">
    // --- tiny DOM helpers -------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const status = (msg) => { $("status").textContent = msg; };

    // Currently-selected primary input file (kept client-side until Run).
    let inputFile = null;

    // --- file picker + drag-and-drop -------------------------------------
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
      if (e.dataTransfer.files.length) setInput(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => { if (fileInput.files.length) setInput(fileInput.files[0]); });

    // Register a chosen file: show its name and load it into the input player.
    function setInput(f) {
      inputFile = f;
      $("filename").textContent = f.name;
      $("in-audio").src = URL.createObjectURL(f);
    }

    // --- operation -> visible fields -------------------------------------
    const opSelect = $("op");
    // Show only the parameter blocks whose data-ops list contains the op.
    function syncParams() {
      const op = opSelect.value;
      for (const el of document.querySelectorAll("#params [data-ops]")) {
        el.hidden = !el.dataset.ops.split(" ").includes(op);
      }
    }
    opSelect.addEventListener("change", syncParams);
    syncParams();

    // --- run: build the multipart request per operation ------------------
    // Endpoints returning a single file vs a zip; drives how we render output.
    const ZIP_OPS = new Set(["split", "separate"]);
    // Operations that return JSON (a number), not an audio file.
    const JSON_OPS = new Set(["resemblance"]);

    $("run").addEventListener("click", async () => {
      const op = opSelect.value;
      const fd = new FormData();
      // Most ops take the primary file under `file`; silence takes none.
      if (op !== "silence") {
        if (!inputFile) { status("Pick an input file first."); return; }
      }

      // Assemble the form fields expected by each API endpoint.
      let url = "/" + op;
      if (op === "convert") {
        fd.append("file", inputFile);
        fd.append("output_format", $("output_format").value);
        fd.append("freq", $("freq").value);
        fd.append("channels", $("channels").value);
      } else if (op === "chunk") {
        fd.append("file", inputFile);
        fd.append("start", $("start").value);
        fd.append("end", $("end").value);
        fd.append("output_format", $("output_format").value);
      } else if (op === "silence") {
        fd.append("duration", $("duration").value);
        fd.append("output_format", $("output_format").value);
      } else if (op === "concat") {
        // concat needs >= 2 files: the primary plus every extra selected.
        fd.append("files", inputFile);
        for (const f of $("extra").files) fd.append("files", f);
        fd.append("output_format", $("output_format").value);
      } else if (op === "roomtone") {
        fd.append("file", inputFile);
        fd.append("db", $("db").value);
        fd.append("color", $("color").value);
        fd.append("output_format", $("output_format").value);
      } else if (op === "split") {
        fd.append("file", inputFile);
        fd.append("seconds", $("seconds").value);
        fd.append("output_format", $("output_format").value);
      } else if (op === "separate") {
        fd.append("file", inputFile);
        fd.append("output_format", $("output_format").value);
      } else if (op === "resemblance") {
        // resemblance compares two files: primary as `a`, first extra as `b`.
        const b = $("extra").files[0];
        if (!b) { status("resemblance needs a second file in 'extra'."); return; }
        fd.append("a", inputFile);
        fd.append("b", b);
      }

      // Fire the request and render the response by response type.
      status("Running…");
      $("run").disabled = true;
      try {
        const res = await fetch(url, { method: "POST", body: fd });
        if (!res.ok) {
          const txt = await res.text();
          status("Error " + res.status + ": " + txt.slice(0, 200));
          return;
        }
        if (JSON_OPS.has(op)) {
          // resemblance returns {"score": <float>}.
          const j = await res.json();
          $("out-audio").removeAttribute("src");
          $("out-extra").textContent = "score: " + (j.score ?? JSON.stringify(j));
          $("download").hidden = true;
          status("Done.");
          return;
        }
        // Binary response (audio file or zip): wrap in an object URL.
        const blob = await res.blob();
        const objUrl = URL.createObjectURL(blob);
        const dl = $("download");
        dl.href = objUrl;
        if (ZIP_OPS.has(op)) {
          // Multi-file output: no inline player, just a zip download.
          dl.download = op + ".zip";
          $("out-audio").removeAttribute("src");
          $("out-extra").textContent = "Multiple files bundled as a .zip.";
        } else {
          // Single audio file: play it inline and offer the download.
          dl.download = "output." + ($("output_format").value || "wav");
          $("out-audio").src = objUrl;
          $("out-extra").textContent = "";
        }
        dl.hidden = false;
        status("Done.");
      } catch (err) {
        status("Request failed: " + err);
      } finally {
        $("run").disabled = false;
      }
    });
  </script>
</body>
</html>
"""
