# audio-helper non-CLI surfaces

`audio-helper` exposes the same nine operations through five surfaces. The
Python library and argparse CLI are always available; the others live behind
optional extras.

## 1. Python library (default)

```python
import audio_helper as ah

ah.load_audio(path, target_sample_rate=None, to_mono=True, to_numpy=False)  # -> (array|tensor, sr)
ah.get_audio_duration(path)                      # -> float seconds
ah.is_valid_audio_file(path)                     # -> bool
ah.sound_converter(inp, out, freq=44100, channels=1, encoding="pcm_s16le")
ah.extract_audio_chunk(inp, start_time, end_time, output_audio_filename=None)
ah.generate_silent_audio(duration, output_audio_filename=None, sample_rate=44100)
ah.audio_concatenation([a, b, c], output_audio_filename=None)
ah.mix_room_tone(inp, output_audio=None, noise_db=-42, color="pink")
ah.split_audio_regularly(inp, chunk_folder, split_time, output_format="mp3")
ah.separate_sources(inp, output_folder=None, device=None, output_format="mp3")  # [demucs]
ah.sound_resemblance(a, b)                       # -> float in [0, 1]
ah.save_audio(signal, path, sample_rate=44100)
```

The public API is fixed via `audio_helper.__all__`; sibling repos (vocal-helper,
youtube-helper) depend on these names — treat them as stable.

## 2. CLI — argparse (default) and click

- **argparse** `audio-helper <sub> …` — ships with the base package, zero extra
  deps. Primary surface. See `cli-reference.md`.
- **click** `audio-helper-click <sub> …` — install `audio-helper[cli]`. Same
  subcommands and flag names; nicer `--help`, shell completion.

## 3. HTTP API — FastAPI (`audio-helper[api]`)

```bash
pip install 'audio-helper[api]'
uvicorn audio_helper.api:app --host 0.0.0.0 --port 8000
# OpenAPI docs: http://localhost:8000/docs
```

Endpoints (multipart `file` upload unless noted):
- `GET  /health` — liveness probe → `{"status":"ok"}`.
- `GET  /` — redirects to `/gui`.
- `GET  /gui` — the single-page GUI (see below).
- `POST /convert` — fields `output_format freq channels encoding` → file.
- `POST /duration` — → JSON `{"duration_seconds": …}`.
- `POST /chunk` — fields `start end output_format` → file.
- `POST /silence` — fields `duration sample_rate output_format` (no upload) → file.
- `POST /concat` — repeated `files` (≥2) + `output_format` → file.
- `POST /roomtone` — fields `db color sample_rate output_format` → file.
- `POST /split` — field `seconds output_format suffix` → **zip** of chunks.
- `POST /separate` — fields `device workers output_format` → **zip** of stems
  (503 if the `[demucs]` extra is missing).
- `POST /resemblance` — two uploads `a` and `b` → JSON `{"score": …}`.

Uploads stream to a temp file; temp dirs are cleaned via `BackgroundTasks`.

## 4. MCP server — FastAPI-MCP (`audio-helper[api,mcp]`)

```bash
pip install 'audio-helper[api,mcp]'
audio-helper-mcp                 # serves FastAPI + MCP on :8000
# or: python -m audio_helper.mcp
```

Wraps the exact FastAPI app with `fastapi_mcp` — the same endpoints become MCP
tools (`convert`, `chunk`, `split`, `separate`, …) for any MCP-aware host. Host
via `AUDIO_HELPER_HOST` / `AUDIO_HELPER_PORT` env vars.

## 5. GUI — minimal audition bench (`GET /gui`)

Served by the FastAPI app; no build step, no framework — a single self-contained
HTML page (Tailwind via CDN + vanilla ES-module JS) defined in
`audio_helper/gui.py`.

Workflow: drop/pick an audio file → choose an operation → fill only the fields
that operation needs → **Run** (POSTs to the same `/convert`, `/chunk`, … routes)
→ compare **input vs output** in two `<audio>` players and download the result
(a single file, or a `.zip` for `split` / `separate`; `resemblance` shows the
score inline).

```bash
uvicorn audio_helper.api:app --port 8000
# open http://localhost:8000/gui  (or just http://localhost:8000/)
```

This page is the canonical minimal-GUI template for the AI Helpers suite:
copy `gui.py`, swap the operation `<option>`s and per-op form fields, keep the
plumbing (drop zone, op→fields sync, fetch → player/zip/JSON rendering).
