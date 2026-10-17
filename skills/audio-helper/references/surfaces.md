# audio-helper non-CLI surfaces

## 1. Library

```python
import audio_helper as ah
```

Public API (`audio_helper.__all__`): `is_valid_audio_file`,
`get_audio_duration`, `load_audio`, `save_audio`, `sound_converter`,
`separate_sources`, `extract_audio_chunk`, `generate_silent_audio`,
`audio_concatenation`, `mix_room_tone`, `split_audio_regularly`,
`sound_resemblance`.

`load_audio` returns samples plus the sample rate, so numeric work stays in
NumPy; `save_audio` writes them back. The other functions are file-to-file and
let ffmpeg do the work without materialising the signal in Python.

Logging and file management route through `os_helper` — the suite convention.

## 2. CLIs — argparse (core) and click (`[cli]`)

`audio-helper` and `audio-helper-click`. See `cli-reference.md`.

## 3. HTTP API — FastAPI (`[api]`)

```bash
pip install 'audio-helper[api]'
python -m audio_helper.api          # 127.0.0.1:8000, OpenAPI docs at /docs
```

| Method | Route |
|---|---|
| GET | `/health`, `/`, `/gui` |
| POST | `/convert`, `/duration`, `/chunk`, `/split`, `/concat` |
| POST | `/silence`, `/roomtone`, `/separate`, `/resemblance` |

Uploads are multipart; responses stream the produced file back, or return JSON
for the measuring routes. Use the API when another service needs the transforms
over HTTP.

## 4. MCP — `fastapi-mcp` over the *same* app (`[mcp]`)

```bash
pip install 'audio-helper[mcp]'
audio-helper-mcp                    # same app, MCP mounted at /mcp
```

`audio_helper.mcp` takes the FastAPI app from `audio_helper.api` and mounts
`FastApiMCP` on it. Every MCP tool *is* an HTTP route, so the two surfaces
cannot drift and one process serves both doors.

Register it with Claude Code:

```bash
claude mcp add audio-helper --transport http http://127.0.0.1:8000/mcp
```

## 5. Recipe Canvas GUI (`[api]`)

`GET /gui` — drop a file, pick an operation, get the result. Reach for it when
a human wants to try a few transforms interactively rather than script them.
See the repo's `GUI.md`.
