"""
Audio Helper: Model Context Protocol (MCP) surface.

A thin adapter that exposes the FastAPI app from :mod:`audio_helper.api` as MCP
tools, so any MCP-aware host (an agent runtime, an IDE integration, a custom
shell) can call the audio-helper actions — convert, chunk, silence, concat,
room tone, split, Demucs source separation, duration, MFCC resemblance — as
first-class tools. Uses `fastapi-mcp`
(https://github.com/tadata-org/fastapi_mcp): one wrapper publishes the whole
existing HTTP surface, so the routes are never duplicated.

Install the extra to pull in ``fastapi-mcp``::

    pip install "audio-helper[mcp]"

Then run the server (HTTP API + MCP endpoint at ``/mcp``)::

    audio-helper-mcp                 # console entry point
    python -m audio_helper.mcp       # equivalent

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

try:
    from fastapi_mcp import FastApiMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'The MCP surface needs the [mcp] extra: pip install "audio-helper[mcp]"'
    ) from exc

# Reuse the exact same FastAPI app: MCP is a thin wrapper on top, no new routes.
from audio_helper.api import app

# Publish the HTTP endpoints (convert / chunk / silence / concat / roomtone /
# split / separate / duration / resemblance) as MCP tools.
mcp = FastApiMCP(
    app,
    name="audio-helper",
    description=(
        "audio-helper MCP tools: load, convert, chunk, concatenate, split, "
        "generate silence, mix room tone, run Demucs source separation, "
        "measure duration, and score MFCC-based resemblance between audio "
        "files — entirely on the local machine."
    ),
)
# Newer fastapi-mcp splits mount() into transport-specific mount_http(); fall back to
# the legacy mount() so a range of fastapi-mcp versions keeps working.
if hasattr(mcp, "mount_http"):
    mcp.mount_http()
else:  # pragma: no cover - legacy fastapi-mcp
    mcp.mount()


def main() -> None:
    """Console entry point (``audio-helper-mcp``): serve the API + MCP endpoint.

    Boots the FastAPI app (now serving both the plain HTTP routes and the
    ``/mcp`` MCP endpoint) with uvicorn in a single worker. Local-first: binds
    to loopback by default (override with ``AUDIO_HELPER_HOST`` /
    ``AUDIO_HELPER_PORT``).
    """
    import os

    import uvicorn

    host = os.environ.get("AUDIO_HELPER_HOST", "127.0.0.1")
    port = int(os.environ.get("AUDIO_HELPER_PORT", "8001"))
    print(f"Audio Helper API + MCP -> http://{host}:{port}  (MCP at /mcp)")
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":  # pragma: no cover
    main()
