"""
Audio Helper — FastAPI HTTP surface.

Exposes every public function in :mod:`audio_helper.main` as an HTTP
endpoint so `audio-helper` can be dropped behind any reverse proxy and
consumed by other services. Kept intentionally minimal:

- Multipart uploads for audio inputs (``UploadFile``), streamed straight
  to a temporary file so large clips don't blow up memory.
- ``FileResponse`` for single-file outputs (``/convert``, ``/chunk`` …).
- ZIP-encoded ``StreamingResponse`` for multi-file outputs
  (``/split``, ``/separate``) so the client gets one download per call.
- ``BackgroundTasks`` cleans temp files after the response has been
  streamed — no leftover garbage on disk after a request.

Install the extra to get the runtime dependencies::

    pip install 'audio-helper[api]'

Then run the app with any ASGI server::

    uvicorn audio_helper.api:app --host 0.0.0.0 --port 8000

Usage Example
-------------
>>> # Start the server:
>>> #   uvicorn audio_helper.api:app --reload
>>> # Convert a file:
>>> #   curl -F 'file=@in.mp3' -F 'freq=22050' -F 'channels=1' \\
>>> #        -o out.wav http://localhost:8000/convert
>>> # Get a file's duration (JSON):
>>> #   curl -F 'file=@in.mp3' http://localhost:8000/duration
>>> # Full OpenAPI docs at http://localhost:8000/docs

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The FastAPI HTTP surface requires the [api] extra. "
        "Install with: pip install 'audio-helper[api]'"
    ) from exc

from . import (
    audio_concatenation,
    extract_audio_chunk,
    generate_silent_audio,
    get_audio_duration,
    mix_room_tone,
    separate_sources,
    sound_converter,
    sound_resemblance,
    split_audio_regularly,
)

# ---------------------------------------------------------------------------
# App factory + shared plumbing
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Audio Helper API",
    description=(
        "HTTP surface for the audio-helper utilities: load, convert, split, "
        "concatenate, silence, room-tone, MFCC similarity, Demucs source separation."
    ),
    version="1.5.5",
    docs_url="/docs",
    redoc_url="/redoc",
)


def _spool(upload: UploadFile, dest_dir: Path, suffix_hint: str | None = None) -> Path:
    """
    Persist an ``UploadFile`` to a temp path on disk.

    We copy the stream instead of holding the bytes in memory so a
    multi-hundred-MB clip does not OOM the worker. The file inherits
    the caller's suffix when available so ffmpeg picks the right
    demuxer.

    Parameters
    ----------
    upload : UploadFile
        The FastAPI upload object.
    dest_dir : Path
        Temp directory that will hold the spooled file.
    suffix_hint : str, optional
        Extension override (with or without the leading dot). Falls back
        to the client-provided filename's suffix.

    Returns
    -------
    Path
        Path to the spooled file on disk.
    """
    # Preserve the caller's extension so ffmpeg selects the right demuxer;
    # ``.bin`` is a last-resort fallback when nothing else is known.
    ext = suffix_hint or (Path(upload.filename or "").suffix or ".bin")
    if not ext.startswith("."):
        ext = "." + ext
    out = dest_dir / (f"upload{ext}")
    # Stream the upload straight to disk (never load it fully into memory) so
    # a large clip cannot OOM the worker process.
    with out.open("wb") as fp:
        shutil.copyfileobj(upload.file, fp)
    return out


def _cleanup(*paths: Path | str) -> None:
    """Best-effort cleanup — never let a tidy-up failure kill a response."""
    for p in paths:
        try:
            path = Path(p)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
        except Exception:
            pass


def _new_tmpdir() -> Path:
    """Create a request-scoped temp directory under the system temp root."""
    return Path(tempfile.mkdtemp(prefix="audio-helper-"))


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Simple liveness probe — no dependency check, just proves the app is up."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@app.post("/convert", tags=["actions"])
def convert(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Form("wav", description="Target container extension (wav / mp3 / …)."),
    freq: int = Form(44100),
    channels: int = Form(1),
    encoding: str = Form("pcm_s16le"),
) -> FileResponse:
    """Re-encode an uploaded file at the requested sample rate / channels / codec."""
    # Every action endpoint follows the same shape: spool the upload to a
    # request-scoped temp dir, run the pure library function on real paths
    # (the library speaks files, not streams), then hand the result back and
    # schedule the temp dir for deletion once the response has been sent.
    tmp = _new_tmpdir()
    src = _spool(file, tmp)
    # The output extension drives the container ffmpeg writes, so derive it
    # from the requested format rather than hard-coding one.
    dst = tmp / f"converted.{output_format.lstrip('.')}"
    sound_converter(
        input_audio=str(src),
        output_audio=str(dst),
        freq=freq,
        channels=channels,
        encoding=encoding,
        overwrite=True,
    )
    # Clean the whole temp dir after the response has been sent.
    background.add_task(_cleanup, tmp)
    return FileResponse(str(dst), filename=dst.name, media_type="application/octet-stream")


@app.post("/duration", tags=["reads"])
def duration(file: UploadFile = File(...), background: BackgroundTasks = None) -> JSONResponse:
    """Return the duration in seconds of the uploaded audio file."""
    tmp = _new_tmpdir()
    try:
        src = _spool(file, tmp)
        seconds = get_audio_duration(str(src))
    finally:
        # Duration is a JSON response — clean synchronously, no background needed.
        _cleanup(tmp)
    return JSONResponse({"duration_seconds": seconds})


@app.post("/chunk", tags=["actions"])
def chunk(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    output_format: str = Form("mp3"),
) -> FileResponse:
    """Extract a ``[start, end]`` slice from the uploaded audio."""
    # Same spool-run-cleanup shape as /convert; the slice bounds come
    # straight from the multipart form fields.
    tmp = _new_tmpdir()
    src = _spool(file, tmp)
    dst = tmp / f"chunk.{output_format.lstrip('.')}"
    extract_audio_chunk(
        audio_file=str(src),
        start_time=start,
        end_time=end,
        output_audio_filename=str(dst),
        overwrite=True,
    )
    background.add_task(_cleanup, tmp)
    return FileResponse(str(dst), filename=dst.name, media_type="application/octet-stream")


@app.post("/silence", tags=["actions"])
def silence(
    background: BackgroundTasks,
    duration_seconds: float = Form(..., alias="duration"),
    sample_rate: int = Form(44100),
    output_format: str = Form("wav"),
) -> FileResponse:
    """Generate a silent audio file of a given duration."""
    # No upload to spool here — silence is synthesized — but we still need a
    # temp dir to hold the generated file until the response is streamed.
    tmp = _new_tmpdir()
    dst = tmp / f"silence.{output_format.lstrip('.')}"
    generate_silent_audio(
        duration=duration_seconds,
        output_audio_filename=str(dst),
        sample_rate=sample_rate,
        overwrite=True,
    )
    background.add_task(_cleanup, tmp)
    return FileResponse(str(dst), filename=dst.name, media_type="application/octet-stream")


@app.post("/concat", tags=["actions"])
def concat(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    output_format: str = Form("mp3"),
) -> FileResponse:
    """Concatenate multiple uploaded audio files head-to-tail."""
    # Concatenating a single file is a no-op the caller almost never means;
    # reject it early with a 400 rather than returning the input unchanged.
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="concat needs at least 2 files")
    tmp = _new_tmpdir()
    # Spool inputs in order — FastAPI preserves multipart part ordering.
    srcs = [str(_spool(f, tmp, suffix_hint=Path(f.filename or "").suffix)) for f in files]
    dst = tmp / f"concat.{output_format.lstrip('.')}"
    audio_concatenation(audio_files=srcs, output_audio_filename=str(dst), overwrite=True)
    background.add_task(_cleanup, tmp)
    return FileResponse(str(dst), filename=dst.name, media_type="application/octet-stream")


@app.post("/roomtone", tags=["actions"])
def roomtone(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: float = Form(-42.0, description="Noise level in dB (typical range -45..-38)."),
    color: str = Form("pink", description="white | pink | brown | red | blue | violet | velvet"),
    sample_rate: int = Form(44100),
    output_format: str = Form("wav"),
) -> FileResponse:
    """Mix low-level colored ambient noise on top of an uploaded speech track."""
    # Room tone defaults to WAV output because the mix is meant to feed back
    # into an edit; a lossy container would defeat the point of a clean bed.
    tmp = _new_tmpdir()
    src = _spool(file, tmp)
    dst = tmp / f"roomtone.{output_format.lstrip('.')}"
    mix_room_tone(
        input_audio=str(src),
        output_audio=str(dst),
        noise_db=db,
        color=color,
        sample_rate=sample_rate,
        overwrite=True,
    )
    background.add_task(_cleanup, tmp)
    return FileResponse(str(dst), filename=dst.name, media_type="application/octet-stream")


def _zip_folder(folder: Path) -> io.BytesIO:
    """Bundle ``folder``'s contents into an in-memory ZIP for streaming."""
    # Build the ZIP entirely in memory: the multi-file outputs are small
    # (a handful of stems/chunks) so this avoids a second round-trip to disk.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Store paths relative to ``folder`` so the archive has clean names
        # (``vocals.mp3``) instead of leaking the temp-dir prefix.
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(folder))
    # Rewind so the StreamingResponse reads from the start of the buffer.
    buf.seek(0)
    return buf


@app.post("/split", tags=["actions"])
def split(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    seconds: float = Form(..., description="Chunk duration in seconds."),
    output_format: str = Form("mp3"),
    suffix: str = Form("split"),
) -> StreamingResponse:
    """Split the uploaded audio into fixed-duration chunks; response is a ZIP."""
    # Chunks land in their own subdir so ``_zip_folder`` bundles only the
    # generated pieces, not the spooled source that sits alongside them.
    tmp = _new_tmpdir()
    src = _spool(file, tmp)
    chunks_dir = tmp / "chunks"
    chunks_dir.mkdir()
    split_audio_regularly(
        sound_path=str(src),
        chunk_folder=str(chunks_dir),
        split_time=seconds,
        output_format=output_format,
        overwrite=True,
        suffix=suffix,
    )
    buf = _zip_folder(chunks_dir)
    background.add_task(_cleanup, tmp)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="chunks.zip"'},
    )


@app.post("/separate", tags=["actions"])
def separate(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    device: str | None = Form(None, description="'cuda' / 'cpu' / None (auto)."),
    workers: int = Form(-2),
    output_format: str = Form("mp3"),
) -> StreamingResponse:
    """Run Demucs source separation; response is a ZIP with the 4 stems."""
    # Isolate the stems in their own subdir (same reasoning as /split).
    tmp = _new_tmpdir()
    src = _spool(file, tmp)
    stems_dir = tmp / "stems"
    stems_dir.mkdir()
    # Demucs lives behind the optional [demucs] extra. If torch is missing
    # the library raises ImportError; translate it into a clean 503 (service
    # unavailable) instead of a 500 so clients can tell it is a config gap.
    try:
        separate_sources(
            input_audio_file=str(src),
            output_folder=str(stems_dir),
            device=device,
            overwrite=True,
            nb_workers=workers,
            output_format=output_format,
        )
    except ImportError as exc:
        _cleanup(tmp)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    buf = _zip_folder(stems_dir)
    background.add_task(_cleanup, tmp)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="stems.zip"'},
    )


@app.post("/resemblance", tags=["reads"])
def resemblance(a: UploadFile = File(...), b: UploadFile = File(...)) -> JSONResponse:
    """MFCC-based similarity score in ``[0, 1]`` between two uploaded audio files."""
    tmp = _new_tmpdir()
    try:
        pa = _spool(a, tmp, suffix_hint=Path(a.filename or "").suffix or ".wav")
        # Second upload needs a different filename or we would overwrite the first spooled file.
        pb = tmp / (f"upload_b{Path(b.filename or '').suffix or '.wav'}")
        with pb.open("wb") as fp:
            shutil.copyfileobj(b.file, fp)
        score = sound_resemblance(str(pa), str(pb))
    finally:
        _cleanup(tmp)
    return JSONResponse({"score": float(score)})
