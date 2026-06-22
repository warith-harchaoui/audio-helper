"""
audio_helper.streaming
======================

Async streaming counterpart to the batch I/O in :mod:`audio_helper.main`.
Yields ``float32`` mono PCM frames at a fixed sample rate from local
files (with real-time pacing), arbitrary ffmpeg-readable URLs, YouTube
(live or VOD via ``yt_helper``), and HTTP chunked podcast feeds.

Pure I/O — no model dependencies. Designed to land upstream as a
follow-up PR on ``warith-harchaoui/audio-helper``.

Author
------
Warith Harchaoui — https://www.linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import AsyncIterator, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Public types.
# ---------------------------------------------------------------------------


class PcmFrame(TypedDict):
    """One mono float32 PCM frame in absolute stream time.

    Keys
    ----
    t_abs_s : float
        Seconds since the source started. Monotonic — preserved across
        reconnects via silence padding.
    pcm : NDArray[np.float32]
        Mono samples, shape ``(n_samples,)``, range ``[-1, 1]``.
    voiced : bool | None
        Always ``None`` here — VAD downstream fills it in.
    """

    t_abs_s: float
    pcm: NDArray[np.float32]
    voiced: bool | None


SourceKind = Literal["file", "ffmpeg", "youtube", "podcast"]


# ---------------------------------------------------------------------------
# Module logger.
# ---------------------------------------------------------------------------
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


async def iter_pcm(
    uri: str,
    *,
    kind: SourceKind = "file",
    target_sample_rate: int = 16000,
    to_mono: bool = True,
    realtime: bool = True,
    frame_ms: int = 20,
    reconnect: bool = True,
    silence_on_gap: bool = True,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[PcmFrame]:
    """Asynchronously yield PCM frames from ``uri``.

    Parameters
    ----------
    uri : str
        File path (kind=``"file"``) or URL (other kinds).
    kind : SourceKind
        Which decode path to use. ``"file"`` is implemented in Phase 1;
        the other kinds land in Phase 6 (see ``.private/PLAN.md``).
    target_sample_rate : int
        Resample target. The whole pipeline runs at 16 kHz mono by
        default.
    to_mono : bool
        Force mono. Multi-channel sources are downmixed by ffmpeg.
    realtime : bool
        For ``kind="file"``, pass ``-re`` to ffmpeg so playback is paced
        to wall-clock — makes downstream behaviour match a live stream.
    frame_ms : int
        Frame length in milliseconds. ``20`` is the default Silero VAD
        frame size, which avoids a re-buffer at the VAD stage.
    reconnect : bool
        Phase 9: enable exponential-backoff reconnect when the source
        dies. Currently logged but not yet implemented.
    silence_on_gap : bool
        Phase 9: pad silence for the wall-clock duration of any
        reconnect gap so ``t_abs_s`` remains monotonic.
    headers : dict[str, str] | None
        HTTP headers passed through to ffmpeg (``-headers`` flag).
        Required for some live URLs (YouTube live tracks need the
        specific ``User-Agent`` / ``Referer`` ``yt-helper`` returned).
        Joined with ``\\r\\n`` per the ffmpeg ``-headers`` convention.

    Yields
    ------
    PcmFrame
        Successive frames in absolute stream time.

    Raises
    ------
    NotImplementedError
        For ``kind`` values not yet implemented.
    """
    # Dispatch on kind so each path can have its own ffmpeg / network glue
    # without the parent caring.
    if kind == "file":
        async for frame in _iter_pcm_via_ffmpeg(
            uri,
            target_sample_rate=target_sample_rate,
            to_mono=to_mono,
            realtime=realtime,
            frame_ms=frame_ms,
            headers=headers,
        ):
            yield frame
        return
    if kind in ("ffmpeg", "podcast"):
        # Both kinds delegate to ffmpeg with realtime=False (the source is
        # already live or self-paced; ``-re`` would needlessly throttle).
        async for frame in _iter_pcm_via_ffmpeg(
            uri,
            target_sample_rate=target_sample_rate,
            to_mono=to_mono,
            realtime=False,
            frame_ms=frame_ms,
            headers=headers,
        ):
            yield frame
        return
    if kind == "youtube":
        # YouTube resolution belongs in yt_helper.streaming; this layer
        # only does PCM decode. The caller should resolve first and pass
        # ``kind="ffmpeg"`` with the resolved URL.
        raise NotImplementedError(
            "audio_helper.streaming kind='youtube' is intentionally "
            "not implemented here — use yt_helper.streaming."
            "resolve_direct_url(...) to get a direct media URL, then "
            "pass it back here with kind='ffmpeg'."
        )
    raise ValueError(f"Unknown source kind: {kind!r}")


# ---------------------------------------------------------------------------
# ffmpeg-decode helper. Used by every kind that needs PCM out of an
# arbitrary URL or local file — only ``realtime`` and ``headers`` differ
# between the kinds.
# ---------------------------------------------------------------------------


async def _iter_pcm_via_ffmpeg(
    input_uri: str,
    *,
    target_sample_rate: int,
    to_mono: bool,
    realtime: bool,
    frame_ms: int,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[PcmFrame]:
    """Decode any ffmpeg-readable input into mono float32 frames.

    Uses ``ffmpeg -re`` (when ``realtime`` is True) so the input plays at
    wall-clock speed — relevant when the source is a local file we want
    to pretend is live. For real live sources (``ffmpeg`` / ``podcast``)
    callers pass ``realtime=False`` since the source paces itself.
    """
    # Number of samples per yielded frame.
    samples_per_frame = max(1, (target_sample_rate * frame_ms) // 1000)
    # ffmpeg emits raw 32-bit float, little-endian, planar/mono. One
    # sample is 4 bytes.
    bytes_per_sample = 4
    bytes_per_frame = samples_per_frame * bytes_per_sample

    # Build the ffmpeg command. We use -loglevel error so the stderr
    # stream stays quiet under normal operation and only carries real
    # problems (which we forward to the logger).
    channels = 1 if to_mono else 2
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if realtime:
        # -re makes ffmpeg read the input at its native frame rate,
        # i.e. wall-clock. Without it, files decode as fast as the disk
        # allows and the downstream pipeline sees burst input.
        cmd += ["-re"]
    if headers:
        # ffmpeg's -headers flag takes a single string with CRLF
        # separators. Required for some YouTube live URLs that 403 on
        # the default UA.
        headers_str = "\r\n".join(f"{k}: {v}" for k, v in headers.items())
        cmd += ["-headers", headers_str + "\r\n"]
    cmd += [
        "-i",
        input_uri,
        "-ac",
        str(channels),
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "-",
    ]

    _log.debug("Launching ffmpeg: %s", shlex.join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # stdout / stderr are guaranteed to be set because we asked for PIPE.
    assert proc.stdout is not None
    assert proc.stderr is not None

    # t_abs_s is incremented per yielded frame; we anchor at 0.0 (the
    # downstream pipeline owns the abs-time origin via pdbms.clock).
    t_abs_s = 0.0
    seconds_per_frame = samples_per_frame / target_sample_rate

    try:
        while True:
            # readexactly raises IncompleteReadError on EOF; that's our
            # signal to stop.
            try:
                raw = await proc.stdout.readexactly(bytes_per_frame)
            except asyncio.IncompleteReadError as exc:
                # If ffmpeg ended mid-frame we still want to surface what
                # it gave us (zero-padded). Keeps cuts at EOF clean.
                if exc.partial:
                    _log.debug(
                        "ffmpeg EOF with %d partial bytes (< %d expected); "
                        "padding with silence to complete final frame.",
                        len(exc.partial),
                        bytes_per_frame,
                    )
                    pad = bytes_per_frame - len(exc.partial)
                    raw = exc.partial + (b"\x00" * pad)
                else:
                    break

            # numpy view over the bytes; copy because the caller may keep
            # the array beyond the lifetime of this read buffer.
            pcm = np.frombuffer(raw, dtype=np.float32).copy()

            frame: PcmFrame = {
                "t_abs_s": t_abs_s,
                "pcm": pcm,
                "voiced": None,
            }
            yield frame
            t_abs_s += seconds_per_frame

            # If readexactly returned a padded final frame because of an
            # IncompleteReadError, we're done.
            if len(raw) > 0 and len(raw) == bytes_per_frame and (
                # Slightly defensive: stop if proc has already exited and
                # there's no more buffered output to drain.
                proc.returncode is not None and proc.stdout.at_eof()
            ):
                break
    finally:
        # Drain stderr for diagnostics and reap the child to avoid
        # zombies. We don't block on the child if it's still running —
        # the iterator is closing, so we kill it.
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        err = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
        if err:
            _log.warning("ffmpeg stderr: %s", err)
