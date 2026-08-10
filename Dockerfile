# syntax=docker/dockerfile:1.6
#
# audio-helper — reproducible container image.
#
# Two-stage build: the base stage pulls system deps (ffmpeg is
# mandatory for the whole toolkit) and installs the package with the
# [api] extra so the container can serve the HTTP surface out of the
# box. The optional [demucs] extra is *not* installed by default — it
# drags in ~2 GB of torch + torchaudio + a downloaded model. Enable it
# by building with `--build-arg WITH_DEMUCS=1`.
#
# Build:
#   docker build -t audio-helper .
#   docker build --build-arg WITH_DEMUCS=1 -t audio-helper:demucs .
#
# Run (HTTP on 0.0.0.0:8000):
#   docker run --rm -p 8000:8000 audio-helper
#
# Run CLI one-shot:
#   docker run --rm -v $PWD:/data audio-helper \
#     audio-helper convert --input /data/in.mp3 --output /data/out.wav

# --- base -------------------------------------------------------------------
FROM python:3.11-slim AS base

# System deps: ffmpeg for every audio pipeline, libsndfile for soundfile,
# tini for signal handling. No compilers — we install from wheels only.
RUN apt-get update && apt-get install --no-install-recommends -y \
        ffmpeg \
        libsndfile1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user; the app never needs root at runtime.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# --- deps -------------------------------------------------------------------
# requirements.txt first (core deps only) so this layer caches independently
# of source changes; the package itself (with its extras) is installed once
# the source is in place, right below.
COPY --chown=app:app requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app audio_helper ./audio_helper

# Build-arg switch: install demucs extra if requested. Default = light image.
ARG WITH_DEMUCS=0
RUN if [ "$WITH_DEMUCS" = "1" ] ; then \
        pip install --no-cache-dir '.[api,demucs]' ; \
    else \
        pip install --no-cache-dir '.[api]' ; \
    fi

# --- runtime ----------------------------------------------------------------
USER app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    AUDIO_HELPER_HOST=0.0.0.0 \
    AUDIO_HELPER_PORT=8000

# tini reaps orphan children (ffmpeg subprocesses) cleanly on SIGTERM.
ENTRYPOINT ["/usr/bin/tini", "--"]
# Default: serve FastAPI. Override for one-shot CLI usage.
CMD ["uvicorn", "audio_helper.api:app", "--host", "0.0.0.0", "--port", "8000"]
