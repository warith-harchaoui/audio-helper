"""
Smoke tests for the FastAPI HTTP surface.

Only exercises endpoints that do not require ffmpeg or the network
(``/health``, plus OpenAPI schema introspection to catch endpoint-name
drift). Heavier round-trip tests belong to the ``integration`` suite
where a real ffmpeg is available.

Usage Example
-------------
>>> #   pytest tests/test_api.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

# FastAPI is in the ``[api]`` optional extra — skip cleanly otherwise.
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Yield a TestClient bound to the audio-helper FastAPI app."""
    from audio_helper.api import app

    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    """``/health`` should return 200 + ``{"status": "ok"}``."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_lists_expected_endpoints(client):
    """The OpenAPI spec should list every expected route path."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    expected = {
        "/health",
        "/convert",
        "/duration",
        "/chunk",
        "/silence",
        "/concat",
        "/roomtone",
        "/split",
        "/separate",
        "/resemblance",
    }
    assert expected.issubset(set(paths.keys()))


def test_docs_endpoint_is_served(client):
    """``/docs`` should serve the Swagger UI landing HTML."""
    r = client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


def test_gui_returns_200_html(client):
    """``GET /gui`` should return 200 with a self-contained HTML page."""
    r = client.get("/gui")
    assert r.status_code == 200
    # It must be an HTML document (correct content type + a doctype/tag).
    assert r.headers["content-type"].startswith("text/html")
    body = r.text.lower()
    assert "<!doctype html>" in body
    # Sanity-check it is the Recipe Canvas and offers the real operations
    # (the JS builds endpoint URLs from OPS, so we assert on the op names).
    assert "recipe canvas" in body
    assert 'value="convert"' in r.text and 'value="separate"' in r.text


def test_root_redirects_to_gui(client):
    """``GET /`` should redirect (or resolve) to the GUI page."""
    # follow_redirects defaults True in the TestClient; assert we land on HTML.
    r = client.get("/")
    assert r.status_code == 200
    assert "recipe canvas" in r.text.lower()


def test_gui_ships_recipe_canvas_features(client):
    """The GUI should ship the Recipe-Canvas surfaces, not just the old bench.

    We assert on stable, load-bearing markers in the served HTML so a future
    edit that silently guts a feature (the runner, the comparator, the YAML
    round-trip, or the CDN libraries) fails CI instead of shipping broken.
    """
    body = client.get("/gui").text
    lower = body.lower()

    # Sequential recipe runner: an ordered step list plus a "Run recipe" action.
    assert 'id="recipe"' in body
    assert "run-recipe" in body

    # Per-step bypass toggle (instant A/B) and reorder controls exist.
    assert "bypass" in lower

    # Ear-first comparator: before/after selectors + Space-bar A/B toggle.
    assert 'id="cmp-a"' in body and 'id="cmp-b"' in body
    assert "cmp-toggle" in body

    # recipe.yaml export/import so a pipeline is a committable artifact.
    assert "export recipe.yaml" in lower
    assert "import recipe.yaml" in lower

    # CDN libraries are wired via <script> tags (no build step, house style).
    assert "wavesurfer.js" in lower  # waveforms
    assert "vega-lite" in lower  # spectrograms
    assert "fflate" in lower  # in-browser unzip for split/separate chaining

    # Local-first honesty is surfaced in the header copy.
    assert "local-first" in lower

    # All eight verbs remain reachable from the "add step" selector.
    for verb in (
        "convert",
        "chunk",
        "silence",
        "concat",
        "roomtone",
        "split",
        "separate",
        "resemblance",
    ):
        assert f'value="{verb}"' in body
