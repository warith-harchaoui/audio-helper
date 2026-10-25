"""
Smoke tests for the ``audio-helper`` command line.

These tests exercise the CLI *parsing* layer and the wiring between
subcommands and the library (the operations themselves, which need ffmpeg,
are covered by the integration suite). The goal is to prevent regressions in
the entry point — subcommand names, flag names, flag arity, error
translation — without pulling in the full ffmpeg stack.

Usage Example
-------------
>>> #   pytest tests/test_cli.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import sys

import pytest
from click.testing import CliRunner

from audio_helper.cli import cli, main

# The nine operations the CLI exposes. Asserted as a set in several places so
# a renamed or dropped subcommand fails here rather than in somebody's shell.
SUBCOMMANDS = [
    "convert",
    "duration",
    "chunk",
    "silence",
    "concat",
    "roomtone",
    "split",
    "separate",
    "resemblance",
]


def test_group_exposes_every_subcommand():
    """The group must expose all nine operations, under these exact names."""
    assert set(SUBCOMMANDS).issubset(set(cli.commands.keys()))


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_exits_zero(flag):
    """Both spellings of the help flag exit 0 and print the tool's purpose.

    ``-h`` is not free: click only wires it when the group declares
    ``help_option_names``, and a CLI that answers ``--help`` but not ``-h``
    is a papercut nobody notices until a user hits it.
    """
    result = CliRunner().invoke(cli, [flag])
    assert result.exit_code == 0
    assert "audio helper" in result.output.lower()


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_exits_zero(flag):
    """Both spellings of the version flag exit 0 and name the package.

    ``-V`` rather than ``-v``: the short lowercase form is ``--verbose``
    across the suite, so the version flag is the capital one.
    """
    result = CliRunner().invoke(cli, [flag])
    assert result.exit_code == 0
    assert "audio-helper" in result.output


@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_subcommand_help_exits_zero(sub):
    """Every subcommand's ``--help`` exits 0 — i.e. its wiring is intact."""
    result = CliRunner().invoke(cli, [sub, "--help"])
    assert result.exit_code == 0


def test_concat_accepts_several_files_after_one_inputs_flag(monkeypatch, tmp_path):
    """``concat --inputs a b c`` must pass all three files, in order.

    A click ``multiple=True`` option binds exactly one value per occurrence,
    so this spelling — the one the docs have always shown — would otherwise
    keep only ``a`` and reject ``b`` as an unexpected argument. The trailing
    files are collected as variadic arguments and appended, which is what
    this asserts.
    """
    files = []
    for name in ("a.wav", "b.wav", "c.wav"):
        path = tmp_path / name
        path.write_bytes(b"")
        files.append(str(path))

    seen: list[list[str]] = []

    def fake_concat(audio_files, output_audio_filename, overwrite):
        seen.append(list(audio_files))
        return output_audio_filename

    monkeypatch.setattr("audio_helper.cli.audio_concatenation", fake_concat)

    out = str(tmp_path / "all.wav")
    result = CliRunner().invoke(cli, ["concat", "--inputs", *files, "--output", out])
    assert result.exit_code == 0, result.output
    assert seen == [files]


def test_concat_accepts_a_repeated_inputs_flag(monkeypatch, tmp_path):
    """``concat --inputs a --inputs b`` — the click-idiomatic spelling — also
    works, and yields the same ordered list as the variadic form above."""
    files = []
    for name in ("a.wav", "b.wav"):
        path = tmp_path / name
        path.write_bytes(b"")
        files.append(str(path))

    seen: list[list[str]] = []

    def fake_concat(audio_files, output_audio_filename, overwrite):
        seen.append(list(audio_files))
        return output_audio_filename

    monkeypatch.setattr("audio_helper.cli.audio_concatenation", fake_concat)

    argv = ["concat"]
    for path in files:
        argv += ["--inputs", path]
    argv += ["--output", str(tmp_path / "all.wav")]
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output
    assert seen == [files]


def test_missing_input_path_is_a_usage_error():
    """A path that does not exist is rejected by the parser, exit code 2.

    ``--input`` is a ``click.Path(exists=True)``, so a typo never reaches the
    library: it comes back as a usage error naming the offending value.
    """
    result = CliRunner().invoke(cli, ["duration", "--input", "/nonexistent/missing.mp3"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_main_prints_one_clean_error_line_on_a_library_exception(monkeypatch, tmp_path, capsys):
    """A library exception must surface as one ``Error: ...`` line + exit 1.

    Click's own handling covers ``ClickException``/``Abort`` only; anything a
    library function raises (an ``AssertionError`` from a validation helper,
    an ``ffmpeg.Error`` from a corrupt file) would otherwise reach the user as
    a raw traceback. :func:`audio_helper.cli.main` is the wrapper that
    translates it, so this calls that wrapper rather than the group.

    The exception is injected rather than provoked with a real bad file: the
    guarantee under test belongs to the wrapper, and provoking it for real
    needs ffmpeg on PATH (see the integration test below).
    """
    existing = tmp_path / "input.mp3"
    existing.write_bytes(b"")

    def boom(*_args, **_kwargs):
        raise AssertionError("not a valid audio file")

    monkeypatch.setattr("audio_helper.cli.get_audio_duration", boom)
    monkeypatch.setattr(sys, "argv", ["audio-helper", "duration", "--input", str(existing)])

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error: not a valid audio file")


@pytest.mark.integration
def test_main_prints_clean_error_on_a_real_unreadable_file(tmp_path, capsys):
    """Same guarantee as above, provoked for real instead of injected.

    An existing-but-not-audio file reaches ``get_audio_duration``'s
    ``ffmpeg.probe()``, which fails — so this one needs a real ffmpeg on
    PATH, hence ``integration``.
    """
    garbage = tmp_path / "not_audio.mp3"
    garbage.write_text("this is not an audio file")

    old_argv = sys.argv
    sys.argv = ["audio-helper", "duration", "--input", str(garbage)]
    try:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    assert captured.out.startswith("Error:") or captured.err.startswith("Error:")
