"""
Smoke tests for the argparse and click CLIs.

These tests exercise the CLI *parsing* layer and the trivial subcommands
that do not need ffmpeg (``duration``-like routes are covered by the
integration suite). The goal here is to prevent regressions in the
CLI entry points — flag names, subcommand names, dispatch wiring —
without pulling in the full ffmpeg stack.

Usage Example
-------------
>>> #   pytest tests/test_cli.py

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import pytest

# The click CLI needs the ``click`` runtime dep, which lives in the
# ``[cli]`` optional extra. Skip cleanly if it is not installed.
click = pytest.importorskip("click")

from click.testing import CliRunner  # noqa: E402


def test_argparse_parser_builds_without_error():
    """Building the parser should never fail (imports, subcommand wiring)."""
    from audio_helper.cli_argparse import build_parser

    parser = build_parser()
    # A parser with at least one subcommand exposes them via _subparsers.
    # We assert on the expected list of subcommand names to catch drift.
    subparsers_action = next(
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    )
    expected = {
        "convert",
        "duration",
        "chunk",
        "silence",
        "concat",
        "roomtone",
        "split",
        "separate",
        "resemblance",
    }
    assert expected.issubset(set(subparsers_action.choices.keys()))


def test_argparse_help_exits_zero(capsys):
    """``audio-helper --help`` should exit with code 0 and print usage."""
    from audio_helper.cli_argparse import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "audio-helper" in captured.out.lower()


@pytest.mark.parametrize(
    "sub",
    [
        "convert",
        "duration",
        "chunk",
        "silence",
        "concat",
        "roomtone",
        "split",
        "separate",
        "resemblance",
    ],
)
def test_argparse_subcommand_help_exits_zero(sub, capsys):
    """Every subcommand's ``--help`` should exit 0 (no wiring bug)."""
    from audio_helper.cli_argparse import main

    with pytest.raises(SystemExit) as exc:
        main([sub, "--help"])
    assert exc.value.code == 0


def test_click_group_has_expected_subcommands():
    """The click group must expose the same subcommands as the argparse CLI."""
    from audio_helper.cli_click import cli

    expected = {
        "convert",
        "duration",
        "chunk",
        "silence",
        "concat",
        "roomtone",
        "split",
        "separate",
        "resemblance",
    }
    assert expected.issubset(set(cli.commands.keys()))


def test_click_help_exits_zero():
    """``audio-helper-click --help`` should exit 0."""
    from audio_helper.cli_click import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "audio helper" in result.output.lower()


@pytest.mark.parametrize(
    "sub",
    [
        "convert",
        "duration",
        "chunk",
        "silence",
        "concat",
        "roomtone",
        "split",
        "separate",
        "resemblance",
    ],
)
def test_click_subcommand_help_exits_zero(sub):
    """Every click subcommand's ``--help`` should exit 0."""
    from audio_helper.cli_click import cli

    runner = CliRunner()
    result = runner.invoke(cli, [sub, "--help"])
    assert result.exit_code == 0
