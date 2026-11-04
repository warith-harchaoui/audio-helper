"""One guard: every ``audio-helper`` command shown in the docs must parse.

The skill's reference was rewritten from ``--help`` in 2.4, yet the quick
examples of ``SKILL.md`` itself kept the positional form
(``audio-helper convert talk.mp3 talk.wav --sample-rate 16000``) until
2026-11-04: an agent reading the skill's front page got a command that fails.
Nothing compared the prose to the parser, so this test does.

It reads each fenced shell line that starts with ``audio-helper <sub>`` in the
documents a user or an agent copies from, and checks it against the click
definition of that subcommand: the subcommand exists, every ``--flag`` is one
of its options, and a bare word appears only where the subcommand takes a
positional argument. Paths are not checked for existence: the files in an
example are illustrations.

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import click
import pytest

from audio_helper.cli import cli

ROOT = Path(__file__).resolve().parent.parent
DOCS = [
    "README.md",
    "LISEZMOI.md",
    "EXAMPLES.md",
    "EXEMPLES.md",
    "skills/audio-helper/SKILL.md",
]


def _doc_commands() -> list[tuple[str, str]]:
    """Every ``audio-helper <sub> ...`` line inside a fenced block."""
    found = []
    for name in DOCS:
        path = ROOT / name
        if not path.exists():
            continue
        in_fence, pending = False, ""
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.lstrip().startswith("```"):
                in_fence, pending = not in_fence, ""
                continue
            if not in_fence:
                continue
            line = pending + raw.strip()
            if line.endswith("\\"):
                pending = line[:-1] + " "
                continue
            pending = ""
            line = re.sub(r"^\$\s*", "", line).split(" #")[0].strip()
            if re.match(r"audio-helper [a-z]", line):
                found.append((name, line))
    return found


COMMANDS = _doc_commands()


def test_the_docs_show_commands_at_all() -> None:
    # If the extraction silently found nothing, every case below would pass.
    assert len(COMMANDS) >= 10


@pytest.mark.parametrize(("doc", "line"), COMMANDS, ids=[f"{d}:{c[:50]}" for d, c in COMMANDS])
def test_documented_command_parses(doc: str, line: str) -> None:
    tokens = shlex.split(line)[1:]
    sub_name, args = tokens[0], tokens[1:]
    sub = cli.commands.get(sub_name)
    assert sub is not None, f"{doc}: no subcommand {sub_name!r} in `{line}`"

    options: dict[str, click.Option] = {}
    for param in sub.params:
        if isinstance(param, click.Option):
            for opt in [*param.opts, *param.secondary_opts]:
                options[opt] = param
    takes_positionals = any(isinstance(p, click.Argument) for p in sub.params)

    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-"):
            flag = tok.split("=", 1)[0]
            assert flag in options, f"{doc}: `{sub_name}` has no option {flag!r} in `{line}`"
            param = options[flag]
            if not param.is_flag and "=" not in tok:
                i += 1  # the option's value
        else:
            assert takes_positionals, (
                f"{doc}: `{sub_name}` takes no positional argument, got {tok!r} in `{line}`"
            )
        i += 1
