"""
Audio Helper — click-based command-line interface.

Twin of :mod:`audio_helper.cli_argparse`: same public surface (identical
subcommand names, identical flag semantics), but implemented with
:mod:`click` so users who already have a click-native shell setup
(bash / zsh completion via ``click.shell_completion``, colored `--help`,
nested command groups) can plug it in without friction. Installed as
the ``audio-helper-click`` entry point in ``pyproject.toml``.

Design notes
------------
- Subcommands mirror ``audio-helper`` (the argparse twin) so both CLIs
  can be introspected identically by higher layers (FastAPI, MCP).
- Flags reuse the argparse names (``--input`` / ``--output`` / …) rather
  than the more idiomatic click positional style — consistency across
  the two CLIs beats micro-idiomaticity here.
- Errors from the library propagate unchanged; click handles the
  formatting.

Usage Example
-------------
>>> #   audio-helper-click convert    --input in.mp3 --output out.wav --freq 44100
>>> #   audio-helper-click duration   --input in.mp3
>>> #   audio-helper-click split      --input in.mp3 --output-dir chunks/ --seconds 30
>>> #   audio-helper-click resemblance --a a.mp3 --b b.mp3

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import json
import sys

try:
    import click
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The click CLI requires the [cli] extra. "
        "Install with: pip install 'audio-helper[cli]'"
    ) from exc

# Same underlying functions as the argparse twin — one source of truth.
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
# Top-level group
#
# ``invoke_without_command=False`` forces the user to name a subcommand;
# ``context_settings`` widens the help output so long option lists stay
# readable on modern terminals.
# ---------------------------------------------------------------------------


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.version_option(package_name="audio-helper", prog_name="audio-helper-click")
def cli() -> None:
    """Audio Helper — click twin of the argparse CLI. Same subcommands."""
    # Nothing to do at the group level — every subcommand carries its
    # own arguments and side effects.


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input", "input_", required=True, type=click.Path(exists=True), help="Input audio path.")
@click.option("--output", required=True, type=click.Path(), help="Output audio path.")
@click.option("--freq", type=int, default=44100, show_default=True, help="Target sample rate in Hz.")
@click.option("--channels", type=int, default=1, show_default=True, help="Channel count (1 = mono).")
@click.option("--encoding", default="pcm_s16le", show_default=True, help="ffmpeg codec name.")
@click.option("--overwrite/--no-overwrite", default=True, show_default=True, help="Overwrite existing output.")
def convert(input_: str, output: str, freq: int, channels: int, encoding: str, overwrite: bool) -> None:
    """Re-encode an audio file (freq / channels / codec)."""
    # Thin dispatch to the library.
    out = sound_converter(
        input_audio=input_,
        output_audio=output,
        freq=freq,
        channels=channels,
        encoding=encoding,
        overwrite=overwrite,
    )
    click.echo(out)


# ---------------------------------------------------------------------------
# duration
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input", "input_", required=True, type=click.Path(exists=True), help="Audio path.")
def duration(input_: str) -> None:
    """Print the duration of an audio file, in seconds."""
    click.echo(f"{get_audio_duration(input_):.6f}")


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input", "input_", required=True, type=click.Path(exists=True))
@click.option("--start", type=float, required=True, help="Start time in seconds.")
@click.option("--end", type=float, required=True, help="End time in seconds.")
@click.option("--output", type=click.Path(), default=None, help="Output path (auto if omitted).")
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
def chunk(input_: str, start: float, end: float, output: str | None, overwrite: bool) -> None:
    """Extract a ``[start, end]`` slice from an audio file."""
    out = extract_audio_chunk(
        audio_file=input_,
        start_time=start,
        end_time=end,
        output_audio_filename=output,
        overwrite=overwrite,
    )
    click.echo(out)


# ---------------------------------------------------------------------------
# silence
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--duration", "duration_", type=float, required=True, help="Duration in seconds.")
@click.option("--output", type=click.Path(), default=None, help="Output path (auto if omitted).")
@click.option("--sample-rate", type=int, default=44100, show_default=True)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def silence(duration_: float, output: str | None, sample_rate: int, overwrite: bool) -> None:
    """Generate a silent audio file of a given duration."""
    out = generate_silent_audio(
        duration=duration_,
        output_audio_filename=output,
        sample_rate=sample_rate,
        overwrite=overwrite,
    )
    click.echo(out)


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--inputs",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Audio files in order (repeat --inputs for each).",
)
@click.option("--output", type=click.Path(), default=None)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def concat(inputs: tuple[str, ...], output: str | None, overwrite: bool) -> None:
    """Concatenate several audio files head-to-tail."""
    out = audio_concatenation(
        audio_files=list(inputs),
        output_audio_filename=output,
        overwrite=overwrite,
    )
    click.echo(out)


# ---------------------------------------------------------------------------
# roomtone
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input", "input_", required=True, type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None)
@click.option("--db", type=float, default=-42.0, show_default=True, help="Noise level in dB.")
@click.option(
    "--color",
    type=click.Choice(["white", "pink", "brown", "red", "blue", "violet", "velvet"]),
    default="pink",
    show_default=True,
)
@click.option("--sample-rate", type=int, default=44100, show_default=True)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def roomtone(input_: str, output: str | None, db: float, color: str, sample_rate: int, overwrite: bool) -> None:
    """Mix low-level colored ambient noise on top of a speech track."""
    out = mix_room_tone(
        input_audio=input_,
        output_audio=output,
        noise_db=db,
        color=color,
        sample_rate=sample_rate,
        overwrite=overwrite,
    )
    click.echo(out)


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input", "input_", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path(), help="Folder that receives the chunks.")
@click.option("--seconds", type=float, required=True, help="Chunk duration in seconds.")
@click.option("--output-format", default="mp3", show_default=True, help="Chunk extension.")
@click.option("--suffix", default="split", show_default=True, help="Filename suffix.")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def split(input_: str, output_dir: str, seconds: float, output_format: str, suffix: str, overwrite: bool) -> None:
    """Split an audio file into fixed-duration chunks."""
    outputs = split_audio_regularly(
        sound_path=input_,
        chunk_folder=output_dir,
        split_time=seconds,
        output_format=output_format,
        overwrite=overwrite,
        suffix=suffix,
    )
    for path in outputs:
        click.echo(path)


# ---------------------------------------------------------------------------
# separate  (Demucs)
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input", "input_", required=True, type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path(), default=None)
@click.option("--device", default=None, help="'cuda' / 'cpu' / None (auto).")
@click.option("--workers", type=int, default=-2, show_default=True, help="Worker threads (sklearn convention).")
@click.option("--output-format", default="mp3", show_default=True)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True)
def separate(input_: str, output_dir: str | None, device: str | None, workers: int, output_format: str, overwrite: bool) -> None:
    """Run Demucs source separation (needs the ``[demucs]`` extra)."""
    try:
        result = separate_sources(
            input_audio_file=input_,
            output_folder=output_dir,
            device=device,
            overwrite=overwrite,
            nb_workers=workers,
            output_format=output_format,
        )
    except ImportError as exc:
        # Surface a clean error rather than a raw traceback in the terminal.
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)
    click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# resemblance
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--a", required=True, type=click.Path(exists=True), help="First audio path.")
@click.option("--b", required=True, type=click.Path(exists=True), help="Second audio path.")
def resemblance(a: str, b: str) -> None:
    """MFCC-based similarity score between two audio files."""
    click.echo(f"{sound_resemblance(a, b):.6f}")


if __name__ == "__main__":  # pragma: no cover
    cli()
