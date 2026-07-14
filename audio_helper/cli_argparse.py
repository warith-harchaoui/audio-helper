"""
Audio Helper — argparse-based command-line interface.

Thin wrapper around the pure functions in :mod:`audio_helper.main` that
exposes the whole toolkit as subcommands under a single ``audio-helper``
entry point. Written with :mod:`argparse` from the standard library so
the CLI works out of the box on any Python install that has the package
installed — no extra dependency required.

Subcommands
-----------
- ``convert``     — re-encode an audio file (sample rate / channels / codec)
- ``duration``    — print the duration in seconds of an audio file
- ``chunk``       — extract ``[start, end]`` slice from an audio file
- ``silence``     — write a silent audio file of a given duration
- ``concat``      — concatenate several audio files head-to-tail
- ``roomtone``    — mix pink/white/brown ambient noise onto a speech track
- ``split``       — split an audio file into fixed-duration chunks
- ``separate``    — run Demucs source separation (needs the ``[demucs]`` extra)
- ``resemblance`` — MFCC-based similarity score between two audio files

Usage Example
-------------
>>> #   audio-helper convert    --input in.mp3 --output out.wav --freq 44100
>>> #   audio-helper duration   --input in.mp3
>>> #   audio-helper split      --input in.mp3 --output-dir chunks/ --seconds 30
>>> #   audio-helper concat     --inputs a.mp3 b.mp3 c.mp3 --output out.mp3
>>> #   audio-helper silence    --duration 5 --output silence.wav
>>> #   audio-helper roomtone   --input speech.wav --output speech-rt.wav --db -42
>>> #   audio-helper chunk      --input in.mp3 --start 3.0 --end 8.5 --output cut.mp3
>>> #   audio-helper separate   --input mix.mp3 --output-dir stems/
>>> #   audio-helper resemblance --a a.mp3 --b b.mp3

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

# Import the pure functions once here — every subcommand is a thin dispatch
# on top of these, no logic duplication.
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
# Subcommand handlers
#
# Each handler receives the parsed ``argparse.Namespace`` and returns a
# process exit code (``0`` on success). Handlers deliberately stay short:
# they translate CLI arguments into keyword arguments for the underlying
# library function, print a machine-friendly result (JSON for structured
# outputs, plain path for single-file outputs), and let exceptions
# propagate as non-zero exit codes.
# ---------------------------------------------------------------------------


def _handle_convert(ns: argparse.Namespace) -> int:
    # sound_converter re-encodes an input file to a target sample rate /
    # channel layout / codec. Returns the output path on success.
    out = sound_converter(
        input_audio=ns.input,
        output_audio=ns.output,
        freq=ns.freq,
        channels=ns.channels,
        encoding=ns.encoding,
        overwrite=ns.overwrite,
    )
    # Emit the resulting path so shell pipelines can chain on stdout.
    print(out)
    return 0


def _handle_duration(ns: argparse.Namespace) -> int:
    # get_audio_duration returns a float in seconds — print it verbatim.
    duration = get_audio_duration(ns.input)
    print(f"{duration:.6f}")
    return 0


def _handle_chunk(ns: argparse.Namespace) -> int:
    # extract_audio_chunk cuts a [start, end] slice into a new file.
    out = extract_audio_chunk(
        audio_file=ns.input,
        start_time=ns.start,
        end_time=ns.end,
        output_audio_filename=ns.output,
        overwrite=ns.overwrite,
    )
    print(out)
    return 0


def _handle_silence(ns: argparse.Namespace) -> int:
    # generate_silent_audio writes a silent file of the given duration.
    out = generate_silent_audio(
        duration=ns.duration,
        output_audio_filename=ns.output,
        sample_rate=ns.sample_rate,
        overwrite=ns.overwrite,
    )
    print(out)
    return 0


def _handle_concat(ns: argparse.Namespace) -> int:
    # audio_concatenation joins N files into one using ffmpeg's concat filter.
    out = audio_concatenation(
        audio_files=list(ns.inputs),
        output_audio_filename=ns.output,
        overwrite=ns.overwrite,
    )
    print(out)
    return 0


def _handle_roomtone(ns: argparse.Namespace) -> int:
    # mix_room_tone overlays a low-level colored-noise bed on a speech track
    # to mask cuts. -42 dB pink noise is a typical post-production default.
    out = mix_room_tone(
        input_audio=ns.input,
        output_audio=ns.output,
        noise_db=ns.db,
        color=ns.color,
        sample_rate=ns.sample_rate,
        overwrite=ns.overwrite,
    )
    print(out)
    return 0


def _handle_split(ns: argparse.Namespace) -> int:
    # split_audio_regularly chops an audio file into fixed-duration chunks.
    # Returns a list of output paths — emit them one per line so shell
    # pipelines can `xargs -n 1` over the result.
    outputs = split_audio_regularly(
        sound_path=ns.input,
        chunk_folder=ns.output_dir,
        split_time=ns.seconds,
        output_format=ns.output_format,
        overwrite=ns.overwrite,
        suffix=ns.suffix,
    )
    for path in outputs:
        print(path)
    return 0


def _handle_separate(ns: argparse.Namespace) -> int:
    # separate_sources runs Demucs to split a mix into vocals/drums/bass/other.
    # Requires the optional [demucs] extra — if torch is missing the
    # underlying function raises ImportError, which we surface with a
    # non-zero exit code and a hint on stderr.
    try:
        result = separate_sources(
            input_audio_file=ns.input,
            output_folder=ns.output_dir,
            device=ns.device,
            overwrite=ns.overwrite,
            nb_workers=ns.workers,
            output_format=ns.output_format,
        )
    except ImportError as exc:
        # Guarantee an actionable error rather than a cryptic traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Structured output: {"vocals": path, "drums": path, ...}
    print(json.dumps(result, indent=2))
    return 0


def _handle_resemblance(ns: argparse.Namespace) -> int:
    # sound_resemblance returns a scalar in [0, 1] (MFCC cosine similarity).
    score = sound_resemblance(ns.a, ns.b)
    print(f"{score:.6f}")
    return 0


# ---------------------------------------------------------------------------
# Parser construction
#
# One helper per subcommand keeps ``build_parser`` readable and lets the
# click twin (:mod:`audio_helper.cli_click`) mirror the exact same flag
# names without any risk of drift.
# ---------------------------------------------------------------------------


def _add_convert(sub: argparse._SubParsersAction) -> None:
    # Convert / re-encode via ffmpeg.
    p = sub.add_parser("convert", help="Re-encode an audio file (freq / channels / codec).")
    p.add_argument("--input", required=True, help="Input audio path.")
    p.add_argument(
        "--output", required=True, help="Output audio path (extension picks the container)."
    )
    p.add_argument(
        "--freq", type=int, default=44100, help="Target sample rate in Hz (default 44100)."
    )
    p.add_argument(
        "--channels", type=int, default=1, help="Target channel count (default 1 = mono)."
    )
    p.add_argument("--encoding", default="pcm_s16le", help="ffmpeg codec name (default pcm_s16le).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output if it exists (default true).",
    )
    p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip if output already valid.",
    )
    p.set_defaults(func=_handle_convert)


def _add_duration(sub: argparse._SubParsersAction) -> None:
    # Print duration in seconds. Handy in shell for `--to` bounds.
    p = sub.add_parser("duration", help="Print the duration of an audio file, in seconds.")
    p.add_argument("--input", required=True, help="Audio path.")
    p.set_defaults(func=_handle_duration)


def _add_chunk(sub: argparse._SubParsersAction) -> None:
    # Extract [start, end] slice.
    p = sub.add_parser("chunk", help="Extract a [start, end] slice from an audio file.")
    p.add_argument("--input", required=True, help="Source audio path.")
    p.add_argument("--start", type=float, required=True, help="Start time in seconds.")
    p.add_argument("--end", type=float, required=True, help="End time in seconds.")
    p.add_argument("--output", default=None, help="Output path (auto if omitted).")
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    p.set_defaults(func=_handle_chunk)


def _add_silence(sub: argparse._SubParsersAction) -> None:
    # Silent WAV / MP3 / … of N seconds.
    p = sub.add_parser("silence", help="Generate a silent audio file of a given duration.")
    p.add_argument("--duration", type=float, required=True, help="Duration in seconds.")
    p.add_argument("--output", default=None, help="Output path (auto if omitted).")
    p.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        dest="sample_rate",
        help="Sample rate (default 44100).",
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=_handle_silence)


def _add_concat(sub: argparse._SubParsersAction) -> None:
    # Head-to-tail join of N files.
    p = sub.add_parser("concat", help="Concatenate several audio files head-to-tail.")
    p.add_argument("--inputs", nargs="+", required=True, help="Audio files, in order.")
    p.add_argument("--output", default=None, help="Output path (auto if omitted).")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=_handle_concat)


def _add_roomtone(sub: argparse._SubParsersAction) -> None:
    # Room-tone masking — standard post-production trick.
    p = sub.add_parser("roomtone", help="Mix low-level colored noise on top of a speech track.")
    p.add_argument("--input", required=True, help="Speech / narration audio path.")
    p.add_argument("--output", default=None, help="Output path (auto if omitted).")
    p.add_argument("--db", type=float, default=-42.0, help="Noise level in dB (default -42).")
    p.add_argument(
        "--color",
        default="pink",
        choices=["white", "pink", "brown", "red", "blue", "violet", "velvet"],
        help="Noise color (default pink).",
    )
    p.add_argument("--sample-rate", type=int, default=44100, dest="sample_rate")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=_handle_roomtone)


def _add_split(sub: argparse._SubParsersAction) -> None:
    # Fixed-duration chunks.
    p = sub.add_parser("split", help="Split an audio file into fixed-duration chunks.")
    p.add_argument("--input", required=True, help="Source audio path.")
    p.add_argument(
        "--output-dir", required=True, dest="output_dir", help="Folder that receives the chunks."
    )
    p.add_argument("--seconds", type=float, required=True, help="Chunk duration in seconds.")
    p.add_argument(
        "--output-format",
        default="mp3",
        dest="output_format",
        help="Chunk extension (default mp3).",
    )
    p.add_argument(
        "--suffix", default="split", help="Filename suffix for each chunk (default 'split')."
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=_handle_split)


def _add_separate(sub: argparse._SubParsersAction) -> None:
    # Demucs. Optional extra.
    p = sub.add_parser("separate", help="Run Demucs source separation (needs the [demucs] extra).")
    p.add_argument("--input", required=True, help="Mixed audio path.")
    p.add_argument(
        "--output-dir", default=None, dest="output_dir", help="Folder that receives the 4 stems."
    )
    p.add_argument("--device", default=None, help="'cuda' / 'cpu' / None (auto).")
    p.add_argument(
        "--workers",
        type=int,
        default=-2,
        help="Number of worker threads (sklearn convention; default -2).",
    )
    p.add_argument(
        "--output-format", default="mp3", dest="output_format", help="Stem extension (default mp3)."
    )
    p.add_argument("--overwrite", action="store_true", default=False)
    p.set_defaults(func=_handle_separate)


def _add_resemblance(sub: argparse._SubParsersAction) -> None:
    # MFCC cosine similarity between two files.
    p = sub.add_parser("resemblance", help="MFCC-based similarity score between two audio files.")
    p.add_argument("--a", required=True, help="First audio path.")
    p.add_argument("--b", required=True, help="Second audio path.")
    p.set_defaults(func=_handle_resemblance)


def build_parser() -> argparse.ArgumentParser:
    """
    Assemble the top-level ``audio-helper`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Fully wired parser with every subcommand attached.
    """
    parser = argparse.ArgumentParser(
        prog="audio-helper",
        description="Audio Helper — utility CLI for load / convert / split / concat / silence / room-tone / MFCC-similarity / Demucs source separation.",
    )
    # Every non-trivial CLI benefits from `--version` — cheap to add and
    # oncall people always look for it. We resolve it lazily to avoid a
    # circular import if importlib.metadata blows up in some edge case.
    try:
        from importlib.metadata import version as _pkg_version

        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {_pkg_version('audio-helper')}",
        )
    except Exception:  # pragma: no cover — never fatal
        pass

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # Register every subcommand. Order matters for help output only.
    _add_convert(subparsers)
    _add_duration(subparsers)
    _add_chunk(subparsers)
    _add_silence(subparsers)
    _add_concat(subparsers)
    _add_roomtone(subparsers)
    _add_split(subparsers)
    _add_separate(subparsers)
    _add_resemblance(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point invoked by ``audio-helper`` (see ``[project.scripts]``).

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # Every subparser sets ``func`` via ``set_defaults`` — no dispatch table
    # needed, argparse resolved it for us.
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
