# audio-helper CLI reference

One command line over the nine operations: `audio-helper`, built with click and
shipped with the core package. `-h` / `--help` works on the group and on every
subcommand; `-V` prints the version.

Every flag below is copied from the command's own `--help`. The CLI takes no
positional arguments (except `concat`'s extra files): paths are always named,
because most of these commands take several of them and a bare positional order
is easy to get backwards.

## `convert` — re-encode

```bash
audio-helper convert --input IN --output OUT [--freq HZ] [--channels N] \
    [--encoding CODEC] [--no-overwrite]
```
Format is inferred from `--output`'s extension. `--channels 1` downmixes to
mono. Defaults: `--freq 44100`, `--channels 1`, `--encoding pcm_s16le`,
overwrite ON. The common speech preset is `--freq 16000 --channels 1`.
Accepts a video file as `--input`: the audio track is decoded.

## `duration` — measure

```bash
audio-helper duration --input IN
```
Prints the duration in seconds as a float, six decimals.

## `chunk` — extract one `[start, end]` slice

```bash
audio-helper chunk --input IN --start SECONDS --end SECONDS [--output OUT] \
    [--no-overwrite]
```
Both bounds are seconds from the start of the file. `--output` is derived from
the input when omitted. Overwrite is ON by default here.

## `split` — fixed-length chunks

```bash
audio-helper split --input IN --output-dir OUTDIR --seconds SECONDS \
    [--output-format EXT] [--suffix SUFFIX] [--overwrite]
```
Writes `OUTDIR/chunk_0000_<suffix>.<ext>`, `_0001`, … and prints every path
produced, one per line. Defaults: `--output-format mp3`, `--suffix split`.
Use it to feed a batch pipeline that has a per-request length limit.

## `concat` — join files

```bash
audio-helper concat --inputs IN1 IN2 [IN3 …] [--output OUT] [--overwrite]
audio-helper concat --inputs IN1 --inputs IN2 [--output OUT]   # same thing
```
Inputs are concatenated in the order given; both spellings of `--inputs` are
accepted. Mismatched sample rates and channel counts are reconciled by ffmpeg
before the join.

## `silence` — generate a blank spacer

```bash
audio-helper silence --duration SECONDS [--output OUT] [--sample-rate HZ] \
    [--overwrite]
```

## `roomtone` — mask abrupt edits

```bash
audio-helper roomtone --input IN [--output OUT] [--db DB] \
    [--color white|pink|brown|red|blue|violet|velvet] [--sample-rate HZ] \
    [--overwrite]
```
Mixes a low-level ambient bed under the whole track so cuts between takes stop
sounding like hard splices. Defaults: `--db -42`, `--color pink`. This is a
*masking* tool, not a denoiser.

## `separate` — stems (needs `[demucs]`)

```bash
audio-helper separate --input IN [--output-dir OUTDIR] [--device cuda|cpu] \
    [--workers N] [--output-format EXT] [--overwrite]
```
Runs a local Demucs and writes one file per stem (vocals / drums / bass /
other), printing the stem-to-path map as JSON. Stems are not selectable: Demucs
produces all four. First run downloads the model weights once; every run after
that is offline. Without the `[demucs]` extra the command fails fast with an
install hint and exit code 2.

## `resemblance` — MFCC similarity

```bash
audio-helper resemblance --a A --b B
```
Prints a similarity score, six decimals. Higher means more alike. Use it to
spot duplicate takes or verify that a transform preserved the content; do not
use it for speaker identification.

## Output contract

- Every command prints the path(s) it wrote, one per line, so the output can be
  piped. `separate` prints JSON instead, one key per stem.
- Measurements (`duration`, `resemblance`) print a bare number.
- A path that does not exist is caught by the parser: exit code 2 with a usage
  line naming the bad value.
- A library failure (an unreadable file, a failing ffmpeg) prints one
  `Error: …` line on stderr and exits 1; partial output files are removed
  rather than left behind.
