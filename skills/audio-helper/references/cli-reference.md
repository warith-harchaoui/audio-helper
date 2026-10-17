# audio-helper CLI reference

Two CLIs over the same nine operations. `audio-helper` (argparse) ships with
the core package; `audio-helper-click` (needs `[cli]`) is the click twin with
grouped help. Flags match between them.

## `convert` — re-encode

```bash
audio-helper convert IN OUT [--sample-rate HZ] [--channels N] [--bitrate K]
```
Format is inferred from `OUT`'s extension. `--channels 1` downmixes to mono.
The common speech preset is `--sample-rate 16000 --channels 1`.
Accepts a video file as `IN`: the audio track is decoded.

## `duration` — measure

```bash
audio-helper duration IN
```
Prints the duration in seconds as a float.

## `chunk` — extract one `[start, end]` slice

```bash
audio-helper chunk IN OUT --start SECONDS --end SECONDS
```
Both bounds are seconds from the start of the file. Out-of-range bounds are
clamped, and an empty range is an error, not a silent empty file.

## `split` — fixed-length chunks

```bash
audio-helper split IN OUTDIR --duration SECONDS
```
Writes `OUTDIR/<stem>_000.<ext>`, `_001`, … and prints every path produced.
Use it to feed a batch pipeline that has a per-request length limit.

## `concat` — join files

```bash
audio-helper concat OUT IN1 IN2 [IN3 …]
```
Inputs are concatenated in the order given. Mismatched sample rates and channel
counts are reconciled by ffmpeg before the join.

## `silence` — generate a blank spacer

```bash
audio-helper silence OUT --duration SECONDS [--sample-rate HZ] [--channels N]
```

## `roomtone` — mask abrupt edits

```bash
audio-helper roomtone IN OUT [--level DB] [--color pink|brown|white]
```
Mixes a low-level ambient bed under the whole track so cuts between takes stop
sounding like hard splices. This is a *masking* tool, not a denoiser.

## `separate` — stems (needs `[demucs]`)

```bash
audio-helper separate IN OUTDIR [--stems vocals,drums,bass,other]
```
Runs a local Demucs and writes one file per requested stem. First run downloads
the model weights once; every run after that is offline. Without the `[demucs]`
extra the command fails fast with an install hint.

## `resemblance` — MFCC similarity

```bash
audio-helper resemblance A B
```
Prints a similarity score. Higher means more alike. Use it to spot duplicate
takes or verify that a transform preserved the content; do not use it for
speaker identification.

## Output contract

- Every command prints the path(s) it wrote, one per line, so the output can be
  piped.
- Measurements (`duration`, `resemblance`) print a bare number.
- Failures are logged through `os_helper` and exit non-zero; partial output
  files are removed rather than left behind.
