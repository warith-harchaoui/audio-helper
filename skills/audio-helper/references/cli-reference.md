# audio-helper CLI reference

Full command surface for the `audio-helper` skill. The argparse CLI
(`audio-helper`) ships with the base package; the click twin
(`audio-helper-click`, `[cli]` extra) mirrors the exact same subcommand and
flag names, so anything below works for both by swapping the program name.

## Subcommands

| Subcommand | Purpose | Notable flags |
|------------|---------|---------------|
| `convert` | Re-encode format / sample rate / channels / codec | `--input --output --freq --channels --encoding --overwrite/--no-overwrite` |
| `duration` | Print duration in seconds | `--input` |
| `chunk` | Extract a `[start, end]` slice | `--input --start --end --output --overwrite/--no-overwrite` |
| `silence` | Generate a silent file | `--duration --output --sample-rate --overwrite` |
| `concat` | Concatenate files head-to-tail | `--inputs A B C --output --overwrite` |
| `roomtone` | Mix colored ambient noise | `--input --output --db --color --sample-rate --overwrite` |
| `split` | Split into fixed-duration chunks | `--input --output-dir --seconds --output-format --suffix --overwrite` |
| `separate` | Demucs stem separation (needs `[demucs]`) | `--input --output-dir --device --workers --output-format --overwrite` |
| `resemblance` | MFCC similarity score in `[0, 1]` | `--a --b` |

`audio-helper --version` and `audio-helper <sub> --help` work for every
subcommand. The click twin is `audio-helper-click <sub> …` with identical flags
(click uses `--overwrite/--no-overwrite` toggles and repeated `--inputs`).

## Flag details

### convert
- `--freq` target sample rate in Hz (default `44100`). Shannon-correct
  resampling via ffmpeg / libswresample.
- `--channels` output channel count (`1` = mono, default).
- `--encoding` ffmpeg codec name (default `pcm_s16le`; use e.g. `libmp3lame`
  for mp3, `aac` for m4a — usually the output extension is enough).
- Output container is chosen from the `--output` extension.

### chunk
- `--start` / `--end` in seconds (floats). Validated against real duration;
  out-of-range bounds raise an error.
- `--output` optional — auto-named `<stem>_chunk-<ms>-<ms>.<ext>` if omitted.

### silence
- `--duration` seconds (float, required).
- `--sample-rate` default `44100`.

### concat
- `--inputs` takes 2+ paths **in order** (argparse: `--inputs a b c`; click:
  repeat `--inputs a --inputs b`). Order is the concatenation order.

### roomtone
- `--db` noise level in dB (default `-42`; typical post-production `-45..-38`).
- `--color` one of `white pink brown red blue violet velvet` (default `pink`).
- Standard trick to mask cuts in a montage of speech takes.

### split
- `--seconds` chunk duration (float). Chunks named
  `chunk_NNNN_<suffix>.<output-format>`.
- `--output-format` default `mp3`; `--suffix` default `split`.

### separate (Demucs)
- Splits into `vocals / drums / bass / other` (`HDEMUCS_HIGH_MUSDB_PLUS`).
- `--device` `cuda` / `cpu` / omit (auto).
- `--workers` thread count, sklearn convention (`-2` = all cores but one;
  forced to 1 on CUDA). Ignored effectively unless CPU.
- Requires the `[demucs]` extra (torch + torchaudio). Missing → exit code 2
  with an install hint on stderr (`print(sources)` JSON on success).

### resemblance
- `--a` / `--b` two audio paths. Prints a float in `[0, 1]` (MFCC cosine
  similarity); `1.0` ≈ identical, near `0` ≈ unrelated.

## Output contract (for scripting)

- `convert` / `chunk` / `silence` / `concat` / `roomtone` print the single
  output path to stdout.
- `split` prints one chunk path per line.
- `separate` prints a JSON `{stem: path}` map.
- `duration` prints seconds (6 decimals); `resemblance` prints the score
  (6 decimals).

## Supported inputs

All common audio containers (`mp3 wav m4a flac ogg opus aac wma aiff …`) **and**
video containers whose audio track ffmpeg can decode (`mp4 mkv mov webm avi
mpeg …`). Extension list lives in `audio_helper.main.audio_extensions`.
