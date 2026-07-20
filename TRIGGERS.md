# TRIGGERS — audio-helper

This is the user-facing, exhaustive catalogue of what `audio-helper` can do and
the natural-language phrasings, commands, functions, and file types that should
invoke it — whether you call it yourself or drive it as a Claude / OpenCode
**skill** (see [`skills/audio-helper/SKILL.md`](skills/audio-helper/SKILL.md)
and its [`references/triggers.md`](skills/audio-helper/references/triggers.md)).

`audio-helper` transforms audio **files** (and the audio track of video files).
It is local-first and ffmpeg-backed. It does **not** transcribe, synthesize, or
download audio.

## The nine operations → how to invoke

| Intent | CLI | Library | API / MCP |
|--------|-----|---------|-----------|
| Re-encode format / sample rate / channels | `audio-helper convert` | `sound_converter` | `POST /convert` |
| Get duration in seconds | `audio-helper duration` | `get_audio_duration` | `POST /duration` |
| Extract a `[start, end]` slice | `audio-helper chunk` | `extract_audio_chunk` | `POST /chunk` |
| Generate silence | `audio-helper silence` | `generate_silent_audio` | `POST /silence` |
| Concatenate files | `audio-helper concat` | `audio_concatenation` | `POST /concat` |
| Mix room-tone ambient noise | `audio-helper roomtone` | `mix_room_tone` | `POST /roomtone` |
| Split into fixed-duration chunks | `audio-helper split` | `split_audio_regularly` | `POST /split` |
| Separate stems (Demucs) | `audio-helper separate` | `separate_sources` | `POST /separate` |
| MFCC similarity score | `audio-helper resemblance` | `sound_resemblance` | `POST /resemblance` |

Every operation is also reachable through the click CLI (`audio-helper-click …`,
same flags) and the browser GUI at `GET /gui`.

## Natural-language phrasings that should fire

- **Convert**: "convert this mp3 to wav", "make it mono 16 kHz", "re-encode /
  transcode", "resample to 48000", "extract the audio from this mp4".
- **Chunk**: "cut from 3s to 8s", "keep the first minute", "trim / crop".
- **Split**: "split into 30-second chunks", "chop this into pieces".
- **Concat**: "merge / join / stitch these files", "combine into one".
- **Silence**: "generate 5 seconds of silence", "make a blank spacer".
- **Room tone**: "mask the cuts with room tone", "add pink/brown ambient hiss",
  "the edits between takes are too abrupt".
- **Separate**: "isolate the vocals", "remove the music", "karaoke / acapella",
  "extract vocals/drums/bass/other" *(needs `[demucs]`)*.
- **Resemblance**: "how similar are these two clips", "compare these takes",
  "MFCC / audio fingerprint similarity".
- **Probe**: "how long is this", "is this a valid audio file".
- **Surfaces**: "run the audio API / MCP server", "open the audio GUI", "install
  audio-helper".

## File types it accepts

- **Audio**: `.mp3 .wav .m4a .m4b .flac .ogg .oga .opus .aac .wma .aiff .aif .au`
  (and more — full list in `audio_helper.main.audio_extensions`).
- **Video** (audio track decoded via ffmpeg): `.mp4 .mkv .mov .webm .avi .mpeg
  .mpg .m4v .3gp .ts …`.

## When NOT to use audio-helper (SKIP)

- Transcription / captions / subtitles / speech-to-text → use `vocal-helper` /
  a whisper skill.
- Text-to-speech, voice cloning, synthesis, music generation.
- Downloading audio from YouTube or a URL → use `youtube-helper`.
- DAW-style editing (EQ, compression, reverb, de-noise beyond room-tone, fades,
  mastering, loudness normalization).
- Pure signal-processing math with no file in play.

## See also

- [`README.md`](README.md) — features, install, quick start.
- [`EXAMPLES.md`](EXAMPLES.md) — runnable recipes.
- [`GUI.md`](GUI.md) — the shipped minimal GUI + the roadmap for a richer one.
- [`skills/README.md`](skills/README.md) — installing this as an agent skill.
