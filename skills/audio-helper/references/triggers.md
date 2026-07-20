# audio-helper skill — exhaustive trigger catalogue

Auditable superset of the `description:` TRIGGER clause in `SKILL.md` (the
description is what a host model sees before loading; this file is the
human-reviewable full list). Keep the two in sync.

## Fire (positive triggers)

**Format / rate / channel conversion**
- "convert this mp3 to wav / flac / m4a / ogg"
- "make it mono", "downmix to mono", "force stereo"
- "resample to 16 kHz / 22050 / 44100 / 48000", "change the sample rate"
- "re-encode / transcode this audio", "change the codec"
- "extract the audio from this video (mp4/mkv/mov/webm)"

**Slicing / trimming**
- "cut / trim from 3s to 8s", "extract seconds A to B", "grab the first minute"
- "keep just this segment", "crop the clip"

**Splitting**
- "split this into 30-second chunks", "chop into fixed-length pieces"
- "break this long recording into parts"

**Concatenation**
- "join / merge / stitch / concatenate these files head-to-tail"
- "combine take1, take2, take3 into one file"

**Silence**
- "generate 5 seconds of silence", "make a silent wav / pad file"
- "I need a blank audio spacer"

**Room tone**
- "add room tone / ambient hiss / background noise to mask the cuts"
- "the edits between takes are too abrupt / audible", "smooth the montage"
- "mix pink / white / brown noise under this speech"

**Source separation (Demucs, `[demucs]` extra)**
- "separate the vocals / isolate the singer", "extract the stems"
- "remove the music / instrumental", "make a karaoke / acapella"
- "split this song into vocals / drums / bass / other"

**Similarity**
- "how similar are these two recordings", "compare these two takes"
- "MFCC similarity / audio fingerprint / resemblance score"

**Probing**
- "how long is this audio", "get the duration", "is this a valid audio file"

**Explicit command / function mentions**
- `audio-helper`, `audio-helper-click`, `audio-helper-mcp`
- subcommands `convert duration chunk silence concat roomtone split separate resemblance`
- functions `load_audio sound_converter extract_audio_chunk split_audio_regularly
  audio_concatenation generate_silent_audio mix_room_tone sound_resemblance
  separate_sources get_audio_duration is_valid_audio_file save_audio`

**Surfaces**
- "run the audio API / audio-helper server", "expose these as HTTP / MCP tools"
- "open the audio GUI / drag-and-drop bench", "audition bench"
- "how do I install / run audio-helper"

**File-type cues** (with a transform intent)
- audio: `.mp3 .wav .m4a .m4b .flac .ogg .oga .opus .aac .wma .aiff .aif .au`
- video (audio track): `.mp4 .mkv .mov .webm .avi .mpeg .mpg .m4v .3gp .ts`

## Do NOT fire (SKIP)

- **Transcription / captions / subtitles / speech-to-text** → vocal-helper /
  whisper skill. audio-helper does not read words out of audio.
- **Text-to-speech / voice cloning / synthesis / music generation** → not this
  skill.
- **Downloading audio from YouTube / a URL** → youtube-helper.
- **DAW editing**: EQ, compression, reverb, de-noise (beyond room-tone), fades,
  automation, mastering, loudness normalization (LUFS) → not this skill.
- **Pure DSP math** with no file to transform.

## Enforcement checklist

A trigger is "enforced" when (1) it is represented in `SKILL.md`'s
`description` TRIGGER clause so the host sees it pre-load; (2) the SKIP clause is
present so the skill does not over-fire; (3) this catalogue lists the positive
and negative buckets so a human can audit coverage against the description.
