"""
Audio Helper — public API surface.

Re-exports the utility functions from :mod:`audio_helper.main` so that
downstream code can simply write ``import audio_helper as ah`` and reach
every supported operation (load, save, convert, split, concatenate,
silence generation, room-tone mix, MFCC similarity, and optional Demucs
source separation) without knowing about the module layout.

Usage Example
-------------
>>> import audio_helper as ah
>>> audio, sr = ah.load_audio("example.mp3", to_numpy=True, to_mono=True)
>>> ah.sound_converter("example.mp3", "example.wav")
>>> chunks = ah.split_audio_regularly("example.mp3", "chunks/", split_time=30.0)

Author
------
Warith Harchaoui, Ph.D. — https://linkedin.com/in/warith-harchaoui/
"""

__author__ = "Warith Harchaoui, Ph.D."
__email__ = "warithmetics@deraison.ai"

# Specify the public API of this module using __all__
__all__ = [
    "is_valid_audio_file",
    "get_audio_duration",
    "load_audio",
    "sound_converter",
    "separate_sources",
    "extract_audio_chunk",
    "generate_silent_audio",
    "audio_concatenation",
    "mix_room_tone",
    "split_audio_regularly",
    "save_audio",
    "sound_resemblance",
]


from .main import (
    audio_concatenation,
    extract_audio_chunk,
    generate_silent_audio,
    get_audio_duration,
    is_valid_audio_file,
    load_audio,
    mix_room_tone,
    save_audio,
    separate_sources,
    sound_converter,
    sound_resemblance,
    split_audio_regularly,
)
