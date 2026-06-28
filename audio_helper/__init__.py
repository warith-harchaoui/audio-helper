"""
Audio Helper

This module provides various helper functions to manipulate audio files.
It includes functionality for audio conversion, audio extraction, and audio source separation.

Author:
- [Warith HARCHAOUI](https://harchaoui.org/warith)
"""

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
    is_valid_audio_file,
    get_audio_duration,
    load_audio,
    sound_converter,
    separate_sources,
    extract_audio_chunk,
    generate_silent_audio,
    audio_concatenation,
    mix_room_tone,
    split_audio_regularly,
    save_audio,
    sound_resemblance,
)
