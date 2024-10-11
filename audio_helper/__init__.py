"""
Audio Helper

This module provides various helper functions to manipulate audio files.
It includes functionality for audio conversion, audio extraction, and audio source separation.

Authors:
- [Warith Harchaoui](https://harchaoui.org/warith)
- [Mohamed Chelali](https://mchelali.github.io)
- [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug)
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
    "split_audio_regularly",
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
    split_audio_regularly,
)
