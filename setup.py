# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['audio_helper']

package_data = \
{'': ['*']}

install_requires = \
['ffmpeg-python>=0.2.0,<0.3.0',
 'os-helper @ git+https://github.com/warith-harchaoui/os-helper.git@v1.0.0',
 'scipy>=1.15.1,<2.0.0',
 'soundfile>=0.13.0,<0.14.0',
 'torch>=2.5.1,<3.0.0',
 'torchaudio>=2.5.1,<3.0.0',
 'tqdm>=4.67.1,<5.0.0']

setup_kwargs = {
    'name': 'audio-helper',
    'version': '0.1.0',
    'description': 'Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.',
    'long_description': '# Audio Helper\n\n`Audio Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.\n\n[🕸️ AI Helpers](https://harchaoui.org/warith/ai-helpers)\n\n[![logo](assets/repository-open-graph-template.png)](https://harchaoui.org/warith/ai-helpers)\n\nAudio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.\n\n# Installation\n\n## Install Package\n\nWe recommend using Python environments. Check this link if you\'re unfamiliar with setting one up:\n\n[🥸 Tech tips](https://harchaoui.org/warith/4ml/#install)\n\n## Install `ffmpeg` \nTo install Audio Helper, you must install `ffmpeg`:\n\n- For macos 🍎\n  \n  Get [brew](https://brew.sh)\n  ```bash\n  brew install ffmpeg\n  ```\n- For Ubuntu 🐧\n  ```bash\n  sudo apt install ffmpeg\n  ```\n- For Windows 🪟\n  \n  Go to the [FFmpeg website](https://ffmpeg.org/download.html) and follow the instructions for downloading FFmpeg. You\'ll need to manually add FFmpeg to your system PATH.\n  \nand finally we still discuss between different python package managers and try to support as much as possible\n\n\n\n```bash\npip install --force-reinstall --no-cache-dir git+https://github.com/warith-harchaoui/audio-helper.git@v1.0.0\n```\n\n# Usage\nHere’s an example of how to use Audio Helper to load, convert, and split an audio file:\n\n(download [example.mp3](https://harchaoui.org/warith/example.mp3) )\n\nIt is part of a JFK speech that is badly recorded\n\n```python\nimport audio_helper as ah\n\n# Load an audio file\naudio_file = "example.mp3"\naudio, sample_rate = ah.load_audio(audio_file)\n\n# Convert the audio file to a different format\noutput_audio = "audio_tests/example.wav"\nah.sound_converter(audio_file, output_audio)\n\n# Split the audio file into chunks of 30 seconds\nchunks = ah.split_audio_regularly(audio_file, "audio_tests/chunks_folder", split_time=30.0, overwrite = True)\n# Concatenate the chunks back together\nnew_concatenated_audio = "audio_tests/concatenated.wav"\nconcatenated_audio = ah.audio_concatenation(chunks, output_audio_filename = new_concatenated_audio)\n```\n\nAnother cool example is about source separation (DEMUCS from META) with AI separating one audio track into 4 tracks:\n- vocals\n- drums\n- bass\n- other\n\nIt works with speech and songs\n\n```python\nimport audio_helper as ah\n\naudio_path = "input_audio.m4a"\n\nsources = ah.separate_sources(\n    audio_path,\n    output_folder="audio_tests",\n    device = "cpu", # or "cuda" if GPU or nothing to let it decide\n    nb_workers = 4, # ignored if not cpu\n    output_format = "mp3",\n)\n\nprint(separated_sources)\n# {\'vocals\': \'audio_tests/vocals.mp3\', \'drums\': \'audio_tests/drums.mp3\', \'bass\': \'audio_tests/bass.mp3\', \'other\': \'audio_tests/other.mp3\'}\n```\n\n# Features\n- Audio Loading: Load audio files with optional resampling and mono conversion.\n- Sound Conversion: Convert audio files to different formats using ffmpeg.\n- Source Separation: Separate an audio file into vocals, drums, bass, and other stems using a pre-trained PyTorch model (Demucs).\n- Audio Splitting: Split audio files into chunks based on duration.\n- Concatenation: Concatenate multiple audio files into one.\n- Silent Audio Generation: Create silent audio files of a specified duration.\n- Chunk Extraction: Extract specific segments from an audio file.\n\n# Authors\n - [Warith Harchaoui](https://harchaoui.org/warith)\n - [Mohamed Chelali](https://mchelali.github.io)\n - [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug)\n\n',
    'author': 'Warith Harchaoui',
    'author_email': 'warith@heedgi.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<3.13',
}


setup(**setup_kwargs)

