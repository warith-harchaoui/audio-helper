# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['audio_helper']

package_data = \
{'': ['*']}

install_requires = \
['ffmpeg-python>=0.2.0,<0.3.0',
 'numpy>=1.24',
 'os-helper @ git+https://github.com/warith-harchaoui/os-helper.git@v1.2.0',
 'scipy>=1.15.1,<2.0.0',
 'soundfile>=0.13.0,<0.14.0',
 'tqdm>=4.67.1,<5.0.0']

extras_require = {
    'demucs': ['torch>=2.5.1,<3.0.0', 'torchaudio>=2.5.1,<3.0.0'],
    'dev': ['pytest>=8.0,<9.0', 'torch>=2.5.1,<3.0.0', 'torchaudio>=2.5.1,<3.0.0'],
}

setup_kwargs = {
    'name': 'audio-helper',
    'version': '1.4.1',
    'description': 'Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.',
    'long_description': '# Audio Helper\n\n`Audio Helper` belongs to a collection of libraries called `AI Helpers` developed for building Artificial Intelligence.\n\n[🕸️ AI Helpers](https://harchaoui.org/warith/ai-helpers)\n\n[![logo](assets/repository-open-graph-template.png)](https://harchaoui.org/warith/ai-helpers)\n\nAudio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.\n',
    'author': 'Warith Harchaoui',
    'author_email': 'warith@heedgi.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.10,<3.14',
}


setup(**setup_kwargs)
