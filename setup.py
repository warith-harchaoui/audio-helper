# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['audio_helper']

package_data = \
{'': ['*']}

install_requires = \
['Jinja2==3.1.4',
 'MarkupSafe==3.0.2',
 'PyYAML==6.0.2',
 'certifi==2024.8.30',
 'cffi==1.17.1',
 'charset-normalizer==3.4.0',
 'ffmpeg-python==0.2.0',
 'filelock==3.16.1',
 'fsspec==2024.10.0',
 'future==1.0.0',
 'idna==3.10',
 'mpmath==1.3.0',
 'networkx==3.4.2',
 'numpy==2.1.2',
 'os-helper @ git+https://github.com/warith-harchaoui/os-helper.git@main',
 'pandas==2.2.3',
 'pillow==11.0.0',
 'pycparser==2.22',
 'python-dateutil==2.9.0.post0',
 'python-dotenv==1.0.1',
 'pytz==2024.2',
 'requests==2.32.3',
 'scipy==1.14.1',
 'setuptools==75.1.0',
 'six==1.16.0',
 'soundfile==0.12.1',
 'sympy==1.13.1',
 'torch==2.5.0',
 'torchaudio==2.5.0',
 'torchvision==0.20.0',
 'tqdm==4.66.5',
 'typing_extensions==4.12.2',
 'tzdata==2024.2',
 'urllib3==2.2.3',
 'validators==0.34.0',
 'wheel==0.44.0']

setup_kwargs = {
    'name': 'audio-helper',
    'version': '0.1.0',
    'description': 'Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.',
    'long_description': 'Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files.',
    'author': 'Warith Harchaoui',
    'author_email': 'warith.harchaoui@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.12,<4.0',
}


setup(**setup_kwargs)

