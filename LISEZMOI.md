# Audio Helper

[🇫🇷](LISEZMOI.md) · [🇬🇧](README.md)

[![CI](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#)

`Audio Helper` fait partie d'une collection de bibliothèques appelée `AI Helpers`, développée pour bâtir des applications d'intelligence artificielle.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](assets/logo.png)](https://harchaoui.org/warith/ai-helpers)

Audio Helper est une bibliothèque Python qui fournit des fonctions utilitaires pour le traitement de fichiers audio. Elle inclut le chargement audio, la conversion de formats, la séparation de sources, et le découpage / la concaténation de fichiers audio.

# Installation

## Installer le paquet

Nous recommandons l'utilisation d'environnements Python. Consultez ce lien si vous ne savez pas comment faire :

[🥸 Conseils techniques](https://harchaoui.org/warith/4ml/#install)

## Installer `ffmpeg`
Pour installer Audio Helper, vous devez d'abord installer `ffmpeg` :

- Sous macOS 🍎

  Récupérer [brew](https://brew.sh)
  ```bash
  brew install ffmpeg
  ```
- Sous Ubuntu 🐧
  ```bash
  sudo apt install ffmpeg
  ```
- Sous Windows 🪟

  Allez sur le [site FFmpeg](https://ffmpeg.org/download.html) et suivez les instructions. Il faut ajouter manuellement FFmpeg au PATH système.

Pour finir, nous discutons encore entre différents gestionnaires de paquets Python et essayons de supporter autant que possible.

Audio Helper se livre en deux versions. Choisissez celle qu'il vous faut :

```bash
# Utilitaires audio de base (load, convert, split, concatenate, silent audio, chunks)
pip install --force-reinstall --no-cache-dir \
  "git+https://github.com/warith-harchaoui/audio-helper.git@v1.4.2"

# Ajout de la séparation de sources Demucs (tire torch + torchaudio, ~2 Go)
pip install --force-reinstall --no-cache-dir \
  "audio-helper[demucs] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.4.2"
```

Si vous appelez `separate_sources` sans l'extra `[demucs]`, la fonction lève une `ImportError` qui vous redirige ici.

# Utilisation

Pour le catalogue complet d'exemples, voir [📋 EXAMPLES.md](EXAMPLES.md).

Voici un exemple d'utilisation d'Audio Helper pour charger, convertir et découper un fichier audio :

(télécharger [example.mp3](https://harchaoui.org/warith/example.mp3))

C'est un extrait d'un discours de JFK qui est très mal enregistré.

```python
import audio_helper as ah

# Charger un fichier audio
audio_file = "example.mp3"
audio, sample_rate = ah.load_audio(audio_file)

# Convertir le fichier audio vers un autre format
output_audio = "audio_tests/example.wav"
ah.sound_converter(audio_file, output_audio)

# Découper le fichier audio en morceaux de 30 secondes
chunks = ah.split_audio_regularly(audio_file, "audio_tests/chunks_folder", split_time=30.0, overwrite = True)
# Reconcaténer les morceaux
new_concatenated_audio = "audio_tests/concatenated.wav"
concatenated_audio = ah.audio_concatenation(chunks, output_audio_filename = new_concatenated_audio)
```

Un autre exemple intéressant concerne la séparation de sources (DEMUCS de META) avec une IA qui sépare une piste audio en 4 pistes :
- voix (vocals)
- batterie (drums)
- basse (bass)
- autres (other)

Cela fonctionne aussi bien avec la parole qu'avec les chansons.

```python
import audio_helper as ah

audio_path = "input_audio.m4a"

sources = ah.separate_sources(
    audio_path,
    output_folder="audio_tests",
    device = "cpu", # ou "cuda" si GPU disponible, ou rien pour laisser décider
    nb_workers = 4, # ignoré si pas en mode cpu
    output_format = "mp3",
)

print(sources)
# {'vocals': 'audio_tests/vocals.mp3', 'drums': 'audio_tests/drums.mp3', 'bass': 'audio_tests/bass.mp3', 'other': 'audio_tests/other.mp3'}
```

# Fonctionnalités
- Chargement audio : charger des fichiers avec rééchantillonnage et downmix mono optionnels.
- Conversion : conversion de format / fréquence d'échantillonnage / canaux via ffmpeg.
- Séparation de sources : voix / batterie / basse / autres via Demucs (extra `[demucs]` optionnel).
- Découpage audio : morceaux de durée fixe ou tranches arbitraires `[start, end]`.
- Concaténation : jointure bout-à-bout dans n'importe quel conteneur ffmpeg.
- Génération de silence : écrire un silence d'une durée spécifiée.
- Mixage de bruit ambiant (room-tone) : bruit rose / blanc / brun pour masquer des coupures.
- Similarité : score `sound_resemblance` basé sur les MFCC pour comparaison A/B.
- Extraction de caractéristiques : primitives Mel / MFCC basées sur scipy.

# Auteur
 - [Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)

# Remerciements
Special thanks to [Mohamed Chelali](https://mchelali.github.io) and [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug) for fruitful discussions.
