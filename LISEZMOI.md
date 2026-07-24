# Audio Helper

[🇫🇷](LISEZMOI.md) · [🇬🇧](README.md)

[![CI](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#) [![Local-first](https://img.shields.io/badge/local--first-ffmpeg%20%2B%20Demucs%20local-brightgreen.svg)](#la-promesse)

`Audio Helper` fait partie d'une collection de bibliothèques appelée `AI Helpers`, développée pour bâtir des applications d'intelligence artificielle.

[🌍 AI Helpers](https://harchaoui.org/warith/ai-helpers)

[![logo](assets/logo.png)](https://harchaoui.org/warith/ai-helpers)

Audio Helper est une bibliothèque Python qui fournit des fonctions utilitaires pour le traitement de fichiers audio. Elle inclut le chargement audio, la conversion de formats, la séparation de sources, et le découpage / la concaténation de fichiers audio.

## La promesse

Audio Helper est **local-first** par conception. Trois cas, en toute honnêteté :

1. **Garanti local.** Chaque opération — y compris l'interface web sur
   `GET /gui` — s'exécute sur votre machine via **ffmpeg** et **Demucs en
   local**. Votre audio n'est **jamais téléversé** vers un tiers. **Aucune
   télémétrie, aucun compte, aucune dépendance SaaS.**
2. **La seule réserve : les poids des modèles.** La séparation de sources
   télécharge les poids du modèle **Demucs** **une seule fois**, au premier
   lancement (un simple cache Hugging Face / PyTorch). Ensuite, tout fonctionne
   hors ligne. Rien d'autre n'a besoin du réseau.
3. **Votre décision.** Rien ici ne force le cloud. Si vous souhaitez un jour
   déployer derrière un proxy ou dans un conteneur, la surface FastAPI le rend
   facile — mais c'est un choix que vous faites, pas un défaut que nous imposons.

## Documentation

[💻 Documentation](https://harchaoui.org/warith/ai-helpers/docs/audio-helper-doc/)

[🗺️ Paysage](https://github.com/warith-harchaoui/audio-helper/blob/main/PAYSAGE.md)

[📋 Exemples](https://github.com/warith-harchaoui/audio-helper/blob/main/EXAMPLES.md)

## Fonctionnalités

- Chargement audio : charger des fichiers avec rééchantillonnage et downmix mono optionnels.
- Conversion : conversion de format / fréquence d'échantillonnage / canaux via ffmpeg.
- Séparation de sources : voix / batterie / basse / autres via Demucs (extra `[demucs]` optionnel).
- Découpage audio : morceaux de durée fixe ou tranches arbitraires `[start, end]`.
- Concaténation : jointure bout-à-bout dans n'importe quel conteneur ffmpeg.
- Génération de silence : écrire un silence d'une durée spécifiée.
- Mixage de bruit ambiant (room-tone) : bruit rose / blanc / brun pour masquer des coupures.
- Similarité : score `sound_resemblance` basé sur les MFCC pour comparaison A/B.
- Extraction de caractéristiques : primitives Mel / MFCC basées sur scipy.

**Cinq surfaces, une boîte à outils** — chaque opération ci-dessus est
accessible via :

- **Bibliothèque** : `import audio_helper as ah`.
- **CLI ×2** : `audio-helper` (argparse, toujours installée) et
  `audio-helper-click` (jumelle click, extra `[cli]`), aux options identiques.
- **API HTTP** : application FastAPI (extra `[api]`), docs OpenAPI sur `/docs`.
- **MCP** : jeu d'outils FastAPI-MCP (extra `[api,mcp]`) pour hôtes MCP.
- **GUI** : un **Recipe Canvas** dans le navigateur, sans étape de build, servi
  sur `GET /gui` — enchaînez les huit verbes en un pipeline séquentiel, écoutez
  chaque étape intermédiaire (formes d'onde WaveSurfer), court-circuitez
  n'importe quelle étape pour un A/B instantané, utilisez le comparateur
  avant/après (bascule à la barre d'espace), et exportez le pipeline en un
  `recipe.yaml` versionnable. Voir [GUI.md](GUI.md).

Elle s'installe aussi comme **skill Claude / OpenCode** — voir
[skills/README.md](skills/README.md) et le catalogue exhaustif
[TRIGGERS.md](TRIGGERS.md).

## Installation

**Prérequis** — **Python 3.10–3.13** et **git**, **ffmpeg**, multiplateforme :

- 🍎 **macOS** ([Homebrew](https://brew.sh)) : `brew install python git ffmpeg`
- 🐧 **Ubuntu/Debian** : `sudo apt update && sudo apt install -y python3 python3-pip git ffmpeg`
- 🪟 **Windows** (PowerShell) : `winget install Python.Python.3.12 Git.Git Gyan.FFmpeg`

Nous recommandons l'utilisation d'environnements Python. Consultez ce lien si vous ne savez pas comment faire : [🥸 Conseils techniques](https://harchaoui.org/warith/4ml/#install).

### Depuis PyPI (recommandé)

```bash
# Utilitaires audio de base (load, convert, split, concatenate, silent audio, chunks)
pip install audio-helper

# Ajout de la séparation de sources Demucs (tire torch + torchaudio, ~2 Go)
pip install "audio-helper[demucs]"

# Surfaces optionnelles
pip install "audio-helper[cli]"       # CLI click jumelle
pip install "audio-helper[api]"       # surface HTTP FastAPI
pip install "audio-helper[api,mcp]"   # outils MCP au-dessus de FastAPI
```

### Depuis les sources (sans PyPI)

```bash
# Utilitaires audio de base
pip install "git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"

# Ajout de la séparation de sources Demucs (tire torch + torchaudio, ~2 Go)
pip install "audio-helper[demucs] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"

# Surfaces optionnelles
pip install "audio-helper[cli] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
pip install "audio-helper[api] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
pip install "audio-helper[api,mcp] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
```

Si vous appelez `separate_sources` sans l'extra `[demucs]`, la fonction lève une `ImportError` qui vous redirige ici.

## Utilisation

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

## Exposition multi-surface

`audio-helper` n'est pas qu'une bibliothèque — les mêmes fonctions sont
exposées en CLI, en API HTTP FastAPI et en outils MCP :

```bash
# Bibliothèque Python (par défaut)
import audio_helper as ah

# CLI argparse (installée automatiquement)
audio-helper convert --input in.mp3 --output out.wav --freq 44100
audio-helper split --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper separate --input mix.mp3 --output-dir stems/
audio-helper resemblance --a a.mp3 --b b.mp3

# CLI click jumelle (extra [cli])
pip install "audio-helper[cli]"
# ou depuis les sources :
pip install "audio-helper[cli] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
audio-helper-click convert --input in.mp3 --output out.wav --freq 44100

# Surface HTTP FastAPI (extra [api])
pip install "audio-helper[api]"
# ou depuis les sources :
pip install "audio-helper[api] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
uvicorn audio_helper.api:app --port 8000
# → docs OpenAPI sur http://localhost:8000/docs

# Outils MCP au-dessus de FastAPI (extras [api,mcp])
pip install "audio-helper[api,mcp]"
# ou depuis les sources :
pip install "audio-helper[api,mcp] @ git+https://github.com/warith-harchaoui/audio-helper.git@v1.6.0"
audio-helper-mcp                  # sert FastAPI + MCP sur le port 8000
```

Image Docker (légère, sans Demucs par défaut) :

```bash
docker build -t audio-helper .
docker run --rm -p 8000:8000 audio-helper
# avec Demucs :
docker build --build-arg WITH_DEMUCS=1 -t audio-helper:demucs .
```

Une GUI minimale (« banc d'écoute ») est déjà disponible : elle est servie
par l'application FastAPI sur `GET /gui` (ouvrez `http://localhost:8000/gui`
après avoir lancé le serveur). La GUI ambitieuse à venir (éditeur de recettes
en canvas, comparateur « ear-first », vue MFCC-cluster batch) est décrite comme
feuille de route dans [GUI.md](GUI.md).

Le paysage concurrentiel (librosa, torchaudio, pydub, essentia,
Demucs, Spleeter, …) est analysé avec une carte de positionnement dans [PAYSAGE.md](PAYSAGE.md).

## Auteur

 - [Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)

## Remerciements

Remerciements chaleureux à [Mohamed Chelali](https://mchelali.github.io) et [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug) pour nos échanges fructueux.

## Licence

Ce projet est distribué sous licence BSD-3-Clause — voir le fichier [LICENSE](LICENSE) pour les détails.
