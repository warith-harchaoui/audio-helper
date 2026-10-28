# Audio Helper

[🇫🇷](https://github.com/warith-harchaoui/audio-helper/blob/main/LISEZMOI.md) · [🇬🇧](https://github.com/warith-harchaoui/audio-helper/blob/main/README.md)

[![CI](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/warith-harchaoui/audio-helper/actions/workflows/ci.yml) [![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/warith-harchaoui/audio-helper/blob/main/LICENSE) [![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](#) [![Local-first](https://img.shields.io/badge/local--first-ffmpeg%20%2B%20Demucs%20local-brightgreen.svg)](#la-promesse)

`Audio Helper` fait partie d'une collection de bibliothèques appelée `AI Helpers`, développée pour bâtir des applications d'intelligence artificielle.

[🌍 AI Helpers](https://deraison.ai/ai-helpers)

[![logo](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/logo.png)](https://deraison.ai/ai-helpers)

Audio Helper est une bibliothèque Python qui fournit des fonctions utilitaires pour le traitement de fichiers audio. Elle inclut le chargement audio, la conversion de formats, la séparation de sources et le découpage / la concaténation de fichiers audio.

Audio Helper est éprouvé en production. Chaque changement passe par des tests automatisés et une intégration continue avant d'atteindre PyPI ; les versions suivent le [versionnage sémantique](https://semver.org/lang/fr/). Deux autres paquets publiés de la suite, [`youtube-helper`](https://github.com/warith-harchaoui/youtube-helper) et [`vocal-helper`](https://github.com/warith-harchaoui/vocal-helper), s'appuient directement dessus en production. Audio Helper repose lui-même sur [`os-helper`](https://github.com/warith-harchaoui/os-helper), le socle commun que chaque bibliothèque de la suite utilise pour la journalisation et la gestion des fichiers. C'est ce que « éprouvé en production » veut dire ici : pas une promesse, mais un graphe de dépendances sur lequel repose déjà du code publié, dans les deux sens autour de ce paquet.

## La promesse

Audio Helper est **local-first** par conception. Trois cas, en toute honnêteté :

1. **Garanti local.** Chaque opération, y compris l'interface web sur
   `GET /gui`, s'exécute sur votre machine via **ffmpeg** et **Demucs en
   local** (le modèle d'IA de Meta qui sépare une piste mixée en ses
   composantes : voix, batterie, basse, le reste). Votre audio n'est
   **jamais téléversé** vers un tiers. **Aucune télémétrie, aucun compte,
   aucune dépendance SaaS.**
2. **La seule réserve : les poids des modèles.** La séparation de sources
   télécharge les poids d'**Hybrid Demucs** **une seule fois**, au premier
   lancement : `separate_sources` fait tourner le pipeline
   `HDEMUCS_HIGH_MUSDB_PLUS` de torchaudio, qui va chercher environ 335 Mo sur
   `download.pytorch.org` pour les ranger dans le cache PyTorch Hub
   (`torch.hub.get_dir()`, soit `~/.cache/torch/hub` par défaut). Ensuite, tout
   fonctionne hors ligne. Rien d'autre n'a besoin du réseau.
3. **Votre décision.** Rien ici ne force le cloud. Si vous souhaitez un jour
   déployer derrière un proxy ou dans un conteneur, la surface FastAPI le rend
   facile, mais c'est un choix que vous faites, pas un défaut que nous imposons.

## Documentation

[💻 Documentation](https://deraison.ai/ai-helpers/docs/audio-helper-doc/)

[🗺️ Paysage](https://github.com/warith-harchaoui/audio-helper/blob/main/PAYSAGE.md)

[📋 Exemples](https://github.com/warith-harchaoui/audio-helper/blob/main/EXEMPLES.md)

## Fonctionnalités

- Chargement audio : charger des fichiers avec rééchantillonnage et downmix mono optionnels.
- Conversion : conversion de format / fréquence d'échantillonnage / canaux via ffmpeg.
- Séparation de sources : voix / batterie / basse / autres via Demucs (extra `[demucs]` optionnel).
- Découpage audio : morceaux de durée fixe ou tranches arbitraires `[start, end]`.
- Concaténation : jointure bout-à-bout dans n'importe quel conteneur ffmpeg.
- Génération de silence : écrire un silence d'une durée spécifiée.
- Mixage de bruit ambiant (room-tone) : bruit rose / blanc / brun pour masquer des coupures.
- Similarité : score `sound_resemblance` pour comparaison A/B, fondé sur les
  coefficients cepstraux à l'échelle de Mel (_Mel-Frequency Cepstral
  Coefficients_ ou MFCC), une façon standard de réduire le spectre d'un
  extrait à un petit ensemble de nombres qui capture sa sonorité plutôt
  que sa forme d'onde brute, si bien que deux enregistrements de la même
  voix obtiennent un score proche même quand les prises diffèrent
  légèrement.
- Extraction de caractéristiques : primitives Mel / MFCC basées sur scipy.

**Cinq surfaces, une boîte à outils.** Chaque opération ci-dessus est
accessible via :

- **Bibliothèque** : `import audio_helper as ah`.
- **CLI** : `audio-helper` — une seule ligne de commande, en click,
  toujours installée.
- **API HTTP** : application FastAPI (extra `[api]`), docs OpenAPI sur `/docs`.
- **MCP** : la même application FastAPI exposée via le protocole de
  contexte de modèle (_Model Context Protocol_ ou MCP), une norme qui
  permet à un agent d'IA d'appeler les fonctions d'un programme comme des
  outils, sans passer par un humain cliquant dans une interface
  (`audio-helper-mcp`, extra `[mcp]`) pour tout hôte agentique compatible.
- **GUI** : un **Recipe Canvas** dans le navigateur, sans étape de build, servi
  sur `GET /gui`. Enchaînez les huit verbes en un pipeline séquentiel, écoutez
  chaque étape intermédiaire (formes d'onde WaveSurfer), court-circuitez
  n'importe quelle étape pour un A/B instantané, utilisez le comparateur
  avant/après (bascule à la barre d'espace) et exportez le pipeline en un
  `recipe.yaml` versionnable. Voir [GUI.md](https://github.com/warith-harchaoui/audio-helper/blob/main/GUI.md).

Pour le catalogue exhaustif, voir [TRIGGERS.md](https://github.com/warith-harchaoui/audio-helper/blob/main/TRIGGERS.md).

## Comment ça marche

![Comment audio-helper fonctionne : un fichier en entrée, ffmpeg et un Demucs local au milieu, des pistes et des extraits en sortie](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/diagrams/comment-ca-marche.png)

## Installation

**Prérequis** : **Python 3.10–3.13**, **git**, **ffmpeg**, multiplateforme :

- 🍎 **macOS** ([Homebrew](https://brew.sh)) : `brew install python git ffmpeg`
- 🐧 **Ubuntu/Debian** : `sudo apt update && sudo apt install -y python3 python3-pip git ffmpeg`
- 🪟 **Windows** (PowerShell) : `winget install Python.Python.3.12 Git.Git Gyan.FFmpeg`

On recommande de travailler dans un environnement Python. Si vous ne savez pas comment en créer un : [🥸 Conseils techniques](https://deraison.ai/4ml/#install).

### Depuis PyPI (recommandé)

```bash
# Utilitaires audio de base (load, convert, split, concatenate, silent audio, chunks)
pip install audio-helper

# Ajout de la séparation de sources Demucs (tire torch + torchaudio, ~2 Go)
pip install "audio-helper[demucs]"

# Surfaces optionnelles
pip install "audio-helper[api]"       # surface HTTP FastAPI
```

### Depuis les sources (sans PyPI)

```bash
git clone https://github.com/warith-harchaoui/audio-helper.git
cd audio-helper
pip install -e .

# Ajout de la séparation de sources Demucs (tire torch + torchaudio, ~2 Go)
pip install -e ".[demucs]"

# Surfaces optionnelles
pip install -e ".[api]"
```

Si vous appelez `separate_sources` sans l'extra `[demucs]`, la fonction lève une `ImportError` qui vous redirige ici.

## Prise en main rapide

Pour le catalogue complet d'exemples, voir [📋 EXEMPLES.md](https://github.com/warith-harchaoui/audio-helper/blob/main/EXEMPLES.md).

Téléchargez le fichier de test — [example.mp3](https://deraison.ai/example.mp3),
60 secondes d'un discours de JFK très mal enregistré, en MP3 48 kHz stéréo —
puis chargez-le, réencodez-le, découpez-le en fenêtres avant de le recoller :

```python
import audio_helper as ah
import os_helper as osh

# Aucune fonction d'écriture ne crée son dossier de destination : ffmpeg reçoit
# un chemin qu'il suppose déjà existant. os-helper, le socle de gestion de
# fichiers de la suite, s'en charge.
osh.make_directory("out")

# Chargement. `to_numpy=True` renvoie un tableau numpy ; par défaut, la
# fonction renvoie un tenseur torch, ce qui exige l'extra `[demucs]`.
audio, sample_rate = ah.load_audio("example.mp3", to_numpy=True)
print(audio.shape, sample_rate)
# (2880000,) 48000

# Réencodage : conteneur, fréquence d'échantillonnage et canaux en un appel.
print(ah.sound_converter("example.mp3", "out/example.wav", freq=16000, channels=1))
# out/example.wav

# Découpage en fenêtres de 25 secondes, puis recollage bout à bout.
chunks = ah.split_audio_regularly("example.mp3", "out/windows", split_time=25.0, overwrite=True)
print(len(chunks))
# 3
joined = ah.audio_concatenation(chunks, "out/rejoined.mp3", overwrite=True)
print(f"{ah.get_audio_duration(joined):.2f}")
# 60.00
```

L'autre opération phare est la séparation de sources : une piste mixée en
entrée, quatre pistes en sortie (`drums`, `bass`, `other`, `vocals` — l'ordre
dans lequel le bundle Hybrid Demucs de torchaudio les déclare). Elle fonctionne
aussi bien sur la parole que sur les chansons. Elle réclame l'extra `[demucs]`
ainsi que ce téléchargement unique de 335 Mo de poids.

```python
import audio_helper as ah

sources = ah.separate_sources(
    "song.mp3",
    output_folder="stems",
    device="cpu",  # ou "cuda" ; passer None laisse le choix se faire tout seul
    nb_workers=4,  # ignoré sauf si device == "cpu"
    output_format="mp3",
    overwrite=True,
)

for stem, path in sources.items():
    print(stem, path)
```

Chaque piste est écrite sous `<output_folder>/<stem>.<output_format>` et elle
est **mono** : le modèle travaille en stéréo, puis `separate_sources` ramène
chaque piste en mono avant de l'écrire.

## Exposition multi-surface

`audio-helper` n'est pas qu'une bibliothèque : les mêmes fonctions sont
exposées en CLI, en API HTTP FastAPI et en outils MCP :

```bash
# Bibliothèque Python (par défaut)
import audio_helper as ah

# CLI (installée automatiquement)
audio-helper convert --input in.mp3 --output out.wav --freq 44100
audio-helper split --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper separate --input mix.mp3 --output-dir stems/
audio-helper resemblance --a a.mp3 --b b.mp3


# Surface HTTP FastAPI (extra [api])
pip install "audio-helper[api]"
# ou depuis les sources :
pip install "audio-helper[api]"
uvicorn audio_helper.api:app --port 8000
# → docs OpenAPI sur http://localhost:8000/docs

# Outils MCP pour tout hôte agentique compatible (extra [mcp])
pip install "audio-helper[mcp]"
audio-helper-mcp
# → même app + un endpoint /mcp (fastapi-mcp)
```

Image Docker (légère, sans Demucs par défaut) :

```bash
docker build -t audio-helper .
docker run --rm -p 8000:8000 audio-helper
# avec Demucs :
docker build --build-arg WITH_DEMUCS=1 -t audio-helper:demucs .
```

Le **Recipe Canvas** est déjà disponible : servi par l'application FastAPI
sur `GET /gui` (ouvrez `http://localhost:8000/gui` après avoir lancé le
serveur), il couvre déjà le constructeur de pipeline séquentiel, le
comparateur avant/après « ear-first » et l'export/import `recipe.yaml`. Le
tri par lot sur de nombreux fichiers (vue en planche-contact par glisser-
déposer, vue par grappes MFCC) reste, lui, sur la feuille de route. Voir
[GUI.md](https://github.com/warith-harchaoui/audio-helper/blob/main/GUI.md)
pour la répartition complète entre ce qui est livré et ce qui est prévu.

La page parle **français et anglais**. `GET /gui?lang=fr|en` choisit la langue,
un drapeau dans l'en-tête bascule de l'une à l'autre et un code que le fichier
de locales ne porte pas retombe sur la langue par défaut au lieu d'échouer.
Toutes les chaînes vivent dans
[`audio_helper/locales/fr/LC_MESSAGES/audio_helper.po`](https://github.com/warith-harchaoui/audio-helper/blob/main/audio_helper/locales/fr/LC_MESSAGES/audio_helper.po),
un catalogue `gettext` standard lu par `os_helper.Catalog` — le chargeur
partagé par la suite — si bien que
traduire la page n'oblige jamais à ouvrir un fichier Python. Deux choses
restent volontairement dans une seule langue et le fichier dit pourquoi : les
libellés de paramètres de chaque verbe nomment les champs que la requête porte
vraiment et l'en-tête de commentaire d'un `recipe.yaml` exporté appartient à
un artefact versionnable, pas à l'interface.

### En tant que skill d'agent

`skills/audio-helper/` empaquette audio-helper comme **skill Claude** *et* skill
**OpenCode**, pour qu'un agent puisse s'en saisir à votre place. Voir
[`skills/README.md`](https://github.com/warith-harchaoui/audio-helper/blob/main/skills/README.md) pour l'installer (liens
symboliques vers `~/.claude/skills/` et `~/.opencode/skills/`) et
[`TRIGGERS.md`](https://github.com/warith-harchaoui/audio-helper/blob/main/TRIGGERS.md) pour le catalogue exhaustif des
déclencheurs — `scripts/check_triggers.py` maintient la description du skill
alignée dessus ; la suite de tests échoue si une capacité cesse d'être
nommée.

## Version Rust

Un crate compagnon, [`audio-helper-rs`](https://github.com/warith-harchaoui/audio-helper-rs),
réécrit la surface au niveau fichier (sonder, convertir, charger en PCM,
découper, générer du silence, concaténer, insérer un bruit de fond, scinder)
en Rust idiomatique. Ce n'est ni un portage ni un remplacement :
`separate_sources` (Demucs) et `sound_resemblance` (MFCC) restent réservés à
la version Python, disproportionnés à réimplémenter dans un crate Rust.
L'intérêt d'y recourir plutôt qu'à ce paquet : une dépendance qui se compile
directement dans votre propre binaire Rust, sans interpréteur Python, sans
`pip install` ni environnement virtuel sur la machine cible — utile quand
audio-helper n'est qu'une brique d'un programme Rust plus large ou quand la
cible de déploiement ne peut tout simplement pas embarquer de runtime Python.

## Auteur

 - [Warith HARCHAOUI](https://linkedin.com/in/warith-harchaoui)

## Remerciements

Remerciements chaleureux à [Mohamed Chelali](https://mchelali.github.io) et [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug) pour nos échanges fructueux.

## Licence

Ce projet est distribué sous licence BSD-3-Clause : voir le fichier [LICENSE](https://github.com/warith-harchaoui/audio-helper/blob/main/LICENSE) pour les détails.
