# Paysage

🇫🇷 Français · [🇬🇧 LANDSCAPE.md](https://github.com/warith-harchaoui/audio-helper/blob/main/LANDSCAPE.md)

Bibliothèques Python voisines et concurrentes dans l'espace
« manipulation de fichiers audio », comparées à `audio-helper`. Les
notes vont de ⭐ (1) à ⭐⭐⭐⭐⭐ (5), évaluées sur la tâche visée par
`audio-helper` — le traitement audio quotidien des pipelines d'IA
(chargement, conversion, découpage, concaténation, silence, room-tone,
séparation de sources, similarité MFCC). Une bibliothèque optimisée pour
un tout autre usage (par ex. l'extraction d'information musicale, le DSP
temps réel) n'est pas pénalisée — la note reflète seulement l'adéquation
à *ce* créneau.

## En un coup d'œil

<!-- TABLE:START -->
| Manipulation audio | E/S multi-format | Conversion de format | Découpage / concat / silence | Caractéristiques spectrales / MFCC | Séparation de sources | Rééchantillonnage correct | Installation légère | Ergonomie pipeline d'IA |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **audio-helper** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| pydub | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| librosa | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| soundfile | ⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| torchaudio | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| ffmpeg-python | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| sox / pysox | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| audioread | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| scipy.io.wavfile | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Demucs | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Spleeter | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
<!-- TABLE:END -->

## Carte de positionnement

<!-- FIGURE:START -->
Représentation 2D du tableau ci-dessus.

![Carte de positionnement](https://raw.githubusercontent.com/warith-harchaoui/audio-helper/main/assets/paysage.png)

La carte est un résumé en 2D des 8 critères : à lire comme une forme, pas comme un classement. « audio-helper » se situe dans le coin en haut à droite. Les axes se lisent **Horizontal — Facilité d'installation ↔ Flexibilité des formats** et **Vertical — Qualité audio ↔ Polyvalence fonctionnelle**.
<!-- FIGURE:END -->

## Positionnement

`audio-helper` se place volontairement à l'intersection de
l'**ergonomie à la pydub** (chargement / conversion / découpage /
concaténation / silence en une ligne) et des **besoins des pipelines
d'IA** (séparation de sources à la demande, similarité MFCC pour les
comparaisons A/B). Il ne cherche délibérément *pas* à concurrencer
`librosa` sur le versant analyse et il garde `torch`
**optionnel** — on ne paie le coût de ~2 Go de torch/torchaudio que si
l'on appelle réellement `separate_sources` (Demucs est livré derrière
l'extra `[demucs]`). Ce compromis est le principal facteur de
différenciation face à `torchaudio` (torch obligatoire) et face à
`librosa` (centré analyse, mince sur la conversion de format et l'E/S du
quotidien).

La nuance derrière chaque note mérite d'être explicitée. L'E/S
multi-format d'`audio-helper` s'appuie sur un repli ffmpeg, si bien
qu'elle lit et écrit à peu près tout ce que ffmpeg comprend, là où
`soundfile` ne couvre que WAV / FLAC / OGG, où `audioread` est en lecture
seule et où `scipy.io.wavfile` ne gère que le WAV. Côté caractéristiques,
`librosa` est la référence de niveau MIR et mérite cinq
étoiles — `audio-helper` expose la similarité MFCC pour la comparaison
A/B, pas une suite d'analyse complète, d'où sa note médiane. Le
rééchantillonnage correct est une force discrète mais réelle :
`audio-helper`, `pydub`, `soundfile`, `ffmpeg-python` et `audioread`
rééchantillonnent proprement, tandis que `torchaudio` et les modèles de
séparation (`Demucs`, `Spleeter`) en laissent la charge à l'utilisateur.
Sur la séparation, `Demucs` est à l'état de l'art et `audio-helper`
l'enveloppe directement ; `torchaudio` atteint une qualité comparable via
son pipeline HDEMUCS, tandis que `Spleeter` repose sur TensorFlow et
n'est plus maintenu.

## Quand choisir quoi

- **`audio-helper`** — préparation audio pour pipelines d'IA :
  conversions par lots, découpage pour inférence par fenêtres, silence
  et room-tone pour la post-production, similarité par MFCC, Demucs à la
  demande.
- **`librosa`** — travail à forte composante analyse (détection
  d'onsets, suivi de tempo, chroma) qui n'a pas besoin de conversion de
  format arbitraire.
- **`torchaudio`** — quand on est déjà tensor-natif et qu'on veut du
  zéro-copie entre l'E/S audio et son modèle.
- **`pydub`** — script rapide, montage à la volée, sans MFCC ni
  séparation.
- **`soundfile`** — lecture/écriture WAV / FLAC / OGG simple et rapide
  quand on n'a besoin ni de conversion ni d'analyse.
- **`ffmpeg-python` / `sox` / `pysox`** — conversion de format et
  transcodage lourds, en pilotant soi-même la CLI ffmpeg ou SoX.
- **`Demucs` / `Spleeter`** — séparation de sources en production avec
  son propre wrapper autour du modèle sous-jacent.
</content>
