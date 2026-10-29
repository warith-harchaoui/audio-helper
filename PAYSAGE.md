# Paysage

🇫🇷 Français · [🇬🇧 LANDSCAPE.md](https://github.com/warith-harchaoui/audio-helper/blob/main/LANDSCAPE.md)

Bibliothèques Python voisines et concurrentes dans l'espace
« manipulation de fichiers audio », comparées à `audio-helper`. Les
notes vont de ⭐ (1) à ⭐⭐⭐⭐⭐ (5), évaluées sur la tâche visée par
`audio-helper` : le traitement audio quotidien des pipelines d'IA
(chargement, conversion, découpage, concaténation, silence, room-tone,
séparation de sources, similarité MFCC). Une bibliothèque optimisée pour
un tout autre usage (par ex. l'extraction d'information musicale, le DSP
temps réel) n'est pas pénalisée : la note reflète seulement l'adéquation
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

La carte est un résumé en 2D des 8 critères : à lire comme une forme, pas comme un classement. « audio-helper » se situe dans le coin en haut à droite. Les axes se lisent **Horizontal : Agilité ↔ Maîtrise** et **Vertical : Adaptabilité ↔ Efficience**.
<!-- FIGURE:END -->

## Positionnement

`audio-helper` se place volontairement à l'intersection de
l'**ergonomie à la pydub** (chargement / conversion / découpage /
concaténation / silence en une ligne) et des **besoins des pipelines
d'IA** (séparation de sources à la demande, similarité MFCC pour les
comparaisons A/B). Il ne cherche délibérément *pas* à concurrencer
`librosa` sur le versant analyse et il garde `torch`
**optionnel** : on ne paie le coût de ~2 Go de torch/torchaudio que si
l'on appelle réellement `separate_sources` (Demucs est livré derrière
l'extra `[demucs]`). Ce compromis est le principal facteur de
différenciation face à `torchaudio` (torch obligatoire) et face à
`librosa` (centré analyse, mince sur la conversion de format et l'E/S du
quotidien).

La nuance derrière chaque note mérite d'être explicitée. L'E/S
multi-format d'`audio-helper` passe par ffmpeg pour chaque décodage, si
bien qu'elle lit et écrit à peu près tout ce que ffmpeg comprend, là où
`soundfile` reste borné à la liste de formats de libsndfile (WAV / FLAC /
OGG / AIFF / CAF et, depuis libsndfile 1.1, MP3), où `audioread` se
contente de décoder et où `scipy.io.wavfile` ne gère que le WAV. Côté
caractéristiques, `librosa` est la référence de niveau MIR et mérite cinq
étoiles : `audio-helper` expose la similarité MFCC pour la comparaison
A/B, pas une suite d'analyse complète, d'où sa note médiane.

Le rééchantillonnage correct partage le terrain en trois groupes, pas en
deux. Les bibliothèques qui possèdent un vrai rééchantillonneur
polyphase ou à sinus cardinal fenêtré occupent le haut du tableau :
`audio-helper` et `ffmpeg-python` via libswresample, `sox` / `pysox` via
SoX, `librosa` via `soxr`, `torchaudio` via sa propre transformation
`Resample`. Celles qui ne rééchantillonnent tout simplement pas —
`soundfile`, `audioread`, `scipy.io.wavfile` — récoltent une étoile,
parce que le travail revient entièrement à l'appelant. `pydub` se tient
entre les deux et c'est le cas intéressant :
`AudioSegment.set_frame_rate` appelle `audioop.ratecv` de la bibliothèque
standard, qui interpole entre les échantillons au lieu d'appliquer un
filtre anti-repliement conçu pour cela — un sous-échantillonnage replie
donc le spectre plutôt que de refuser de le faire.

Sur la séparation, Hybrid Demucs est à l'état de l'art et `audio-helper`
ne l'atteint pas via le paquet `demucs` : `separate_sources` fait tourner
le bundle `HDEMUCS_HIGH_MUSDB_PLUS` de torchaudio — même architecture,
entraînée par Meta, livrée comme pipeline torchaudio. C'est toute la
raison pour laquelle l'extra `[demucs]` n'installe que `torch` et
`torchaudio`. `Spleeter` est l'autre option bien connue et elle a vieilli
autrement : la version 2.4.2 épingle `tensorflow==2.12.1` en même temps
que `numpy<2.0.0` et déclare `requires-python >=3.8,<3.12` — elle ne
s'installe donc ni à côté d'un NumPy récent ni sur les versions de Python
que prend en charge `audio-helper`.

## Quand choisir quoi

- **`audio-helper`** : préparation audio pour pipelines d'IA :
  conversions par lots, découpage pour inférence par fenêtres, silence
  et room-tone pour la post-production, similarité par MFCC, Demucs à la
  demande.
- **`librosa`** : travail à forte composante analyse (détection
  d'onsets, suivi de tempo, chroma) qui n'a pas besoin de conversion de
  format arbitraire. Attention au plancher : `librosa` 1.0.0 exige
  Python 3.12 au minimum, si bien que sur 3.10 et 3.11 vous obtenez la
  lignée 0.11.
- **`torchaudio`** : quand on est déjà tensor-natif et qu'on veut du
  zéro-copie entre l'E/S audio et son modèle.
- **`pydub`** : script rapide, montage à la volée, sans MFCC ni
  séparation. Aucune version publiée depuis mars 2021 et le module
  `audioop` sur lequel il repose a quitté la bibliothèque standard en
  Python 3.13 : sur 3.13, il lui faut le rétroportage `audioop-lts`.
- **`soundfile`** : lecture/écriture simple et rapide de ce que
  libsndfile prend en charge, quand on n'a besoin ni de conversion ni
  d'analyse.
- **`ffmpeg-python` / `sox` / `pysox`** : conversion de format et
  transcodage lourds, en pilotant soi-même la CLI ffmpeg ou SoX.
- **`Demucs` / `Spleeter`** : séparation de sources en production avec
  son propre wrapper autour du modèle sous-jacent. Préférez le paquet
  `demucs` amont à celui-ci dès qu'il vous faut les pistes en stéréo, un
  autre point de contrôle Demucs ou le modèle à six pistes :
  `audio-helper` fixe un seul bundle et ramène chaque piste en mono.

## Sources et date de relevé

**Relevé le 2026-10-27.** Les étoiles ci-dessus sont un jugement d'adéquation
à la tâche d'`audio-helper`, pas un banc d'essai : rien dans le tableau n'a été
mesuré sur un corpus commun et une bibliothèque conçue pour un autre usage
n'est pas sanctionnée pour cela. Ce qui est *sourcé*, en revanche, c'est chaque
affirmation factuelle de la prose.

Versions et dates lues dans les métadonnées PyPI de chaque projet
(`https://pypi.org/pypi/<nom>/json`), le 2026-10-27 :

| Projet | Dernière version | Publiée le |
| --- | --- | --- |
| `pydub` | 0.25.1 | 2021-03-10 |
| `librosa` | 1.0.0 | 2026-08-11 |
| `soundfile` | 0.14.0 | 2026-06-06 |
| `torchaudio` | 2.11.0 | 2026-03-23 |
| `ffmpeg-python` | 0.2.0 | 2019-07-06 |
| `sox` (pysox) | 1.5.0 | 2024-03-20 |
| `audioread` | 3.1.0 | 2025-10-26 |
| `scipy` | 1.18.1 | 2026-08-21 |
| `spleeter` | 2.4.2 | 2025-04-03 |
| `demucs` | 4.1.0 | 2026-07-11 |

D'où viennent les autres affirmations :

- **Épinglages et versions de Python prises en charge**, des mêmes
  métadonnées : `spleeter` 2.4.2 déclare `tensorflow==2.12.1`,
  `numpy<2.0.0` ainsi que `requires-python >=3.8,<3.12` ; `librosa` 1.0.0
  déclare `requires-python >=3.12` et dépend de `soxr>=1.0.0`.
- **Le rééchantillonnage de `pydub`**, du code installé de
  `pydub.AudioSegment.set_frame_rate` (0.25.1), qui appelle
  `audioop.ratecv`. `audioop` a été retiré de la bibliothèque standard en
  Python 3.13 par la PEP 594 : c'est la raison d'être du rétroportage
  `audioop-lts`.
- **La liste de formats de `soundfile`**, de `soundfile.available_formats()`
  sur soundfile 0.13.1 / libsndfile 1.2.2, où `MP3` figure.
- **Le modèle de séparation**, de `audio_helper/main.py`
  (`separate_sources` → `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`) ainsi
  que de `pipelines/_source_separation_pipeline.py` de torchaudio. Les poids
  sont récupérés par `torchaudio.utils.download_asset` depuis
  `https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt` ;
  une requête HEAD sur cette URL répond 334 697 255 octets, soit les ~335 Mo
  annoncés par le LISEZMOI.
- **Les notes d'`audio-helper` lui-même**, de ce dépôt en version 2.5.0.

Laissés volontairement à l'état d'opinion plutôt que sourcés : la colonne
*Ergonomie pipeline d'IA* ainsi que l'ordre des recommandations ci-dessus.

## Idées à reprendre

Étudié plutôt que deviné : ce que chaque voisin fait mieux, puis ce que cette bibliothèque devrait changer pour l'égaler. Rien ici n'est un engagement : c'est la liste de lecture dont part la version suivante.

- **`librosa` — le vocabulaire des caractéristiques spectrales** — `sound_resemblance` calcule ici une similarité MFCC, mais le cadrage de librosa — bancs de filtres mel, saut puis fenêtre en paramètres explicites, un jeu de caractéristiques documenté — est la référence que tout lecteur connaît déjà. Nommer les paramètres de la même façon ne coûte rien puis supprime une étape de traduction.

- **`pydub` — une API qui se lit comme le montage qu'on décrit** — Découper un segment avec `[]` puis concaténer avec `+` se lit mieux qu'un appel de fonction pour du travail ponctuel. La surface par appels convient aux pipelines ; une fine enveloppe fluide par-dessus servirait le cas du carnet sans toucher au cœur.

- **`soundfile` — demander à libsndfile ce qu'il gère, ne pas le coder en dur** — `available_formats()` rapporte les formats réellement compilés dans le libsndfile installé. L'équivalent ici est d'interroger le ffmpeg installé plutôt que d'embarquer une liste qui dérive de ce que l'utilisateur possède.

- **`sox` — les effets comme une chaîne composable** — Normaliser, compresser, trembler puis fondre sont des étapes distinctes qu'on empile. Le fond sonore a déjà cette forme ; les autres verbes de post-production se composeraient pareil.
