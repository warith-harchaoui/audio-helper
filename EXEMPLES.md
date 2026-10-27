# Exemples pratiques d'audio-helper

🇫🇷 Français · [🇬🇧 EXAMPLES.md](https://github.com/warith-harchaoui/audio-helper/blob/main/EXAMPLES.md)

Recettes pratiques pour la surface publique de `audio-helper`. Chaque extrait
ci-dessous a tourné sur un seul fichier —
[example.mp3](https://deraison.ai/example.mp3), 60 secondes d'un discours de
JFK très mal enregistré, en MP3 48 kHz stéréo — et les commentaires de sortie
sont la sortie qui est effectivement revenue. Téléchargez-le à côté de votre
script : toute la page se rejoue.

Deux prérequis pour chaque recette : `ffmpeg` installé, accessible dans le
`PATH`, ainsi que les imports de la [mise en place](#mise-en-place). Seule la
[recette 7](#7-séparer-un-mixage-en-voix-batterie-basse-et-autres) réclame
l'extra optionnel `demucs`.

---

## Table des matières

1. [Mise en place](#mise-en-place)
2. [Valider un fichier et lire sa durée](#1-valider-un-fichier-et-lire-sa-durée)
3. [Charger un enregistrement puis en écrire un](#2-charger-un-enregistrement-puis-en-écrire-un)
4. [Convertir format, fréquence et canaux](#3-convertir-format-fréquence-et-canaux)
5. [Extraire un passage dans un long enregistrement](#4-extraire-un-passage-dans-un-long-enregistrement)
6. [Découper en fenêtres fixes puis recoller](#5-découper-en-fenêtres-fixes-puis-recoller)
7. [Combler avec du silence et masquer les coupes](#6-combler-avec-du-silence-et-masquer-les-coupes)
8. [Séparer un mixage en voix, batterie, basse et autres](#7-séparer-un-mixage-en-voix-batterie-basse-et-autres)
9. [Mesurer la ressemblance entre deux prises](#8-mesurer-la-ressemblance-entre-deux-prises)
10. [Construire ses propres descripteurs Mel et MFCC](#9-construire-ses-propres-descripteurs-mel-et-mfcc)
11. [Ligne de commande](#ligne-de-commande)
12. [API HTTP, MCP et interface web](#api-http-mcp-et-interface-web)
13. [Dépannage](#dépannage)

---

## Mise en place

```bash
# Noyau : E/S via ffmpeg, conversion, découpage, concaténation, silence,
# bruit de fond, MFCC
pip install audio-helper

# Ajoute la séparation de sources — torch et torchaudio, environ 2 Go
pip install "audio-helper[demucs]"
```

Tous les extraits qui suivent partent de ces imports :

```python
import os

import numpy as np

import audio_helper as ah
import os_helper as osh
```

`os_helper` n'est pas décoratif. `audio-helper` journalise à travers lui ; il
porte aussi la gestion de fichiers que les fonctions audio laissent
délibérément de côté. Deux appels méritent d'être connus avant tout le reste :

```python
# 1 = INFO, les lignes de progression d'audio-helper deviennent visibles ;
# 0 = WARNING (le défaut) rend aussi ffmpeg silencieux, car la bibliothèque
# passe quiet=True dès que verbosity() <= 0.
osh.verbosity(1)

# Aucune fonction d'écriture ne crée son dossier de destination. Celle-ci, si.
osh.make_directory("out")
```

## 1. Valider un fichier et lire sa durée

`is_valid_audio_file` lance `ffprobe` *et* vérifie que l'extension figure dans
la liste reconnue des formats audio et vidéo : un fichier texte renommé `.wav`
échoue à la sonde, un WAV valide renommé `.xyz` échoue au contrôle d'extension
— les deux renvoient `False` au lieu de lever une exception.
`get_audio_duration` lit la durée du premier flux audio.

```python
print(ah.is_valid_audio_file("example.mp3"))
# True

print(f"{ah.get_audio_duration('example.mp3'):.2f} s")
# 60.00 s
```

Les conteneurs vidéo sont acceptés : toutes les opérations de cette page
s'appliquent à la piste audio d'un `.mp4` comme d'un `.mkv`.

## 2. Charger un enregistrement puis en écrire un

`load_audio` renvoie `(signal, sample_rate)`. Le décodage passe toujours par
ffmpeg, jamais par libsndfile : `.m4a`, `.opus`, `.webm` ainsi que les
conteneurs vidéo se chargent comme le reste.

```python
audio, sr = ah.load_audio(
    "example.mp3",
    target_sample_rate=16000,  # rééchantillonne à la sortie
    to_mono=True,  # ramène en mono (c'est le défaut)
    to_numpy=True,  # rend un tableau numpy
)
print(audio.shape, sr, audio.dtype)
# (960000,) 16000 float32
```

**Attention au défaut.** Par défaut, `to_numpy=False` renvoie un tenseur
*torch* : l'appel nu `ah.load_audio(chemin)` réclame donc l'extra `[demucs]`,
faute de quoi il lève une `ImportError`.

```python
tensor, native_sr = ah.load_audio("example.mp3")
print(type(tensor).__name__, tuple(tensor.shape), native_sr)
# Tensor (2880000,) 48000
```

En conservant les deux canaux, numpy donne `(temps, canaux)` là où torch donne
`(canaux, temps)` :

```python
stereo, sr = ah.load_audio("example.mp3", to_mono=False, two_channels=True, to_numpy=True)
print(stereo.shape, sr)
# (2880000, 2) 48000
```

`save_audio` écrit un tableau numpy ou un tenseur torch ; l'extension choisit
le conteneur ainsi que le codec.

```python
beep = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
ah.save_audio(beep, "beep.wav", sample_rate=16000)
print(f"{ah.get_audio_duration('beep.wav'):.2f} s")
# 1.00 s
```

## 3. Convertir format, fréquence et canaux

`sound_converter` est l'appel ffmpeg unique qui couvre le conteneur, la
fréquence d'échantillonnage, le nombre de canaux ainsi que le codec. Il renvoie
le chemin de sortie et, contrairement à la plupart de ses voisines, il part sur
`overwrite=True`.

```python
print(
    ah.sound_converter(
        "example.mp3", "speech.wav", freq=16000, channels=1, encoding="pcm_s16le", overwrite=True
    )
)
# speech.wav
```

Le rééchantillonnage est le point qui mérite l'attention : il a lieu dans
`libswresample` de ffmpeg, donc un sous-échantillonnage est d'abord limité en
bande et ne replie pas le spectre. C'est le même chemin que celui de
`load_audio(target_sample_rate=...)`.

## 4. Extraire un passage dans un long enregistrement

`extract_audio_chunk(audio_file, start_time, end_time, output_audio_filename)`
découpe une tranche bornée dans le temps — en secondes — puis valide ce
qu'elle a écrit. Des bornes hors du fichier lèvent une `AssertionError`.

```python
print(ah.extract_audio_chunk("example.mp3", 10.0, 25.0, "highlight.mp3", overwrite=True))
# highlight.mp3

print(f"{ah.get_audio_duration('highlight.mp3'):.2f} s")
# 15.00 s
```

Sans chemin de sortie, la tranche se pose à côté de son entrée, nommée d'après
les bornes **en millisecondes** — `example.mp3` coupé sur `[0, 2]` devient
`example_chunk-0-2000.mp3` — et le chemin renvoyé est absolu.

## 5. Découper en fenêtres fixes puis recoller

`split_audio_regularly` est le découpeur pour l'inférence par fenêtres. C'est
aussi l'une des deux seules fonctions de cette page qui crée son dossier de
sortie.

```python
chunks = ah.split_audio_regularly(
    "example.mp3", "windows", split_time=25.0, output_format="mp3", overwrite=True
)
print(len(chunks))
# 3

print([os.path.basename(c) for c in chunks])
# ['chunk_0000_split.mp3', 'chunk_0001_split.mp3', 'chunk_0002_split.mp3']

print([round(ah.get_audio_duration(c), 2) for c in chunks])
# [25.0, 25.0, 10.0]
```

La dernière fenêtre vaut ce qui reste : 60 s par tranches de 25 s donnent
25 + 25 + 10, pas trois fenêtres complétées par du vide. Les chemins renvoyés
sont absolus ; le `_split` du nom vient de l'argument `suffix=`.

`audio_concatenation` assemble les fichiers bout à bout, dans l'ordre fourni ;
l'extension de sortie détermine le conteneur.

```python
joined = ah.audio_concatenation(chunks, "rejoined.mp3", overwrite=True)
print(joined, f"{ah.get_audio_duration(joined):.2f} s")
# rejoined.mp3 60.00 s
```

## 6. Combler avec du silence et masquer les coupes

`generate_silent_audio` écrit un silence numérique, pour compléter un montage
jusqu'à une durée fixe ou pour espacer deux prises.

```python
pad = ah.generate_silent_audio(
    duration=5.0, output_audio_filename="pad.wav", sample_rate=44100, overwrite=True
)
print(pad, f"{ah.get_audio_duration(pad):.2f} s")
# pad.wav 5.00 s
```

Or c'est précisément le silence numérique entre deux coupes qui rend un montage
audible : d'où `mix_room_tone`, qui pose une ambiance constante une quarantaine
de décibels sous la parole, assez pour masquer la jointure tout en restant sous
le seuil d'écoute conscient.

```python
rt = ah.mix_room_tone(
    "speech.wav", "speech-roomtone.wav", noise_db=-42.0, color="pink", overwrite=True
)
print(rt, f"{ah.get_audio_duration(rt):.2f} s")
# speech-roomtone.wav 60.00 s
```

`color` reprend la palette `anoisesrc` de ffmpeg : `white`, `pink`, `brown`
(alias `red`), `blue`, `violet`, `velvet`. Le rose à −42 dB est le défaut ; le
`brown` à −38 dB sonne plus chaud et plus présent. Sans chemin de sortie, le
résultat se pose à côté de l'entrée sous `<nom>-roomtone.<ext>` : deux appels
sur la même entrée écrivent donc au même endroit.

## 7. Séparer un mixage en voix, batterie, basse et autres

`separate_sources` fait tourner **Hybrid Demucs**, atteint via le pipeline
`HDEMUCS_HIGH_MUSDB_PLUS` de torchaudio plutôt que via le paquet `demucs`
autonome. Il réclame l'extra `[demucs]` ; le premier appel télécharge environ
335 Mo de poids depuis `download.pytorch.org` vers le cache PyTorch Hub.
**C'est la seule recette de cette page qui n'a pas été exécutée** : elle est
vérifiée sur le code seul, raison pour laquelle aucune sortie n'est citée
ci-dessous.

```python
sources = ah.separate_sources(
    "song.mp3",
    output_folder="stems",
    device="cpu",  # ou "cuda" ; None laisse le choix se faire tout seul
    nb_workers=4,  # nombre de fils, ignoré sauf si device == "cpu"
    output_format="mp3",
    overwrite=True,
)

for stem, path in sources.items():
    print(stem, path)
```

Ce que le code garantit :

- Le dictionnaire renvoyé est clé par `drums`, `bass`, `other`, `vocals`, dans
  cet ordre — celui dans lequel le bundle torchaudio déclare ses sources, ni
  alphabétique ni voix d'abord.
- Chaque fichier vaut `<output_folder>/<stem>.<output_format>`. Avec
  `output_folder=None`, les pistes se posent dans le dossier de l'entrée.
- Chaque piste est écrite en **mono**. Le modèle sépare en stéréo ; l'écriture
  ramène en mono. S'il vous faut les pistes stéréo, pilotez `demucs` amont
  vous-même.
- Sans `torch` installé, l'appel lève une `ImportError` portant l'indication
  d'installation, avant d'avoir touché le moindre fichier.
- Cela marche sur la parole autant que sur la musique — c'est là l'astuce
  utile : sur un entretien bruité, `vocals` est une piste de voix débruitée.

## 8. Mesurer la ressemblance entre deux prises

`sound_resemblance(a, b)` rééchantillonne les deux fichiers en 24 kHz mono,
complète le plus court par des zéros, calcule les MFCC puis renvoie la
similarité cosinus des deux matrices, dans `[0, 1]`.

```python
print(f"{ah.sound_resemblance('speech.wav', 'speech-roomtone.wav'):.3f}")
# 0.978

print(f"{ah.sound_resemblance('speech.wav', 'speech.wav'):.3f}")
# 1.000
```

Le bruit de fond est un bruit aléatoire regénéré à chaque passe : le premier
chiffre bouge donc à partir de la quatrième décimale d'une exécution à
l'autre. À trois décimales, il est resté stable sur six essais.

À lire comme un **instrument de sensibilité entre prises quasi identiques**,
pas comme un classifieur pareil/différent. La même mesure sur des sons
franchement étrangers reste haute :

```python
print(f"{ah.sound_resemblance('speech.wav', 'beep.wav'):.3f}")
# 0.895

print(f"{ah.sound_resemblance('speech.wav', 'pad.wav'):.3f}")
# 0.912
```

Un sinus à 440 Hz obtient 0,895 contre de la parole ; cinq secondes de
silence numérique obtiennent 0,912 — davantage que le sinus. La raison est dans
l'implémentation : la similarité cosinus porte sur les matrices MFCC brutes,
sans retrait de la moyenne, si bien que le coefficient 0 — l'énergie
logarithmique globale — domine chaque comparaison. Servez-vous du score pour
confirmer qu'un réencodage, un passage de bruit de fond ou un aller-retour n'a
pas abîmé un fichier. Ne vous en servez pas pour décider si deux fichiers sont
le même enregistrement.

## 9. Construire ses propres descripteurs Mel et MFCC

Les primitives fondées sur scipy vivent dans `audio_helper.main` et ne sont pas
réexportées à la racine du paquet : il faut les importer explicitement.

```python
from audio_helper.main import hz_to_mel, mel_to_hz, mel_filter_banks, mfcc

print(hz_to_mel(440.0), mel_to_hz(549.6386753811499))
# 549.6386753811499 440.0

fb = mel_filter_banks(num_filters=26, n_fft=512, sample_rate=16000, low_freq=0, high_freq=8000)
print(fb.shape)
# (26, 257)

# `audio` est le tableau mono 16 kHz de la recette 2 — 60 s de parole.
coefs = mfcc(audio, 16000, num_mfcc=13, n_fft=512)
print(coefs.shape)
# (5999, 13)
```

Le fenêtrage est figé à 25 ms de fenêtre pour 10 ms de pas, d'où le nombre de
trames ; la chaîne, elle, est celle des manuels : pré-accentuation à 0,97, fenêtre
de Hamming, spectre de puissance, banc de filtres Mel, décibels, DCT-II, puis
on garde les `num_mfcc` premiers coefficients.

## Ligne de commande

Une seule ligne de commande, en click, installée avec le paquet — aucun extra
requis. Les options sont nommées plutôt que positionnelles, parce que ces
commandes prennent plusieurs chemins et qu'un ordre nu s'inverse vite.

```bash
$ audio-helper --help          # -h marche aussi ; -V affiche la version
$ audio-helper duration    --input example.mp3
60.000000

$ audio-helper convert     --input example.mp3 --output out.wav --freq 16000 --channels 1
out.wav

$ audio-helper chunk       --input example.mp3 --start 3.0 --end 8.5 --output cut.mp3
cut.mp3

$ audio-helper split       --input example.mp3 --output-dir chunks/ --seconds 20 --overwrite
/chemin/vers/chunks/chunk_0000_split.mp3
/chemin/vers/chunks/chunk_0001_split.mp3
/chemin/vers/chunks/chunk_0002_split.mp3

$ audio-helper concat      --inputs chunks/chunk_0000_split.mp3 chunks/chunk_0001_split.mp3 \
                           --output all.mp3 --overwrite
all.mp3

$ audio-helper roomtone    --input out.wav --output out-rt.wav --db -42 --color pink --overwrite
out-rt.wav

$ audio-helper silence     --duration 2 --output pad.wav --overwrite
pad.wav

$ audio-helper resemblance --a out.wav --b cut.mp3
0.877206

$ audio-helper separate    --input mix.mp3 --output-dir stems/   # nécessite [demucs]
```

Les commandes à fichier unique affichent le chemin, `split` affiche un chemin
absolu par ligne, `separate` affiche du JSON : chacune se branche donc sur la
suivante. `concat` accepte les deux écritures, la liste ci-dessus ou le drapeau
répété sous la forme `--inputs a.mp3 --inputs b.mp3`. `separate` est la seule
sous-commande non exécutée pour cette page, pour la même raison de
téléchargement de poids que la recette 7 ; sans `torch`, elle sort en code 2
avec une ligne `error:` au lieu d'une trace d'appels.

Les `--overwrite` ci-dessus ne sont pas du décor : `split`, `concat`, `silence`
ainsi que `roomtone` partent sur `--no-overwrite`, tandis que `convert` et
`chunk` partent sur `--overwrite`.

## API HTTP, MCP et interface web

Les mêmes fonctions, en HTTP, derrière l'extra `[api]` :

```bash
pip install "audio-helper[api]"
uvicorn audio_helper.api:app --port 8000
```

```bash
$ curl -s http://localhost:8000/health
{"status":"ok"}

$ curl -s -F 'file=@example.mp3' http://localhost:8000/duration
{"duration_seconds":60.0}

# Binaire en entrée, binaire en sortie : le fichier converti est le corps de
# la réponse.
$ curl -s -F 'file=@example.mp3' -F 'output_format=wav' -F 'freq=16000' \
       -o out.wav http://localhost:8000/convert
$ ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
          -of default=noprint_wrappers=1 out.wav
codec_name=pcm_s16le
sample_rate=16000
channels=1
```

Chaque opération de la bibliothèque a sa route : `/convert`, `/duration`,
`/chunk`, `/silence`, `/concat`, `/roomtone`, `/split` (qui renvoie une archive
zip des morceaux), `/separate`, `/resemblance`. Les envois sont en multipart ;
la documentation OpenAPI est sur `/docs`.

Deux surfaces de plus se posent sur cette même application. `GET /gui` sert le
**Recipe Canvas** — enchaîner les verbes en pipeline, écouter chaque étape
intermédiaire, exporter le résultat en `recipe.yaml` versionnable ; voir
[GUI.md](https://github.com/warith-harchaoui/audio-helper/blob/main/GUI.md).
Quant à `audio-helper-mcp` (extra `[mcp]`), il republie les routes de
l'application en outils MCP pour un hôte agentique :

```bash
pip install "audio-helper[mcp]"
audio-helper-mcp
# même application, plus un endpoint /mcp
```

## Dépannage

**`FileNotFoundError` ou `ffmpeg.Error` à l'écriture.** Le dossier de
destination n'existe pas. `split_audio_regularly` ainsi que `separate_sources`
créent le leur ; `sound_converter`, `extract_audio_chunk`,
`generate_silent_audio`, `audio_concatenation`, `mix_room_tone` et
`save_audio` le supposent tous déjà là. Appelez `osh.make_directory(...)`
d'abord.

**`ImportError: This feature requires the optional 'demucs' extra.`** Soit
`separate_sources`, soit `load_audio` sans `to_numpy=True` : le type de retour
par défaut est un tenseur torch. Ajoutez `to_numpy=True` si vous ne vouliez
qu'un tableau.

**Rien ne s'est passé et le fichier n'a pas bougé.** `overwrite=False` ne veut
pas dire « n'écris jamais », mais « saute si c'est à jour » : la sortie n'est
réutilisée que tant qu'elle est plus récente que chacune de ses entrées.
Touchez une entrée et l'appel suivant la reconstruit. Pour refaire le travail
sans condition, passez `overwrite=True`.

**ffmpeg est bavard — ou muet quand vous voulez le voir.** Les deux relèvent de
`osh.verbosity()`. À `0` (le défaut, WARNING) la bibliothèque lance ffmpeg avec
`quiet=True` ; à `1` ou plus, vous récupérez les lignes INFO d'audio-helper ; ffmpeg
cesse d'être muselé.

**`ffmpeg is required to decode audio but was not found on PATH`.** Le message
est littéral. `brew install ffmpeg` ou `apt install ffmpeg`, puis réessayez.

Voir [TRIGGERS.md](https://github.com/warith-harchaoui/audio-helper/blob/main/TRIGGERS.md)
pour le catalogue complet des formulations, commandes et types de fichiers
reconnus.
