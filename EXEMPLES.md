# Exemples pratiques d'audio-helper

Recettes pratiques pour la surface publique de `audio-helper`. Chaque extrait
suppose :

```python
import audio_helper as ah
import os_helper as osh
```

et que `ffmpeg` est installé et accessible dans le `PATH`. L'extra optionnel
`demucs` (Torch et torchaudio) n'est nécessaire que pour la
[séparation des sources](#séparation-des-sources).

---

## Sommaire

1. [Installation](#installation)
2. [Validité et durée](#validité-et-durée)
3. [Charger et enregistrer](#charger-et-enregistrer)
4. [Convertir les formats](#convertir-les-formats)
5. [Découpage](#découpage)
   - [Extraire un segment](#extraire-un-segment)
   - [Découper à intervalles réguliers](#découper-à-intervalles-réguliers)
   - [Concaténer](#concaténer)
6. [Silence et bruit de fond](#silence-et-bruit-de-fond)
   - [Générer un silence](#générer-un-silence)
   - [Ajouter un bruit de fond](#ajouter-un-bruit-de-fond)
7. [Séparation des sources](#séparation-des-sources)
8. [Similarité](#similarité)
9. [Extraction de caractéristiques (Mel / MFCC)](#extraction-de-caractéristiques-mel--mfcc)

---

## Installation

Installer avec pip (remplacer le tag par la version voulue) :

```bash
pip install --force-reinstall --no-cache-dir \
    audio-helper
```

Pour activer la séparation de sources fondée sur Demucs :

```bash
pip install --force-reinstall --no-cache-dir \
    "audio-helper[demucs]"
```

## Validité et durée

`is_valid_audio_file` lance `ffprobe` et vérifie que l'extension du fichier
figure dans la liste reconnue des formats audio et vidéo. `get_audio_duration`
lit la durée à partir du premier flux audio.

```python
if ah.is_valid_audio_file("interview.mp3"):
    seconds = ah.get_audio_duration("interview.mp3")
    print(f"Interview is {seconds:.2f}s long.")
    # Interview is 1834.27s long.
```

Un fichier texte portant l'extension `.wav` échoue à la sonde et renvoie
`False` ; un WAV valide renommé en `.xyz` échoue au contrôle d'extension et
renvoie `False` lui aussi.

## Charger et enregistrer

`load_audio` renvoie `(signal, sample_rate)`, où `signal` est par défaut un
tableau numpy. Utiliser `to_mono=True` pour obtenir du mono, `two_channels=True`
pour préserver le stéréo et `target_sample_rate` pour rééchantillonner à la
volée.

```python
audio, sr = ah.load_audio(
    "recording.wav",
    target_sample_rate=16000,   # rééchantillonne à 16 kHz
    to_mono=True,               # ramène en mono
    to_numpy=True,              # renvoie un numpy.ndarray (par défaut sans torch installé)
)
print(audio.shape, sr)          # (n_samples,) 16000
```

`save_audio` écrit un tableau numpy (ou un tenseur torch) sur le disque ;
l'extension détermine le conteneur et le codec.

```python
import numpy as np
sr = 16000
signal = (0.1 * np.random.randn(sr)).astype(np.float32)  # 1 s de bruit
ah.save_audio(signal, "noise.wav", sample_rate=sr)
```

## Convertir les formats

`sound_converter` est une enveloppe autour de ffmpeg qui couvre en un seul
appel le format, la fréquence d'échantillonnage et le nombre de canaux.

```python
ah.sound_converter(
    "speech.m4a",
    "speech.wav",
    freq=44100,
    channels=1,
    encoding="pcm_s16le",
    overwrite=True,
)
```

## Découpage

### Extraire un segment

`extract_audio_chunk(input, start_s, end_s, output, overwrite=...)` découpe
une tranche bornée dans le temps et valide le fichier produit. Des bornes
hors plage lèvent une `AssertionError`.

```python
ah.extract_audio_chunk("podcast.mp3", 60.0, 75.0, "highlight.mp3", overwrite=True)
```

### Découper à intervalles réguliers

`split_audio_regularly(source, output_folder, split_time_s, output_format=...)`
tranche la source en segments de durée fixe et renvoie la liste des chemins
produits.

```python
chunks = ah.split_audio_regularly(
    "lecture.mp3",
    "lecture-chunks",
    split_time=30.0,
    output_format="mp3",
    overwrite=True,
)
print(f"{len(chunks)} chunks written.")
# 42 chunks written.
```

### Concaténer

`audio_concatenation([files], output, overwrite=...)` assemble plusieurs
fichiers bout à bout. Le conteneur est déterminé par l'extension du fichier
de sortie.

```python
ah.audio_concatenation(
    ["intro.wav", "body.wav", "outro.wav"],
    "episode.mp3",
    overwrite=True,
)
```

## Silence et bruit de fond

### Générer un silence

```python
ah.generate_silent_audio(
    duration=5.0,
    output_audio_filename="pad.wav",
    sample_rate=44100,
    overwrite=True,
)
```

### Ajouter un bruit de fond

Superpose un bruit ambiant discret et constant, le bruit de fond de pièce
(« room tone »), sur une piste de parole : les silences entre deux coupes ne
sonnent plus creux. Le réglage par défaut est un bruit rose à −42 dB,
inaudible mais présent.

```python
ah.mix_room_tone(
    "narration.wav",
    "narration-rt.wav",
    noise_db=-42.0,
    color="pink",              # white / pink / brown / blue / violet / velvet
    overwrite=True,
)
```

Utiliser `color="brown"` et `noise_db=-38` pour un bruit de fond plus chaud
et plus présent.

## Séparation des sources

`separate_sources` lance Demucs pour produire les pistes voix, batterie,
basse et autres. Nécessite l'extra `[demucs]` (Torch et torchaudio).

```python
sources = ah.separate_sources(
    "song.mp3",
    output_folder="stems",
    device="cuda",        # ou "cpu" ; passer None laisse le choix se faire automatiquement
    nb_workers=4,         # ignoré si device != "cpu"
    output_format="mp3",
    overwrite=True,
)
print(sources)
# {'vocals': 'stems/vocals.mp3', 'drums': 'stems/drums.mp3',
#  'bass': 'stems/bass.mp3',   'other': 'stems/other.mp3'}
```

Si Torch n'est pas installé, l'appel lève une `ImportError` accompagnée de
l'indication d'installation.

## Similarité

`sound_resemblance(a, b)` renvoie un score dans `[0, 1]` fondé sur la
similarité des MFCC. Un fichier comparé à lui-même approche 1,0 ; deux sons
sans rapport font chuter le score nettement.

```python
score = ah.sound_resemblance("original.wav", "reconstructed.mp3")
print(f"resemblance = {score:.3f}")
# resemblance = 0.974
```

## Extraction de caractéristiques (Mel / MFCC)

Fonctions bas niveau, fondées sur scipy, pour construire son propre pipeline
de caractéristiques.

```python
import numpy as np
from audio_helper.main import hz_to_mel, mel_to_hz, mel_filter_banks, mfcc

# Hz <-> Mel
print(hz_to_mel(440.0), mel_to_hz(549.6))
# 549.6386500664797 440.00057651...

# Banc de filtres Mel
sample_rate = 16000
fb = mel_filter_banks(num_filters=26, n_fft=512,
                      sample_rate=sample_rate, low_freq=0,
                      high_freq=sample_rate // 2)
print(fb.shape)   # (26, 257)

# MFCC à partir d'un signal 1D brut
signal = np.random.randn(sample_rate).astype(np.float32)
coefs = mfcc(signal, sample_rate, num_mfcc=13, n_fft=512)
print(coefs.shape)  # (n_frames, 13)
```

---

## Surfaces (CLI / API / GUI)

Les mêmes opérations sont accessibles sans écrire une ligne de Python.

**CLI argparse (toujours installée) :**

```bash
audio-helper convert --input in.mp3 --output out.wav --freq 16000 --channels 1
audio-helper chunk   --input in.mp3 --start 3.0 --end 8.5 --output cut.mp3
audio-helper split   --input in.mp3 --output-dir chunks/ --seconds 30
audio-helper concat  --inputs a.mp3 b.mp3 c.mp3 --output all.mp3
audio-helper roomtone --input speech.wav --output speech-rt.wav --db -42 --color pink
audio-helper separate --input mix.mp3 --output-dir stems/          # nécessite [demucs]
audio-helper resemblance --a take1.wav --b take2.wav
```

**Jumelle CLI click (extra `[cli]`), mêmes options :**

```bash
pip install "audio-helper[cli]"
audio-helper-click convert --input in.mp3 --output out.wav --freq 16000
```

**Surface HTTP FastAPI et GUI (extra `[api]`) :**

```bash
pip install "audio-helper[api]"
uvicorn audio_helper.api:app --port 8000

# Conversion par HTTP (envoi multipart) :
curl -F 'file=@in.mp3' -F 'output_format=wav' -F 'freq=16000' \
     -o out.wav http://localhost:8000/convert

# Documentation OpenAPI :  http://localhost:8000/docs
# GUI dans le navigateur : http://localhost:8000/gui   (déposer un fichier, choisir une opération, comparer A/B)
```

Voir [TRIGGERS.md](TRIGGERS.md) pour le catalogue complet des formulations,
commandes et types de fichiers reconnus.
