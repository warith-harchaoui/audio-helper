"""
Audio Helper

This module provides a set of functions to process audio files using PyTorch, torchaudio, ffmpeg, and various helper functions.
The main functionalities include loading, converting, and separating audio sources, as well as splitting and concatenating audio files.

Dependencies
------------
- torch
- torchaudio
- ffmpeg-python
- tqdm
- numpy
- soundfile
- scipy
- os-helper

Authors:
------------
- [Warith Harchaoui](https://harchaoui.org/warith)
- [Mohamed Chelali](https://mchelali.github.io)
- [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug)

"""

from typing import List, Union
import torch
import os_helper as osh
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
import ffmpeg
from tqdm import tqdm
import concurrent.futures
import numpy as np
import soundfile as sf
from scipy.signal import resample
import scipy.io.wavfile as wav
from scipy.signal import correlate
import warnings
from scipy.fftpack import dct
from scipy.signal import get_window
from scipy.fftpack import fft
from typing import Optional

import logging

audio_extensions = [
    "aif",
    "aiff",
    "alac",
    "amr",
    "ape",
    "au",
    "flac",
    "gsm",
    "iff",
    "m4a",
    "m4b",
    "m4p",
    "mp3",
    "ogg",
    "oga",
    "opus",
    "ra",
    "ram",
    "raw",
    "sln",
    "tta",
    "voc",
    "vox",
    "wav",
    "wma",
    "wv",
    "webm",
    "rmi",
]


video_extensions = [
    "mp4",
    "avi",
    "mov",
    "wmv",
    "flv",
    "mkv",
    "webm",
    "mpeg",
    "mpg",
    "m4v",
    "3gp",
    "ogv",
    "mxf",
    "ts",
    "vob",
    "m2ts",
    "mts",
    "rm",
    "asf"
]

audio_extensions += video_extensions

def _overwrite_audio_file(output_audio_filename: str, overwrite: bool = True) -> Union[str, None]:
    """
    Check if the output audio file already exists and handle based on the overwrite flag.

    Parameters
    ----------
    output_audio_filename : str
        Path to the output audio file.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists (default is False).

    Returns
    -------
    str or None
        The output audio file path if it needs to be overwritten, None otherwise.

    Notes
    -----
    The function checks if the output audio file already exists and handles it based on the overwrite flag.
    If the file exists and overwrite is False, the function checks if the file is a valid audio file.
    If the file exists and overwrite is True, the function deletes the existing file to overwrite it.
    """

    # Check if the file already exists and handle based on the overwrite flag
    if not(overwrite) and osh.file_exists(output_audio_filename):
        logging.info(f"Output audio file already exists:\n\t{output_audio_filename}")
        if is_valid_audio_file(output_audio_filename):
            return output_audio_filename
        else:
            osh.remove_files([output_audio_filename])  # Remove invalid file
            logging.info(f"Deleting invalid output audio file:\n\t{output_audio_filename}")
    elif overwrite and osh.file_exists(output_audio_filename):
        osh.remove_files([output_audio_filename])  # Overwrite existing file
        logging.info(f"Deleting output audio file for overwrite:\n\t{output_audio_filename}")

    return None

def _overwrite_audio_list(output_audio_list: List[str], overwrite: bool = True) -> Union[str, None]:
    """
    Check if the output audio files already exist and handle based on the overwrite flag.

    Parameters
    ----------
    output_audio_list : List[str]
        List of paths to the output audio files.
    overwrite : bool, optional
        Whether to overwrite the output files if they already exist (default is False).

    Returns
    -------
    dict or None
        Dictionary mapping source names to the paths of the separated audio files if they need to be overwritten, None otherwise.

    Notes
    ----- 
    The function checks if the output audio files already exist and handles them based on the overwrite flag.
    If each of the files exist and overwrite is False, the function checks if the files are valid audio files.
    If each of the files exist and overwrite is True, the function deletes the existing files to overwrite them.
    """
    if not(overwrite) and all([(osh.file_exists(f) and is_valid_audio_file(f)) for f in output_audio_list]):
        stem_keys = []
        stem_files = []
        for f in output_audio_list:
            _, b, _ = osh.folder_name_ext(f)
            o = osh.relative2absolute_path(f)
            stem_keys.append(b)
            stem_files.append(o)
        d = {k: v for k, v in zip(stem_keys, stem_files)}
        s = "\n\t".join([f"{k}:\t{v}" for k, v in d.items()])
        logging.info(
            f"Sources already separated for at:\n\t{s}"
        )
        return d
    elif overwrite:
        for f in output_audio_list:
            if osh.file_exists(f):
                osh.remove_files([f])
                logging.info(f"Deleting output audio file for overwrite:\n\t{f}")
    
    return None


def is_valid_audio_file(file_path: str) -> bool:
    """
    Check if the given file is a valid audio file using ffmpeg-python (ffprobe).

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    bool
        True if the file contains a valid audio stream, False otherwise.

    Notes
    -----
    The function uses ffprobe to inspect the file and determine if an audio stream is present.
    """
    global audio_extensions
    # By default, the file is considered invalid
    valid = False
    try:
        probe = ffmpeg.probe(file_path)
        audio_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
            None,
        )
        valid = audio_stream is not None
    except Exception as e:
        valid = False

    _,_,ext = osh.folder_name_ext(file_path)
    if not(ext.lower() in audio_extensions):
        valid = False

    logging.info(f"Audio file {file_path} is {'valid' if valid else 'invalid'}")
    return valid


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds using ffmpeg.

    Parameters
    ----------
    file_path : str
        Path to the audio file.

    Returns
    -------
    float
        Duration of the audio file in seconds.

    Raises
    ------
    Error
        If no audio stream is found in the file.
    """
    osh.checkfile(file_path, msg=f"Audio file not found at {file_path}")
    probe = ffmpeg.probe(file_path)
    audio_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    assert audio_stream is not None, f"No audio stream found in the file: {file_path}"
    return float(audio_stream["duration"])


def load_audio(
    file_path: str,
    target_sample_rate: int = None,
    to_mono: bool = True,
    to_numpy: bool = False,
    two_channels: bool = False,
) -> tuple[Union[torch.Tensor,np.ndarray], int]:
    """
    Load an audio file using soundfile, optionally resample, convert to mono or stereo,
    and return as a torch.Tensor (or optionally as a NumPy array).

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    target_sample_rate : int, optional
        The target sample rate to resample the audio to. Defaults to the original sample rate.
    to_mono : bool, optional
        Whether to convert the audio to mono (default is True).
    to_numpy : bool, optional
        Whether to return the audio as a NumPy array (default is False). Otherwise, returns a torch.Tensor.
    two_channels : bool, optional
        Whether to force the audio into two channels (stereo).

    Returns
    -------
    torch.Tensor or np.ndarray
        The loaded audio signal.
    int
        Sample rate of the loaded audio.
    """

    # Load the audio using soundfile thay gives (time, channels) shape
    audio, sample_rate = sf.read(file_path, always_2d=True)

    if target_sample_rate is None:
        target_sample_rate = sample_rate

    if sample_rate != target_sample_rate:
        num_samples = int(len(audio) * target_sample_rate / sample_rate)
        # scipy.signal.resample expects (time, channels) shape
        audio = resample(audio, num_samples)
        sample_rate = target_sample_rate


    # If mono is requested, average the channels
    if to_mono:
        audio = np.mean(audio, axis=1)


    if two_channels:
        if len(audio.shape) == 1:
            audio = np.vstack([audio,audio])
        elif len(audio.shape) > 2 and audio.shape[1] == 1:
            t = audio.ravel()
            audio = np.vstack([t,t])
        else:
            # Split audio into two channels by averaging left and right parts
            audio_left = np.mean(audio[:, : audio.shape[1] // 2], axis=1)
            audio_right = np.mean(audio[:, audio.shape[1] // 2 : ], axis=1)
            audio = np.vstack([audio_left,audio_right])

        audio = audio.T

    if to_numpy:
        return audio, sample_rate
    
    # Convert the audio to a torch.Tensor with (channels, time) shape
    audio = torch.from_numpy(audio.T)

    return audio, sample_rate

def sound_converter(
    input_audio: str,
    output_audio: str,
    freq: int = 44100,
    channels: int = 1,
    encoding: str = "pcm_s16le",
    overwrite: bool = True,
) -> None:
    """
    Convert an audio file to the specified format using ffmpeg-python.

    Parameters
    ----------
    input_audio : str
        Path to the input audio file.
    output_audio : str
        Path to the output audio file with the desired format extension.
    freq : int, optional
        Output sample rate in Hz (default is 44100).
    channels : int, optional
        Number of audio channels in the output (default is 1 for mono).
    encoding : str, optional
        Audio codec to use for encoding the output file (default is 'pcm_s16le').

    Returns
    -------
    str :
        Path to the output audio file.

    Raises
    ------
    Error
        If the input audio file does not exist or are not valid.

    Notes
    -----
    The conversion is handled using a temporary file structure to manage intermediate formats.
    Two intermediate WAV files are used before generating the final output audio file.
    """

    logging.info(f"Converting audio file: {input_audio} into {output_audio}")

    # Check if the input audio file exists
    osh.checkfile(input_audio, msg=f"Input audio file not found: {input_audio}")

    # Check if the input audio file is valid
    assert is_valid_audio_file(input_audio), f"Invalid audio file: {input_audio}"

    o = _overwrite_audio_file(output_audio, overwrite)
    if o is not None:
        return o
    
    _, _, ext_in = osh.folder_name_ext(input_audio)
    _, _, ext_out = osh.folder_name_ext(output_audio)

    # Get verbosity settings from the environment
    verbose = False
    quiet = not verbose

    # Use temporary files for intermediate WAV processing (for robustness)
    with osh.temporary_filename(
        suffix=".wav", mode="wb"
    ) as first_wav, osh.temporary_filename(
        suffix=".wav", mode="wb"
    ) as second_wav:

        if not(ext_in.lower() == "wav"):
            # Convert the input audio file to a temporary WAV file
            ffmpeg.input(input_audio).output(first_wav, format="wav").run(
                overwrite_output=True, quiet=quiet
            )
        else:
            osh.copyfile(input_audio, first_wav)


        # Convert the temporary WAV file to another WAV file with specified parameters
        ffmpeg.input(first_wav).output(
            second_wav, ar=freq, ac=channels, acodec=encoding
        ).run(overwrite_output=True, quiet=quiet)

        if not(ext_out.lower() == "wav"):
            # Final conversion to the specified output format
            ffmpeg.input(second_wav).output(output_audio).run(
                overwrite_output=True, quiet=quiet
            )
        else:
            osh.copyfile(second_wav, output_audio)

    # Check if the output audio file was successfully created
    osh.checkfile(
        output_audio, msg=f"Failed to convert audio file:\n\t{output_audio}"
    )
    assert is_valid_audio_file(output_audio), f"Invalid audio file:\n\t{output_audio}"

    logging.info(f"Audio file converted successfully:\n\t{output_audio}")

    return output_audio


def save_audio(signal: Union[torch.Tensor, np.ndarray], file_path: str, sample_rate: int=44100) -> None:
    """
    save_audio saves a torch.Tensor or a NumPy array as an audio file using torchaudio or scipy.

    Parameters
    ----------
    signal : torch.Tensor or np.ndarray
        The audio signal to save.
    file_path : str
        Path to the output audio file.
    sample_rate : int
        The sample rate of the audio signal.

    Raises
    ------
    Error
        If the audio signal is not a torch.Tensor or a NumPy array.

    """    
    if isinstance(signal, torch.Tensor): # (channels, time) convention

        signal = signal.detach().cpu().numpy()
        if len(signal.shape) == 1:
            signal = signal.reshape(1,-1) # (1, time) convention

        signal = signal.T # transpose to the (time, channels) convention
        save_audio(signal, sample_rate, file_path)

    elif isinstance(signal, np.ndarray): # (time, channels) convention
        _,_,ext = osh.folder_name_ext(file_path)
        if ext.lower() == "wav":
            wav.write(file_path, sample_rate, signal)
        else:
            with osh.temporary_filename(suffix=".wav", mode="wb") as wav_audio_file:
                wav.write(wav_audio_file, sample_rate, signal)
                channels = 1 if len(signal.shape)==1 else signal.shape[1]
                sound_converter(input_audio = wav_audio_file, output_audio=file_path, freq=sample_rate, channels=channels, encoding="pcm_s16le", overwrite=True)

        assert is_valid_audio_file(file_path), f"Audio file not saved to {file_path}"
        logging.info(f"Audio signal saved to {file_path}")



def _separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    device: str = None,
    nb_workers: int = 2,
) -> torch.Tensor:
    """
    Apply a source separation model to a given audio mixture, processing the mixture in segments with overlap and fades,
    using multithreading to parallelize segment processing.

    Parameters
    ----------
    model : torch.nn.Module
        The pre-trained source separation model to apply to the audio mixture.
    mix : torch.Tensor
        The audio mixture tensor with shape (batch_size, channels, length).
    sample_rate : int
        The sample rate of the audio mixture.
    segment : float, optional
        The length of each segment in seconds to process (default is 10.0 seconds).
    overlap : float, optional
        The overlap duration between consecutive segments in seconds (default is 0.1 seconds).
    device : str, optional
        The device on which to run the computations.
    nb_workers : int, optional
        The number of threads to use for parallel processing (default is 2).

    Returns
    -------
    torch.Tensor
        Tensor containing the separated sources, with shape (batch_size, num_sources, channels, length).


    Notes
    -----
    The function processes the audio mixture in overlapping segments, applies the source separation model to each segment
    in parallel using multithreading, and uses linear fades to smooth transitions between overlapping segments. The
    separated sources are then reassembled into the final output tensor.
    """

    # Do not use all cores and leave one for the system!

    # Get the number of workers from osh if nb_workers is not provided
    if nb_workers is None:
        nb_workers = osh.get_nb_workers()

    # Adjust workers count if nb_workers is negative (relative to the system like sklearn convention)
    if nb_workers < 0:
        nb_workers = osh.get_nb_workers() - nb_workers  + 1

    # Limit the number of workers to the maximum available minus one for the system
    MAX_NB_WORKERS = osh.get_nb_workers()
    if nb_workers >= MAX_NB_WORKERS:
        nb_workers = MAX_NB_WORKERS - 1

    # Check if cuda is available
    if device is None:
        # check pytorch device
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # If CUDA, limit the number of workers to 1 to avoid CUDA out-of-memory errors
    if device == "cuda" or device == torch.device("cuda"):
        nb_workers = 1

    # Convert the device to a torch.device object if it is a string
    if isinstance(device, str):
        device = torch.device(device)

    # Move the audio mixture and the model to the specified device
    mix.to(device)
    mix = mix.float() # Convert mix to float32

    model.to(device)

    # Get the batch size, number of channels, and length of the audio mixture
    batch, channels, length = mix.shape

    # Calculate the length of each chunk (segment) in frames, accounting for overlap
    chunk_len = int(sample_rate * segment * (1 + overlap))

    # Calculate the number of overlap frames
    overlap_frames = int(overlap * sample_rate)

    # Create a Fade transformation to apply linear fades between segments
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

    # Initialize a tensor to store the final separated sources
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    # Calculate the total number of chunks to process
    total_chunks = (length - overlap_frames) // (chunk_len - overlap_frames) + 1

    # Define the function to process each chunk in parallel
    def process_chunk(start: int, end: int):
        """
        Function to process a single chunk of the audio mixture.

        Parameters
        ----------
        start : int
            Start index of the chunk.
        end : int
            End index of the chunk.

        Returns
        -------
        tuple
            (start, end, torch.Tensor) - The start and end indices and the processed output.
        """
        if end > length:
            end = length
            fade.fade_out_len = 0  # Disable fade out for the last chunk

        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)

        return start, end, out

    # Use ThreadPoolExecutor to process the chunks in parallel
    if nb_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=nb_workers) as executor:
            futures = []
            start = 0
            end = chunk_len

            # Submit each chunk processing task to the executor
            for _ in range(total_chunks):
                futures.append(executor.submit(process_chunk, start, end))
                start += chunk_len - overlap_frames
                end = start + chunk_len

            # Collect results and assemble the final output
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Processing chunks",
                total=total_chunks,
            ):
                start, end, out = future.result()
                final[:, :, :, start:end] += out
    else:
        for i in tqdm(range(total_chunks), desc="Processing chunks for source separation", total=total_chunks):
            start = i * (chunk_len - overlap_frames)
            end = start + chunk_len
            start, end, out = process_chunk(start, end)
            final[:, :, :, start:end] += out

    return final


separator_engine = None
separator_engine_sample_rate = None


def separate_sources(
    input_audio_file: str,
    output_folder: str = None,
    device: str = None,
    overwrite: bool = False,
    nb_workers: int = -2,
    output_format: str = "mp3",
) -> dict:
    """
    Separate an input audio file into different sources (e.g., vocals, bass, drums, other) using a pre-trained model from pytorch called DEMUCS.

    Parameters
    ----------
    input_audio_file : str
        Path to the input audio file.
    output_folder : str, optional
        Folder to save the separated sources. If None, the output folder will be created based on the input file's name.
    device : str, optional
        The device on which to run the computations. If not specified, CUDA will be used if available, otherwise CPU.
    overwrite : bool, optional
        Whether to overwrite existing files if they already exist (default is False).
    nb_workers : int, optional
        The number of workers (threads) to use for parallel processing of segments (default is -2 which corresponds to all cores except one).
    output_format : str, optional
        The format of the output audio files (default is 'mp3').
    
    Returns
    -------
    dict
        A dictionary mapping source names (e.g., 'vocals', 'bass', etc.) to the paths of the separated audio files.

    Examples
    --------
    >>> separated_sources = separate_sources("input_audio.mp3", output_folder="output_folder", overwrite=True)
    >>> print(separated_sources)
    {'vocals': 'output_folder/vocals.mp3', 'drums': 'output_folder/drums.mp3', 'bass': 'output_folder/bass.mp3', 'other': 'output_folder/other.mp3'}


    Notes
    -----
    The function uses the HDEMUCS_HIGH_MUSDB_PLUS model to separate audio into its constituent sources. It processes
    the audio in segments with optional multithreading for parallel processing. The separated sources are saved as
    audio files in the specified or generated output folder.
    """

    global separator_engine, separator_engine_sample_rate
    logging.info(f"Separating sources for:\n\t{input_audio_file}")

    # Set up the output folder if not specified
    if output_folder is None:
        f, _, _ = osh.folder_name_ext(input_audio_file)
        output_folder = f

    # Check if files already exist and skip if not overwriting
    stem_keys = ["vocals", "drums", "bass", "other"]
    stem_files = [
        osh.os_path_constructor([output_folder, f"{stem}.{output_format}"])
        for stem in stem_keys
    ]
    d = _overwrite_audio_list(stem_files, overwrite)
    if d is not None:
        return d

    # Initialize the separator engine if it hasn't been initialized yet
    if separator_engine is None:
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        separator_engine = bundle.get_model()
        separator_engine_sample_rate = bundle.sample_rate

    two_channels = True

    # Load the audio file and resample it if needed
    waveform, sample_rate = load_audio(
        input_audio_file,
        target_sample_rate=separator_engine_sample_rate,
        to_numpy=False,
        two_channels=two_channels,
    )
    waveform = waveform.reshape(1, 2 if two_channels else 1, -1)

    # Normalize the audio signal
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    # Separate the audio into sources using the multithreaded _separate_sources function
    sources = _separate_sources(
        separator_engine,
        waveform,
        separator_engine_sample_rate,
        device=device,
        segment=10,
        overlap=0.1,
        nb_workers=nb_workers,
    )[0]

    # Denormalize the separated sources
    sources = sources * ref.std() + ref.mean()

    # Get the list of sources from the model and process each source
    sources_list = separator_engine.sources
    sources = list(sources)

    # Dictionary to store the output file paths for each source
    res = {}
    for stem in sources_list:
        audio = sources.pop(0) # in (channels, time) shape
        # convert it in scipy (time, channels) shape
        audio = audio.detach().cpu().numpy().T
        # reduce to mono which means channels = 1
        audio = np.mean(audio, axis=1)
        # check sample rate
        if sample_rate != separator_engine_sample_rate:        
            resampler = torchaudio.transforms.Resample(
                orig_freq=separator_engine_sample_rate, new_freq=sample_rate
            )
            audio = resampler(audio)

        osh.make_directory(output_folder)
        output_audio_file = osh.os_path_constructor(
                            [output_folder, f"{stem}.{output_format}"]
                        )
        save_audio(audio, output_audio_file, sample_rate)
        res[stem] = output_audio_file

        logging.info(f"Saved {stem} to\n\t{output_audio_file}")
        


    return res


def extract_audio_chunk(
    audio_file: str,
    start_time: float,
    end_time: float,
    output_audio_filename: str = None,
    overwrite: bool = True,
) -> str:
    """
    Extract a chunk of audio from an audio file between specified start and end times.

    Parameters
    ----------
    audio_file : str
        Path to the input audio file.
    start_time : float
        Start time in seconds for the chunk to extract.
    end_time : float
        End time in seconds for the chunk to extract.
    output_audio_filename : str, optional
        Path to save the extracted audio chunk. If None, the output file will be saved in the same directory
        as the input file with a filename based on the chunk's time range.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists.

    Returns
    -------
    str
        The path to the extracted audio chunk file.

    Raises
    ------
    Error
        If the input audio file is not found or not valid or intervals are absurd.

    Notes
    -----
    This function uses ffmpeg to extract a specific portion of the audio file between `start_time` and `end_time`.
    It verifies that the input file is a valid audio file and checks that the specified time range is within the audio's duration.
    """

    # Check if the input audio file exists and is valid
    osh.checkfile(audio_file, msg=f"Audio file not found at:\n\t{audio_file}")
    assert is_valid_audio_file(audio_file), f"Invalid audio file (impossible to extract chunk):\n\t{audio_file}"

    # Generate the output file name if not provided
    if osh.emptystring(output_audio_filename):
        f, b, ext = osh.folder_name_ext(audio_file)
        s = round(start_time * 1000)  # Start time in milliseconds
        e = round(end_time * 1000)  # End time in milliseconds
        output_audio_filename = osh.os_path_constructor(
            [f, f"{b}_chunk-{s}-{e}.{ext}"]
        )
    
    # if not(_overwrite_audio_file(output_audio_filename, overwrite) is None):
    #     return output_audio_filename
    
    # Get the duration of the audio file to validate start and end times
    duration = get_audio_duration(audio_file)
    assert start_time >= 0 and start_time < duration, f"Invalid start time: start={start_time}, end={end_time} for duration={duration}"
    assert end_time > start_time and end_time <= duration, f"Invalid end time: start={start_time}, end={end_time} for duration={duration}"

    _, _, ext_in = osh.folder_name_ext(audio_file)
    _, _, ext_out = osh.folder_name_ext(output_audio_filename)

    # Use ffmpeg to extract the audio chunk from the input file
    quiet = True
    # Use wav format for intermediate files
    with osh.temporary_filename(
        suffix=".wav", mode="wb"
    ) as temp_wav, osh.temporary_filename(
        suffix=".wav", mode="wb"
    ) as output_wav:
        
        if not(ext_in.lower() == "wav"):
            # Convert the input audio file to a temporary WAV file
            ffmpeg.input(audio_file).output(temp_wav).run(
                overwrite_output=True, quiet=quiet
            )
        else:
            osh.copyfile(audio_file, temp_wav)

        # Extract the audio chunk from the input file to a temporary WAV file
        ffmpeg.input(temp_wav, ss=start_time, t=end_time - start_time).output(
            output_wav
        ).run(overwrite_output=True, quiet=quiet)

        if not(ext_out.lower() == "wav"):
            # Convert the temporary WAV file to the specified output format
            sound_converter(output_wav, output_audio_filename, freq=44100)
        else:
            osh.copyfile(output_wav, output_audio_filename)

    # Verify that the output file was created and is valid
    osh.checkfile(
        output_audio_filename,
        msg=f"Failed to extract audio chunk from:\n\t{audio_file} to:\n\t{output_audio_filename}",
    )
    assert is_valid_audio_file(output_audio_filename), f"Failed to extract audio chunk from {start_time} to {end_time}:\n\t{output_audio_filename}"

    logging.info(
        f"Extracted audio chunk from\n\t{audio_file} to\n\t{output_audio_filename}"
    )

    return output_audio_filename



def generate_silent_audio(
    duration: float,
    output_audio_filename: str = None,
    sample_rate: int = 44100,
    overwrite: bool = False
) -> str:
    """
    Generate a silent audio file of a specified duration.

    Parameters
    ----------
    duration : float
        The duration of the silent audio file in seconds.
    output_audio_filename : str, optional
        The path to save the generated silent audio file. If None, a default file name will be generated.
    sample_rate : int, optional
        The sample rate of the silent audio file in Hz (default is 44100 Hz).
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists (default is False).

    Returns
    -------
    str
        The path to the generated silent audio file.

    Raises
    ------
    Error
        If the silent audio generation fails.

    Notes
    -----
    This function uses ffmpeg to generate a silent audio file of the specified duration and sample rate.
    If the output filename is not provided, it generates a default name based on the duration.
    """

    # Generate default output file name if not provided
    if osh.emptystring(output_audio_filename):
        t = round(duration * 1000)  # Convert duration to milliseconds for the filename
        output_audio_filename = osh.os_path_constructor([f"silent_{t}.wav"])

    # Check if the file already exists and handle based on the overwrite flag
    if not(_overwrite_audio_file(output_audio_filename, overwrite) is None):
        return output_audio_filename

    # Control ffmpeg's verbosity based on environment settings
    quiet = True

    # Just make zeros
    zeros = np.zeros(int(duration * sample_rate))
    _,_,ext = osh.folder_name_ext(output_audio_filename)
    if ext.lower() == "wav":
        sf.write(output_audio_filename, zeros, sample_rate)
    else:
        with osh.temporary_filename(suffix=".wav", mode="wb") as temp_wav:
            sf.write(temp_wav, zeros, sample_rate)
            sound_converter(temp_wav, output_audio_filename, freq=sample_rate)

    # Verify that the file was successfully generated and is valid
    osh.checkfile(
        output_audio_filename,
        msg=f"Failed to generate silent audio file: {output_audio_filename}",
    )
    assert is_valid_audio_file(output_audio_filename), f"Generated silent audio file is invalid: {output_audio_filename}"

    signal, sample_rate = load_audio(output_audio_filename, to_numpy=True, to_mono=True)
    assert np.sum(np.abs(signal)) == 0, f"Generated silent audio file is not silent:\n\t{output_audio_filename}"

    logging.info(f"Generated silent audio file: {output_audio_filename}")

    return output_audio_filename



def audio_concatenation(audio_files, output_audio_filename: str = None, overwrite:bool = False) -> str:
    """
    Concatenate multiple audio files into a single audio file.

    Parameters
    ----------
    audio_files : list
        List of paths to the audio files.
    output_audio_filename : str or None, optional
        Path to save the concatenated audio file.
        If None, the output file will be saved in the same folder as the first audio file.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists (default is False).

    Returns
    -------
    str
        Path to the concatenated audio file

    Notes
    -----
    The function uses ffmpeg to concatenate multiple audio files into a single audio file.
    """

    assert isinstance(audio_files, list) and len(audio_files) > 0, f"Invalid audio files list: {audio_files}"
    s = "\n\t".join(audio_files)
    assert all([osh.file_exists(f) for f in audio_files]), f"Invalid audio files (file existence):\n\t{s}"
    assert all([is_valid_audio_file(f) for f in audio_files]), f"Invalid audio files (audio type):\n\t{s}"

    if osh.emptystring(output_audio_filename):
        folder, _, ext = osh.folder_name_ext(audio_files[0])
        audio_files_basename = []
        for f in audio_files:
            _, b, _ = osh.folder_name_ext(f)
            audio_files_basename.append(b)
        b = "-".join(audio_files_basename)
        output_audio_filename = osh.os_path_constructor(
            [folder, f"{b}-concatenated.{ext}"]
        )

    # Check if the file already exists and handle based on the overwrite flag
    if not(_overwrite_audio_file(output_audio_filename, overwrite) is None):
        return output_audio_filename

    input_streams = [ffmpeg.input(f) for f in audio_files]

    quiet = True

    _,_,ext = osh.folder_name_ext(output_audio_filename)

    if ext.lower() == "wav":
        (
            ffmpeg.concat(*input_streams, v=0, a=1)
            .output(output_audio_filename)
            .run(overwrite_output=True, quiet=quiet)
        )
    else:
        with osh.temporary_filename(suffix=".wav", mode="wb") as temp_wav:
            (
                ffmpeg.concat(*input_streams, v=0, a=1)
                .output(temp_wav)
                .run(overwrite_output=True, quiet=quiet)
            )
            sound_converter(temp_wav, output_audio_filename, freq=44100)

    osh.checkfile(
        output_audio_filename, msg=f"Failed to concatenate audio files: {audio_files}"
    )
    assert is_valid_audio_file(output_audio_filename), f"Failed to concatenate audio files: {audio_files}"

    logging.info(f"Concatenated audio files into: {output_audio_filename}")

    return output_audio_filename


def split_audio_regularly(sound_path: str, chunk_folder: str, split_time: float, output_format = "mp3", overwrite: bool = False, suffix:str="split") -> List[str]:
    """
    Split an audio file into chunks of a specified duration.

    Parameters
    ----------
    sound_path : str
        Path to the audio file.
    chunk_folder : str
        Path to the folder where the audio chunks will be saved.
    split_time : float
        Duration of each audio chunk in seconds.
    output_format : str, optional
        The format of the output audio files (default is 'mp3').
    overwrite : bool, optional
        Whether to overwrite the output files if they already exist (default is False).
    suffix : str, optional
        Suffix to add to the output audio files (default is 'split').

    Returns
    -------
    List of audio file paths

    Notes
    -----
    The function uses ffmpeg to split the audio file into chunks of the specified duration.
    """
    
    assert is_valid_audio_file(sound_path), f"Invalid audio file: {sound_path}"

    output_format = output_format.lower().replace(".", "")

    # Ensure the output directory exists
    osh.make_directory(chunk_folder)

    # Calculate the total duration of the audio file
    total_duration = get_audio_duration(sound_path)

    # Process the chunks (actual splitting)
    time_cursor = 0
    counter = 0
    output_audio_paths = []
    while time_cursor < total_duration - 1:
        chunk_path = osh.os_path_constructor(
            [chunk_folder, f"chunk_{counter:04d}_{suffix}.{output_format}"]
        )
        s = time_cursor
        e = min(time_cursor + split_time, total_duration)
        extract_audio_chunk(sound_path, s, e, output_audio_filename = chunk_path, overwrite=True)
        added_duration = get_audio_duration(chunk_path)
        logging.info(
            f"Chunk {counter:04d} of duration {added_duration} saved to:\n\t{chunk_path}"
        )
        output_audio_paths.append(chunk_path)
        time_cursor += added_duration
        counter += 1

    s = "\n\t".join(output_audio_paths)
    logging.info(
        f"Audio file {sound_path} split into chunks of {split_time} seconds in {chunk_folder}:\n\t{s}"
    )

    return output_audio_paths




def hz_to_mel(hz: float) -> float:
    """
    Convert a frequency in Hertz to the Mel scale.
    
    Parameters
    ----------
    hz : float
        Frequency in Hertz.
    
    Returns
    -------
    float
        Frequency in Mels.
    """
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    """
    Convert a frequency in the Mel scale back to Hertz.
    
    Parameters
    ----------
    mel : float
        Frequency in Mels.
    
    Returns
    -------
    float
        Frequency in Hertz.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filter_banks(num_filters: int, 
                     n_fft: int, 
                     sample_rate: int, 
                     low_freq: int, 
                     high_freq: int) -> np.ndarray:
    """
    Compute a Mel-filter bank for given parameters.
    
    Parameters
    ----------
    num_filters : int
        Number of Mel filters to generate.
    n_fft : int
        The size of the FFT (number of FFT points).
    sample_rate : int
        The sample rate of the audio signal (in Hz).
    low_freq : int
        The lowest frequency in the Mel filter bank (in Hz).
    high_freq : int
        The highest frequency in the Mel filter bank (in Hz).
    
    Returns
    -------
    np.ndarray
        A 2D array where each row is a filter in the Mel-filter bank.
    """
    
    # Convert frequencies to the Mel scale
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)  # Equally spaced in Mel scale
    
    # Convert Mel frequencies back to Hz
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz frequencies to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)
    
    # Create the Mel filter bank
    fbank = np.zeros((num_filters, int(np.floor(n_fft / 2 + 1))))
    for i in range(1, num_filters + 1):
        f_m_minus = bin_points[i - 1]  # Left
        f_m = bin_points[i]            # Center
        f_m_plus = bin_points[i + 1]   # Right
        
        # Construct the filters
        for j in range(f_m_minus, f_m):
            fbank[i - 1, j] = (j - f_m_minus) / (f_m - f_m_minus)
        for j in range(f_m, f_m_plus):
            fbank[i - 1, j] = (f_m_plus - j) / (f_m_plus - f_m)

    return fbank


def mfcc(signal: np.ndarray, 
         sample_rate: int, 
         num_mfcc: int = 13, 
         n_fft: int = 512, 
         num_filters: int = 26, 
         low_freq: int = 0, 
         high_freq: Optional[int] = None) -> np.ndarray:
    """
    Compute Mel-frequency Cepstral Coefficients (MFCC) for an audio signal.
    
    Parameters
    ----------
    signal : np.ndarray
        The input audio signal as a 1D NumPy array.
    sample_rate : int
        The sample rate of the audio signal (in Hz).
    num_mfcc : int, optional
        The number of MFCC features to return, by default 13.
    n_fft : int, optional
        The FFT size to use, by default 512.
    num_filters : int, optional
        The number of Mel filters to use, by default 26.
    low_freq : int, optional
        The lowest frequency to consider in the Mel filter bank, by default 0 Hz.
    high_freq : int, optional
        The highest frequency to consider in the Mel filter bank, by default None (set to half the sample rate).
    
    Returns
    -------
    np.ndarray
        A 2D NumPy array containing the computed MFCC features for each frame.
    """
    
    # 1. Pre-emphasis filter: Apply a high-pass filter to amplify high frequencies
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # 2. Framing: Split the signal into overlapping frames of 25ms (default)
    frame_size = 0.025  # 25 milliseconds per frame
    frame_stride = 0.01  # 10 milliseconds between consecutive frames
    frame_length = int(round(frame_size * sample_rate))  # Convert frame length to samples
    frame_step = int(round(frame_stride * sample_rate))  # Convert frame step to samples
    
    # Compute the number of frames and pad the signal to fit exact frames
    num_frames = int(np.ceil(float(len(emphasized_signal) - frame_length) / frame_step)) + 1
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emphasized_signal, np.zeros(pad_signal_length - len(emphasized_signal)))
    
    # Create an index array for all frames and extract frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 3. Windowing: Apply a Hamming window to reduce spectral leakage
    frames *= get_window('hamming', frame_length)
    
    # 4. FFT and Power Spectrum: Compute the FFT and power spectrum for each frame
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))  # Magnitude of the FFT
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)  # Power spectrum

    # 5. Mel Filter Banks: Convert the power spectrum into the Mel scale
    high_freq = high_freq or sample_rate / 2  # If not provided, use half the sample rate (Nyquist frequency)
    mel_filters = mel_filter_banks(num_filters, n_fft, sample_rate, low_freq, high_freq)
    filter_banks = np.dot(pow_frames, mel_filters.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Avoid log(0)
    filter_banks = 20 * np.log10(filter_banks)  # Convert to decibels (logarithmic scale)

    # 6. DCT: Compute the Discrete Cosine Transform (DCT) of the log Mel filter banks
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_mfcc]

    return mfccs


def sound_resemblance(audio_file_1: str, audio_file_2: str) -> float:
    """
    Compute the resemblance score between two audio files using Mel-frequency Cepstral Coefficients (MFCC).

    Measure the resemblance between two audio files using the correlation coefficient.

    Score is between 0 and 1.
    The closer to 1, the more similar the signals.
    The closer to 0, the more different the signals.

    Parameters
    ----------
    audio_file_1 : str
        Path to the first audio file.
    audio_file_2 : str
        Path to the second audio file.

    Returns
    -------
    float
        The resemblance score between the two audio files based on MFCC.

    Notes
    -----
    The function computes the resemblance score between two audio files using the cosine similarity of their MFCC features.

    """
    sample_rate = 24000
    # Load audio files, convert to mono and numpy arrays, resample to target sample rate
    audio_1, _ = load_audio(audio_file_1, to_numpy=True, to_mono=True, target_sample_rate=sample_rate)
    audio_2, _ = load_audio(audio_file_2, to_numpy=True, to_mono=True, target_sample_rate=sample_rate)

    max_len = max(len(audio_1), len(audio_2))
    audio_1 = np.pad(audio_1, (0, max_len - len(audio_1)))
    audio_2 = np.pad(audio_2, (0, max_len - len(audio_2)))

    mfcc1 = mfcc(audio_1, sample_rate)
    mfcc2 = mfcc(audio_2, sample_rate)

    a = np.abs(np.dot(mfcc1.ravel(), mfcc2.ravel()))
    b = np.sqrt(np.dot(mfcc1.ravel(), mfcc1.ravel()) * np.dot(mfcc2.ravel(), mfcc2.ravel()))

    if b == 0:
        return 0.0 # by convention (one of the signals is made of zeros)
    
    score = a / b
        
    return score
