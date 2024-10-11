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
- os-helper

Authors:
------------
- [Warith Harchaoui](https://harchaoui.org/warith)
- [Mohamed Chelali](https://mchelali.github.io)
- [Bachir Zerroug](https://www.linkedin.com/in/bachirzerroug)

"""

from typing import List, Union
import torch
import os_helper
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
import ffmpeg
from tqdm import tqdm
import concurrent.futures
import numpy as np


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
    if not(overwrite) and os_helper.file_exists(output_audio_filename):
        os_helper.info(f"Output audio file already exists:\n\t{output_audio_filename}")
        if is_valid_audio_file(output_audio_filename):
            return output_audio_filename
        else:
            os_helper.remove_files([output_audio_filename])  # Remove invalid file
            os_helper.info(f"Deleting invalid output audio file:\n\t{output_audio_filename}")
    elif overwrite and os_helper.file_exists(output_audio_filename):
        os_helper.remove_files([output_audio_filename])  # Overwrite existing file
        os_helper.info(f"Deleting output audio file for overwrite:\n\t{output_audio_filename}")

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
    if not(overwrite) and all([(os_helper.file_exists(f) and is_valid_audio_file(f)) for f in output_audio_list]):
        stem_keys = []
        stem_files = []
        for f in output_audio_list:
            _, b, _ = os_helper.folder_name_ext(f)
            o = os_helper.relative2absolute_path(f)
            stem_keys.append(b)
            stem_files.append(o)
        d = {k: v for k, v in zip(stem_keys, stem_files)}
        s = "\n\t".join([f"{k}:\t{v}" for k, v in d.items()])
        os_helper.info(
            f"Sources already separated for at:\n\t{s}"
        )
        return d
    elif overwrite:
        for f in output_audio_list:
            if os_helper.file_exists(f):
                os_helper.remove_files([f])
                os_helper.info(f"Deleting output audio file for overwrite:\n\t{f}")
    
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

    os_helper.info(f"Audio file {file_path} is {'valid' if valid else 'invalid'}")
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
    os_helper.checkfile(file_path, msg=f"Audio file not found at {file_path}")
    probe = ffmpeg.probe(file_path)
    audio_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None
    )
    os_helper.check(
        audio_stream is not None,
        msg=f"No audio stream found in the file: {file_path}",
    )
    return float(audio_stream["duration"])


def load_audio(
    file_path: str,
    target_sample_rate: int = None,
    to_mono: bool = True,
    to_numpy: bool = False,
    two_channels: bool = False,
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and optionally resample, convert to mono or stereo, and return as a NumPy array.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    target_sample_rate : int, optional
        The target sample rate to resample the audio to. Defaults to the original sample rate.
    to_mono : bool, optional
        Whether to convert the audio to mono (default is True).
    to_numpy : bool, optional
        Whether to convert the audio to a NumPy array (default is False).
    two_channels : bool, optional
        Whether to force the audio into two channels (stereo).

    Returns
    -------
    torch.Tensor or np.ndarray
        The loaded audio signal.
    int
        Sample rate of the loaded audio.
    """
    torchaudio.set_audio_backend("sox_io")
    os_helper.checkfile(file_path, msg=f"Audio file not found at {file_path}")
    os_helper.check(
        is_valid_audio_file(file_path),
        msg=f"Invalid audio file (impossible to load): {file_path}",
    )

    _,_,ext = os_helper.folder_name_ext(file_path)
    if not(ext.lower() == "wav"):
        with os_helper.temporary_filename(suffix=".wav", mode="wb") as wav_audio_file:
            ffmpeg.input(file_path).output(wav_audio_file).run(overwrite_output=True, quiet=True)
            audio, sample_rate = torchaudio.load(wav_audio_file, format='wav')
    else:
        print(file_path)
        audio, sample_rate = torchaudio.load(file_path, format='wav')

    if target_sample_rate is None:
        target_sample_rate = sample_rate

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        audio = resampler(audio)

    if to_mono and not two_channels:
        audio = audio.mean(dim=0)

    os_helper.info(
        f"Loaded audio file {file_path} with shape {audio.shape} and sample rate {target_sample_rate}"
    )

    if two_channels:
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        elif audio.shape[0] > 2:
            # Split audio into two channels by averaging left and right parts
            audio_left = audio[: audio.shape[0] // 2].mean(dim=0)
            audio_right = audio[audio.shape[0] // 2 :].mean(dim=0)
            audio = torch.cat([audio_left, audio_right], dim=0)

    if to_numpy:
        audio = audio.numpy()

    return audio, target_sample_rate



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

    os_helper.info(f"Converting audio file: {input_audio} into {output_audio}")

    # Check if the input audio file exists
    os_helper.checkfile(input_audio, msg=f"Input audio file not found: {input_audio}")

    # Check if the input audio file is valid
    os_helper.check(
        is_valid_audio_file(input_audio),
        msg=f"Invalid audio file: {input_audio}",
    )

    o = _overwrite_audio_file(output_audio, overwrite)
    if o is not None:
        return o
    
    _, _, ext_in = os_helper.folder_name_ext(input_audio)
    _, _, ext_out = os_helper.folder_name_ext(output_audio)

    # Get verbosity settings from the environment
    verbose = os_helper.verbosity() > 0
    quiet = not verbose

    # Use temporary files for intermediate WAV processing (for robustness)
    with os_helper.temporary_filename(
        suffix=".wav", mode="wb"
    ) as first_wav, os_helper.temporary_filename(
        suffix=".wav", mode="wb"
    ) as second_wav:

        if not(ext_in.lower() == "wav"):
            # Convert the input audio file to a temporary WAV file
            ffmpeg.input(input_audio).output(first_wav, format="wav").run(
                overwrite_output=True, quiet=quiet
            )
        else:
            os_helper.copyfile(input_audio, first_wav)


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
            os_helper.copyfile(second_wav, output_audio)

    # Check if the output audio file was successfully created
    os_helper.checkfile(
        output_audio, msg=f"Failed to convert audio file:\n\t{output_audio}"
    )
    os_helper.check(
        is_valid_audio_file(output_audio), msg=f"Invalid audio file:\n\t{output_audio}"
    )

    os_helper.info(f"Audio file converted successfully:\n\t{output_audio}")

    return output_audio


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

    # Get the number of workers from os_helper if nb_workers is not provided
    if nb_workers is None:
        nb_workers = os_helper.get_nb_workers()

    # Adjust workers count if nb_workers is negative (relative to the system like sklearn convention)
    if nb_workers < 0:
        nb_workers = os_helper.get_nb_workers() - nb_workers  + 1

    # Limit the number of workers to the maximum available minus one for the system
    MAX_NB_WORKERS = os_helper.get_nb_workers()
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

    global separator_engine
    os_helper.info(f"Separating sources for:\n\t{input_audio_file}")

    # Set up the output folder if not specified
    if output_folder is None:
        f, _, _ = os_helper.folder_name_ext(input_audio_file)
        output_folder = f

    # Check if files already exist and skip if not overwriting
    stem_keys = ["vocals", "drums", "bass", "other"]
    stem_files = [
        os_helper.os_path_constructor([output_folder, f"{stem}.{output_format}"])
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
        audio = sources.pop(0)
        # Merge the audio channels
        audio = audio.mean(0).unsqueeze(0)
        # Save the output audio files as with format set by output_format
        if output_format == "wav":
            wav_audio_file = os_helper.os_path_constructor([output_folder, f"{stem}.wav"])
            torchaudio.save(wav_audio_file, audio, sample_rate)
            res[stem] = wav_audio_file
        else:
            with os_helper.temporary_filename(suffix=".wav", mode="wb") as wav_audio_file:
                torchaudio.save(wav_audio_file, audio, sample_rate)
                output_audio_file = os_helper.os_path_constructor(
                    [output_folder, f"{stem}.{output_format}"]
                )
                sound_converter(wav_audio_file, output_audio_file, freq=sample_rate)
                os_helper.info(f"Saved {stem} to {output_audio_file}")
                res[stem] = output_audio_file

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
    os_helper.checkfile(audio_file, msg=f"Audio file not found at:\n\t{audio_file}")
    os_helper.check(
        is_valid_audio_file(audio_file),
        msg=f"Invalid audio file (impossible to extract chunk):\n\t{audio_file}",
    )

    # Generate the output file name if not provided
    if os_helper.emptystring(output_audio_filename):
        f, b, ext = os_helper.folder_name_ext(audio_file)
        s = round(start_time * 1000)  # Start time in milliseconds
        e = round(end_time * 1000)  # End time in milliseconds
        output_audio_filename = os_helper.os_path_constructor(
            [f, f"{b}_chunk-{s}-{e}.{ext}"]
        )
    
    if not(_overwrite_audio_file(output_audio_filename, overwrite) is None):
        return output_audio_filename
    
    # Get the duration of the audio file to validate start and end times
    duration = get_audio_duration(audio_file)
    os_helper.check(
        start_time >= 0 and start_time < duration,
        msg=f"Invalid start time: start={start_time}, end={end_time} for duration={duration}",
    )
    os_helper.check(
        end_time > start_time and end_time <= duration,
        msg=f"Invalid end time: start={start_time}, end={end_time} for duration={duration}",
    )

    # Use ffmpeg to extract the audio chunk from the input file
    quiet = os_helper.verbosity() == 0  # Control ffmpeg's verbosity
    ffmpeg.input(audio_file, ss=start_time, t=end_time - start_time).output(
        output_audio_filename
    ).run(overwrite_output=True, quiet=quiet)

    # Verify that the output file was created and is valid
    os_helper.checkfile(
        output_audio_filename,
        msg=f"Failed to extract audio chunk from:\n\t{audio_file} to:\n\t{output_audio_filename}",
    )
    os_helper.check(
        is_valid_audio_file(output_audio_filename),
        msg=f"Failed to extract audio chunk from {start_time} to {end_time}:\n\t{output_audio_filename}",
    )

    os_helper.info(
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
    if os_helper.emptystring(output_audio_filename):
        t = round(duration * 1000)  # Convert duration to milliseconds for the filename
        output_audio_filename = os_helper.os_path_constructor([f"silent_{t}.wav"])

    # Check if the file already exists and handle based on the overwrite flag
    if not(_overwrite_audio_file(output_audio_filename, overwrite) is None):
        return output_audio_filename

    # Control ffmpeg's verbosity based on environment settings
    quiet = os_helper.verbosity() == 0

    # Use ffmpeg to generate the silent audio file
    (
        ffmpeg.input(f"anullsrc=r={sample_rate}:cl=stereo", f="lavfi")
        .output(output_audio_filename, t=duration)
        .run(overwrite_output=True, quiet=quiet)
    )

    # Verify that the file was successfully generated and is valid
    os_helper.checkfile(
        output_audio_filename,
        msg=f"Failed to generate silent audio file: {output_audio_filename}",
    )
    os_helper.check(
        is_valid_audio_file(output_audio_filename),
        msg=f"Generated silent audio file is invalid: {output_audio_filename}",
    )

    t = load_audio(output_audio_filename, to_numpy=True, to_mono=True)
    os_helper.check(np.sum(t) == 0, msg=f"Generated silent audio file is not silent:\n\t{output_audio_filename}")

    os_helper.info(f"Generated silent audio file: {output_audio_filename}")

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

    os_helper.check(
        isinstance(audio_files, list) and len(audio_files) > 0,
        msg=f"Invalid audio files list: {audio_files}",
    )
    os_helper.check(
        all([os_helper.file_exists(f) for f in audio_files]),
        msg=f"Invalid audio files (file existence):\n\t{"\n\t".join(audio_files)}",
    )
    os_helper.check(
        all([is_valid_audio_file(f) for f in audio_files]),
        msg=f"Invalid audio files (audio type):\n\t{"\n\t".join(audio_files)}",
    )

    if os_helper.emptystring(output_audio_filename):
        folder, _, ext = os_helper.folder_name_ext(audio_files[0])
        audio_files_basename = []
        for f in audio_files:
            _, b, _ = os_helper.folder_name_ext(f)
            audio_files_basename.append(b)
        b = "-".join(audio_files_basename)
        output_audio_filename = os_helper.os_path_constructor(
            [folder, f"{b}-concatenated.{ext}"]
        )

    # Check if the file already exists and handle based on the overwrite flag
    if not(_overwrite_audio_file(output_audio_filename, overwrite) is None):
        return output_audio_filename

    input_streams = [ffmpeg.input(f) for f in audio_files]

    quiet = os_helper.verbosity() == 0

    _,_,ext = os_helper.folder_name_ext(output_audio_filename)

    if ext.lower() == "wav":
        (
            ffmpeg.concat(*input_streams, v=0, a=1)
            .output(output_audio_filename)
            .run(overwrite_output=True, quiet=quiet)
        )
    else:
        with os_helper.temporary_filename(suffix=".wav", mode="wb") as temp_wav:
            (
                ffmpeg.concat(*input_streams, v=0, a=1)
                .output(temp_wav)
                .run(overwrite_output=True, quiet=quiet)
            )
            sound_converter(temp_wav, output_audio_filename, freq=44100)

    os_helper.checkfile(
        output_audio_filename, msg=f"Failed to concatenate audio files: {audio_files}"
    )
    os_helper.check(
        is_valid_audio_file(output_audio_filename),
        msg=f"Failed to concatenate audio files: {audio_files}",
    )

    os_helper.info(f"Concatenated audio files into: {output_audio_filename}")

    return output_audio_filename

def split_audio_regularly(sound_path: str, chunk_folder: str, split_time: float, overwrite: bool = False) -> List[str]:
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
    overwrite : bool, optional
        Whether to overwrite the output files if they already exist (default is False).

    Returns
    -------
    List of audio file paths

    Notes
    -----
    The function uses ffmpeg to split the audio file into chunks of the specified duration.
    """

    os_helper.checkfile(sound_path, msg=f"Audio file not found at {sound_path}")

    # Ensure the output directory exists
    os_helper.make_directory(chunk_folder)

    # Calculate the total duration of the audio file
    total_duration = get_audio_duration(sound_path)

    # Generate the output audio chunk paths based on the split time
    output_audio_paths = [
        os_helper.os_path_constructor([chunk_folder, f"chunk_{i:04d}.wav"])
        for i in range(int(total_duration // split_time) + 1) if i * split_time < total_duration
    ]

    # Check if the file already exists and handle based on the overwrite flag
    d = _overwrite_audio_list(output_audio_paths, overwrite)
    if d is not None:
        return d

    # Process the chunks (actual splitting)
    time_cursor = 0
    counter = 0
    output_audio_paths = []
    while time_cursor < total_duration - 1.0:
        chunk_path = os_helper.os_path_constructor(
            [chunk_folder, f"chunk_{counter:04d}.wav"]
        )
        s = time_cursor
        e = min(time_cursor + split_time, total_duration)
        extract_audio_chunk(sound_path, s, e, chunk_path)
        added_duration = get_audio_duration(chunk_path)
        os_helper.info(
            f"Chunk {counter:04d} of duration {added_duration} saved to:\n\t{chunk_path}"
        )
        output_audio_paths.append(chunk_path)
        time_cursor += added_duration
        counter += 1

    s = "\n\t".join(output_audio_paths)
    os_helper.info(
        f"Audio file {sound_path} split into chunks of {split_time} seconds in {chunk_folder}:\n\t{s}"
    )

    return output_audio_paths
