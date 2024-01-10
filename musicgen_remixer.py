# MusicGen Remixer Replicate API-Caller

### Remix the music into another styles with MusicGen Chord

# [MusicGen Remixer](https://replicate.com/sakemin/musicgen-remixer) is an app based on MusicGen Chord.

# Users can upload a music track with vocals, type in the text description prompt,
# and the app will create a new background track based on the input and then make a remixed music output.

# This Python script breaks down the process of MusicGen Remixer,
# by calling separate Replicate API calls and processing the outputs of the API calls.

### Before run the script, you must install the Replicate CLI and login to Replicate.
### REPLICATE_API_TOKEN must be set as an environment variable.
### ```export REPLICATE_API_TOKEN=<your token>```
### You can check your token [here](https://replicate.com/account).

import replicate
from pathlib import Path
import urllib.request, json
import numpy as np
import requests
from pydub import AudioSegment
from io import BytesIO
from scipy.signal import resample
import argparse
import logging


def download_audio_and_load_as_numpy(url):
    """
    Downloads an audio file (MP3 or WAV) from a given URL and loads it into a NumPy array.

    Parameters:
    url (str): The URL of the audio file to download.

    Returns:
    np.ndarray: The audio data as a NumPy array.
    int: The sample rate of the audio file.
    """
    # Download the audio file
    response = requests.get(url)
    response.raise_for_status()

    # Read the audio file as a byte stream
    audio_file = BytesIO(response.content)

    # Determine the file format from the URL
    file_format = url.split(".")[-1].lower()

    # Load the audio file using pydub
    if file_format == "mp3":
        audio = AudioSegment.from_mp3(audio_file)
    elif file_format == "wav":
        audio = AudioSegment.from_wav(audio_file)
    else:
        raise ValueError("Unsupported file format: only MP3 and WAV are supported.")

    # Convert the audio to a NumPy array with a channel dimension
    channel_count = audio.channels
    audio_data = np.array(audio.get_array_of_samples())
    if channel_count == 2:
        audio_data = audio_data.reshape(-1, 2)
    else:
        audio_data = audio_data.reshape(-1, 1)

    return audio_data, audio.frame_rate


def save_numpy_as_audio(audio_data, sample_rate, output_filename):
    """
    Saves a NumPy array as an audio file (MP3 or WAV).

    Parameters:
    audio_data (np.ndarray): The audio data to save.
    sample_rate (int): The sample rate of the audio data.
    output_filename (str): The name of the output file.
    """
    # Determine the file format from the output filename
    file_format = output_filename.split(".")[-1].lower()

    # Determine the number of channels based on the shape of the audio data
    channels = audio_data.shape[1] if audio_data.ndim > 1 else 1

    # Convert the NumPy array to an AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=channels,
    )

    # Export the AudioSegment as an audio file
    audio_segment.export(output_filename, format=file_format)


def resample_audio(audio_data, original_sample_rate, new_sample_rate):
    """
    Resamples the audio data to a new sample rate.

    Parameters:
    audio_data (np.ndarray): The audio data to resample.
    original_sample_rate (int): The original sample rate of the audio data.
    new_sample_rate (int): The new sample rate to resample to.

    Returns:
    np.ndarray: The resampled audio data.
    """
    # Calculate the number of samples in the resampled audio
    num_original_samples = audio_data.shape[0]
    resample_ratio = new_sample_rate / original_sample_rate
    num_new_samples = int(num_original_samples * resample_ratio)

    # Resample the audio data
    resampled_audio = resample(audio_data, num_new_samples)

    return resampled_audio


def normalize_audio(audio_data):
    """
    Normalizes the audio data in a NumPy array to a range of -1 to 1.

    Parameters:
    audio_data (np.ndarray): The audio data to normalize.

    Returns:
    np.ndarray: The normalized audio data.
    """
    # Find the maximum absolute value in the audio data
    max_val = np.max(np.abs(audio_data))

    # Normalize the audio data to the range [-1, 1]
    normalized_audio = audio_data / max_val

    return normalized_audio


def mix_audio_volumes(audio1, audio2, weight1=0.5, weight2=0.5):
    """
    Mixes two audio numpy arrays with given weights to ensure even volume.

    Parameters:
    audio1 (np.ndarray): The first audio data to mix.
    audio2 (np.ndarray): The second audio data to mix.
    weight1 (float): The weight for the first audio data.
    weight2 (float): The weight for the second audio data.

    Returns:
    np.ndarray: The mixed audio data.
    """
    if audio1.shape != audio2.shape:
        raise ValueError("Both audio arrays must have the same shape")

    # Normalize each audio array
    audio1_normalized = audio1 / np.max(np.abs(audio1))
    audio2_normalized = audio2 / np.max(np.abs(audio2))

    # Apply weights and mix
    mixed_audio = (audio1_normalized * weight1) + (audio2_normalized * weight2)

    # Scale to int16 range and convert
    max_int16 = np.iinfo(np.int16).max
    mixed_audio_scaled = np.clip(mixed_audio * max_int16, -max_int16, max_int16).astype(
        np.int16
    )

    return mixed_audio_scaled


def int16_scale(audio):
    # Scale to int16 range and convert
    max_int16 = np.iinfo(np.int16).max
    audio_scaled = np.clip(audio * max_int16, -max_int16, max_int16).astype(np.int16)

    return audio_scaled


def main(
    prompt,
    audio_path,
    model_version="chord",
    beat_sync_threshold=None,
    upscale=False,
    mix_weight=0.65,
    output_path="output",
):
    """
    Generates a remix of an audio file based on a given prompt.

    Parameters:
    prompt (str): The prompt to use for generating the remix.
    audio_path (str): The path to the audio file to remix.
    model_version (str): The version of the model to use for generating the remix. [`chord`, `chord-large`, `stereo-chord`, `stereo-chord-large`]
    beat_sync_threshold (float): The threshold for beat synchronization. If None, beat synchronization is automatically set to `1.1/(bpm/60)`.
    upscale (bool): Whether to upscale the audio to 48 kHz.
    output_path (str): The path to save the output audio file.
    """

    logger = logging.getLogger("postprocessor")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ### Make the output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    (Path(output_path) / "WIPs").mkdir(parents=True, exist_ok=True)

    ### Determine the number of channels based on the model version
    if "stereo" in model_version:
        channel = 2
    else:
        channel = 1

    ## Get BPM and downbeat analysis of input audio, using [All-In-One Music Structure Analyzer](https://replicate.com/sakemin/all-in-one-music-structure-analyzer)
    logger.info("Getting BPM and downbeat analysis of input audio...")
    time_analysis_url = replicate.run(
        "sakemin/all-in-one-music-structure-analyzer:001b4137be6ac67bdc28cb5cffacf128b874f530258d033de23121e785cb7290",
        input={"music_input": Path(audio_path)},
    )

    ### Download the output JSON and get BPM and downbeat values
    with urllib.request.urlopen(time_analysis_url[0]) as url:
        data = json.load(url)

    time_analysis = data

    bpm = time_analysis["bpm"]
    input_downbeats = time_analysis["downbeats"]

    print("BPM:", bpm)
    print("Downbeats:", input_downbeats)

    ### Set `beat_sync_threshold` when it is `None`
    if not beat_sync_threshold or beat_sync_threshold == -1:
        if bpm is not None:
            beat_sync_threshold = 1.1 / (int(bpm) / 60)
        else:
            beat_sync_threshold = 0.75
        logger.info(f"Setting beat_sync_threshold to {beat_sync_threshold}")

    ## Separate vocal track out of instrumental track, using [Demucs](https://replicate.com/cjwbw/demucs)
    logger.info("Separating vocal track out of instrumental track...")
    track_urls = replicate.run(
        "cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953",
        input={
            "audio": Path(audio_path),
            "stem": "vocals",
            "shifts": 2,  # higher values for better quality, but takes more time
            "float32": True,
            "output_format": "mp3",
        },
    )

    ### Download the separated vocal track
    vocal_track, vocal_sr = download_audio_and_load_as_numpy(track_urls["vocals"])

    ### Save the vocal track in mp3 format
    vocal_path = (
        str(Path(output_path) / "WIPs" / Path(audio_path).name.rsplit(".", 1)[0])
        + "_vocals.mp3"
    )
    save_numpy_as_audio(vocal_track, vocal_sr, vocal_path)
    logger.info(f"Saved vocal track to {vocal_path}")

    ### Download the separated instrumental track
    instrumental_track, instrumental_sr = download_audio_and_load_as_numpy(
        track_urls["other"]
    )

    ### Save the instrumental track in mp3 format
    instrumental_path = (
        str(Path(output_path) / "WIPs" / Path(audio_path).name.rsplit(".", 1)[0])
        + "_inst.mp3"
    )
    save_numpy_as_audio(instrumental_track, instrumental_sr, instrumental_path)
    logger.info(f"Saved instrumental track to {instrumental_path}")

    ## Generate a new instrumental track with the prompt and the original instrumental track as input, using [MusicGen-Stereo-Chord](https://replicate.com/sakemin/musicgen-stereo-chord)
    # - This might take a while.
    logger.info("Generating a new instrumental track...")
    generated_instrumental_url = replicate.run(
        "sakemin/musicgen-stereo-chord:fbdc5ef7200220ed300015d9b4fd3f8e620f84547e970b23aa2be7f2ff366a5b",
        input={
            "model_version": model_version,
            "prompt": prompt + ", bpm: " + str(bpm),
            "audio_chords": Path(instrumental_path),
            "duration": int(instrumental_track.shape[0] / instrumental_sr),
        },
    )

    ### Download the generated instrumental track
    (
        generated_instrumental_track,
        generated_instrumental_sr,
    ) = download_audio_and_load_as_numpy(generated_instrumental_url)

    ### Save the generated instrumental track in mp3 format
    generated_instrumental_path = (
        str(Path(output_path) / "WIPs" / Path(audio_path).name.rsplit(".", 1)[0])
        + f"_{prompt}"
        + "_generated_inst.mp3"
    )
    save_numpy_as_audio(
        generated_instrumental_track,
        generated_instrumental_sr,
        generated_instrumental_path,
    )
    logger.info(f"Saved generated instrumental track to {generated_instrumental_path}")

    ## Sample rate matching(Choose one of the 2 options below)
    # - Choose one of the two ways given below

    ### 1. Force upsample the generated track to the input sample rate
    if not upscale:
        logger.info("Upsampling the generated instrumental track...")
        resampled_instrumental_track = resample_audio(
            generated_instrumental_track, generated_instrumental_sr, vocal_sr
        )
        resampled_instrumental_track = int16_scale(
            normalize_audio(resampled_instrumental_track)
        )

    ### 2. Upscale the tracks to 48khz with [Audio-Super-Resolution](https://replicate.com/sakemin/audiosr-long-audio)
    if upscale:
        logger.info("Upscaling the tracks to 48khz...")
        resampled_instrumental_url = replicate.run(
            "sakemin/audiosr-long-audio:44b37256d8d2ade24655f05a0d35128642ca90cbad0f5fa0e9bfa2d345124c8c",
            input={"input_file": Path(generated_instrumental_path)},
        )
        (
            resampled_instrumental_track,
            resampled_instrumental_sr,
        ) = download_audio_and_load_as_numpy(resampled_instrumental_url)
        resampled_vocal_url = replicate.run(
            "sakemin/audiosr-long-audio:44b37256d8d2ade24655f05a0d35128642ca90cbad0f5fa0e9bfa2d345124c8c",
            input={"input_file": Path(vocal_path)},
        )
        vocal_track, vocal_sr = download_audio_and_load_as_numpy(resampled_vocal_url)

    ### Save the resampled instrumental track in mp3 format
    resampled_instrumental_path = (
        str(Path(output_path) / "WIPs" / Path(audio_path).name.rsplit(".", 1)[0])
        + f"_{prompt}"
        + "_resampled_inst.mp3"
    )
    save_numpy_as_audio(
        resampled_instrumental_track, vocal_sr, resampled_instrumental_path
    )
    logger.info(f"Saved resampled instrumental track to {resampled_instrumental_path}")

    ## Beat synchronization
    ### Get BPM and downbeat analysis of generated audio, using [All-In-One Music Structure Analyzer](https://replicate.com/sakemin/all-in-one-music-structure-analyzer)
    logger.info("Getting BPM and downbeat analysis of generated audio...")
    output_time_analysis_url = replicate.run(
        "sakemin/all-in-one-music-structure-analyzer:001b4137be6ac67bdc28cb5cffacf128b874f530258d033de23121e785cb7290",
        input={"music_input": Path(resampled_instrumental_path)},
    )

    ### Download the output JSON and get downbeat value
    with urllib.request.urlopen(output_time_analysis_url[0]) as url:
        data = json.load(url)

    output_time_analysis = data

    generated_downbeats = output_time_analysis["downbeats"]

    ### Align the downbeats pair-wise
    logger.info("Aligning the downbeats pair-wise...")
    aligned_generated_downbeats = []
    aligned_input_downbeats = []

    for generated_downbeat in generated_downbeats:
        input_beat = min(
            input_downbeats, key=lambda x: abs(generated_downbeat - x), default=None
        )
        if input_beat is None:
            continue
        print(generated_downbeat, input_beat)
        if (
            len(aligned_input_downbeats) != 0
            and int(input_beat * vocal_sr) == aligned_input_downbeats[-1]
        ):
            print("Dropped")
            continue
        if abs(generated_downbeat - input_beat) > beat_sync_threshold:
            input_beat = generated_downbeat
            print("Replaced")
        aligned_generated_downbeats.append(int(generated_downbeat * vocal_sr))
        aligned_input_downbeats.append(int(input_beat * vocal_sr))

    wav_length = resampled_instrumental_track.shape[-2]
    downbeat_offset = aligned_input_downbeats[0] - aligned_generated_downbeats[0]
    if downbeat_offset > 0:
        resampled_instrumental_track = np.concatenate(
            [
                np.zeros([1, channel, int(downbeat_offset)]),
                resampled_instrumental_track,
            ],
            dim=-1,
        )
        for i in range(len(aligned_generated_downbeats)):
            aligned_generated_downbeats[i] = (
                aligned_generated_downbeats[i] + downbeat_offset
            )
    aligned_generated_downbeats = [0] + aligned_generated_downbeats + [wav_length]
    aligned_input_downbeats = [0] + aligned_input_downbeats + [wav_length]

    s_ap = ""
    for i in range(len(aligned_generated_downbeats) - 1):
        s_ap += (
            str(aligned_generated_downbeats[i])
            + ":"
            + str(aligned_input_downbeats[i])
            + ", "
        )
    s_ap += (
        str(aligned_generated_downbeats[-1]) + ":" + str(aligned_input_downbeats[-1])
    )

    ### Apply dynamic time-stretching on the generated instrumental track, using [PyTSMod](https://replicate.com/sakemin/pytsmod)
    logger.info(
        "Applying dynamic time-stretching on the generated instrumental track..."
    )
    time_stretched_instrumental_track_url = replicate.run(
        "sakemin/pytsmod:41b355721c8a7ed501be7fd89e73631e7c07d75e1c94b1372c1c119b0774cdae",
        input={
            "audio_input": Path(resampled_instrumental_path),
            "s_ap": s_ap,
            "absolute_frame": True,
        },
    )

    ### Download the time-stretched instrumental track
    (
        time_stretched_instrumental_track,
        time_stretched_instrumental_sr,
    ) = download_audio_and_load_as_numpy(time_stretched_instrumental_track_url)

    ### Save the time-stretched instrumental track in mp3 format
    time_stretched_instrumental_path = (
        str(Path(output_path) / "WIPs" / Path(audio_path).name.rsplit(".", 1)[0])
        + f"_{prompt}"
        + "_time_stretched_inst.mp3"
    )
    save_numpy_as_audio(
        time_stretched_instrumental_track,
        time_stretched_instrumental_sr,
        time_stretched_instrumental_path,
    )
    logger.info(
        f"Saved time-stretched instrumental track to {time_stretched_instrumental_path}"
    )

    ## Combine the generated instrumental track and the original vocal track
    ### Pad the generated track's length
    pad = vocal_track.shape[0] - time_stretched_instrumental_track.shape[0]
    if pad > 0:
        padded_instrumental_track = np.pad(
            time_stretched_instrumental_track, ((0, pad), (0, 0)), "constant"
        )
    else:
        padded_instrumental_track = time_stretched_instrumental_track[
            : vocal_track.shape[0]
        ]

    ### Make the number of channels consistent
    if channel == 1 and vocal_track.shape[1] == 2:
        padded_instrumental_track = np.repeat(padded_instrumental_track, 2, axis=1)
    if channel == 2 and vocal_track.shape[1] == 1:
        vocal_track = np.repeat(vocal_track, 2, axis=1)

    ### Mix and normalize two tracks
    logger.info("Mixing and normalizing two tracks...")
    mixed_track = mix_audio_volumes(
        padded_instrumental_track,
        vocal_track,
        weight1=mix_weight,
        weight2=1 - mix_weight,
    )

    ## Save the remixed track
    remixed_path = (
        str(Path(output_path) / Path(audio_path).name.rsplit(".", 1)[0])
        + f"_{prompt}"
        + "_remixed.mp3"
    )
    save_numpy_as_audio(mixed_track, time_stretched_instrumental_sr, remixed_path)
    logger.info(f"Saved remixed track to {remixed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--prompt", dest="prompt", help="The prompt to use for generating the remix."
    )
    parser.add_argument(
        "--audio_path", dest="audio_path", help="The path to the audio file to remix."
    )
    parser.add_argument(
        "--model_version",
        dest="model_version",
        default="chord",
        help="The version of the model to use for generating the remix. [`chord`, `chord-large`, `stereo-chord`, `stereo-chord-large`]",
    )
    parser.add_argument(
        "--beat_sync_threshold",
        dest="beat_sync_threshold",
        default=None,
        help="The threshold for beat synchronization. If None, beat synchronization is automatically set to `1.1/(bpm/60)`.",
    )
    parser.add_argument(
        "--upscale",
        dest="upscale",
        action="store_true",
        help="Whether to upscale the audio to 48 kHz.",
    )
    parser.add_argument(
        "--mix_weight",
        dest="mix_weight",
        default=0.7,
        help="The weight for the generated instrumental track when mixing with the vocal.(0~1)",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        default="output",
        help="The path to save the output audio file.",
    )
    args = parser.parse_args()
    main(
        args.prompt,
        args.audio_path,
        args.model_version,
        args.beat_sync_threshold,
        args.upscale,
        args.mix_weight,
        args.output_path,
    )
