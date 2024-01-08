# MusicGen Remixer Replicate API-Caller
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakemin/musicgen-remixer-replicate-api-caller/blob/main/musicgen_remixer.ipynb)
### Remix the music into another styles with MusicGen Chord
[MusicGen Remixer](https://replicate.com/sakemin/musicgen-remixer) is an app based on MusicGen Chord. Users can upload a music track with vocals, type in the text description prompt, and the app will create a new background track based on the input and then make a remixed music output.
This Jupyter notebook breaks down the process of MusicGen Remixer, by calling separate Replicate API calls and processing the outputs of the API calls.
# Installation
## Requirements
Works on `Python>=3.8 && <3.11`.
```
pip install -r requirements.txt
```
```
  annotated-types==0.6.0
  anyio==4.2.0
  certifi==2023.11.17
  charset-normalizer==3.3.2
  exceptiongroup==1.2.0
  h11==0.14.0
  httpcore==1.0.2
  httpx==0.26.0
  idna==3.6
  numpy==1.26.3
  packaging==23.2
  pydantic==2.5.3
  pydantic_core==2.14.6
  pydub==0.25.1
  replicate==0.22.0
  requests==2.31.0
  scipy==1.11.4
  sniffio==1.3.0
  typing_extensions==4.9.0
  urllib3==2.1.0
```
## Replicate API Token Login
To call the APIs from Replicate, you must login to Replicate after installing `replicate` python package with `pip`.
You can find your Replicate API tokens [here](https://replicate.com/account).
```
export REPLICATE_API_TOKEN=<your token>
```
# Run
## Python on Command Line
```
python musicgen_remixer.py --prompt="bossa nova" --audio_path=/path/to/your/audio/input.mp3
```
### Arguments
- `--prompt` : The prompt to use for generating the remix.
- `--audio_path` : The path to the audio file to remix.
- `--model_version` : The version of the model to use for generating the remix.
  - Default : `chord`
  - Options : [`chord`, `chord-large`, `stereo-chord`, `stereo-chord-large`]
- `--beat_sync_threshold` : The threshold for beat synchronization. If None, beat synchronization is automatically set to `1.1/(bpm/60)`.
  - Default : `None`
- `--upscale` : Whether to upscale the audio to 48 kHz. (boolean Flag)
  - Default : `False`
- `--mix_weight` : The weight for the generated instrumental track when mixing with the vocal.(0~1)
  - Default : `0.7`
- `--output_path` : The path to save the output audio file.
  - Default : `output` 
