# MusicGen Remixer
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
---
# References
- Chord recognition from audio file is performed using [BTC](https://github.com/jayg996/BTC-ISMIR19) model, by [Jonggwon Park](https://github.com/jayg996).
	- Paper : [A Bi-Directional Transformer for Musical Chord Recognition](https://arxiv.org/abs/1907.02698)
- Vocal dropping is implemented using Meta's [`demucs`](https://github.com/facebookresearch/demucs).
- Downbeat tracking and BPM retrieval is perfromed using [All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one#all-in-one-music-structure-analyzer) by [Taejun Kim](https://github.com/mir-aidj).
  - Paper : [All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio](https://arxiv.org/abs/2307.16425)
- Beat-syncing is performed with [PyTSMod](https://github.com/KAIST-MACLab/PyTSMod) by [MAC Lab @KAIST](https://github.com/KAIST-MACLab)
