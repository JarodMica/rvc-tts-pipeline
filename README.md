# RVC-TTS-Pipeline
Pipeline for TTS to RVC.  This seems to produce the best sounding TTS with the closest representation to the original speaker's voice that one may have trained on.  This works by passing in an audio file generated from some type of TTS (tortoise, vits, etc.) and then converting it using the trained weights from an RVC model.  

To get this to work, pytorch must be installed first on the system.  As well, this differs from the main branch as well as you'll need to download the rvc folder found on the releases sectin of my rvc fork

**It is still a work in progress, there will be bugs and issues.**

## Installation

1. Install pytorch first here: https://pytorch.org/get-started/locally/

2. Then, to get rvc, go to the following HF link, extract rvc_lightweight, and then place the folder named ```rvc``` into the parent directory of wherever you're running your script from: https://huggingface.co/Jmica/rvc/blob/main/rvc_lightweight.7z
3. You'll then need to cd into the rvc and install ```pip install -r requirements.txt```

**If you want to install rvc-tts-pipeline as it's own package, run the following (recommended)**

```
pip install git+https://github.com/JarodMica/rvc-tts-pipeline.git@lightweight#egg=rvc_tts_pipe
```

This will allow you to import ```rvc_pipe.rvc_infer``` so that you do not have to move this package around.

## Basic usage
The only function that should be called is the ```rvc_convert``` function.  The only required parameters that are absolutely needed are:

```model_path = path to the model```

```input_path = path to the audio file to convert (or tts output audio file)```

Then, it can simply be called like the following:

```
from rvc_pipe.rvc_infer import rvc_convert

rvc_convert(model_path="your_model_path_here", input_path="your_audio_path_here")
```

The docstrings of rvc_convert details out other values you may want to play around with, probably the most important being pitch and f0method.

## Notes
The previous releases had this installed in "edittable" mode, but that is no longer necessary for the rvc-tts-pipeline as I've learned a few things since then.

## Acknowledgements
Huge thanks to the RVC creators as none of this would be possible without them.  This takes and uses a lot of their code in order to make this possible.
