# RVC-TTS-Pipeline
Pipeline for TTS to RVC.  This seems to produce the best sounding TTS with the closest representation to the original speaker's voice that one may have trained on.  This works by passing in an audio file generated from some type of TTS (tortoise, vits, etc.) and then converting it using the trained weights from an RVC model.  

To get this to work, pytorch must be installed first on the system to allow RVC to be installable.  If it's not, I was running into issues of having to uninstall and reinstall torch (though probably I should just adjust the requirements inside of rvc).

**It is still a work in progress, there will be bugs and issues.**

## Installation

1. Install pytorch first here: https://pytorch.org/get-started/locally/

2. Then, to install rvc, run the following:

```
pip install -e git+https://github.com/JarodMica/rvc.git#egg=rvc
```

3. Lastly, you need to get the ```hubert_base.pt``` and ```rmvpe.pt``` files from rvc and put them into the parent directory of whatever project you're working on (or the SAME location of whereever you're running the scripts)

**If you want to install rvc-tts-pipeline as it's own package, run the following (recommended)**

```
pip install -e git+https://github.com/JarodMica/rvc-tts-pipeline.git#egg=rvc_tts_pipe
```

This will allow you to import ```rvc_infer``` so that you do not have to move this package around.

## Basic usage
The only function that should be called is the ```rvc_convert``` function.  The only required parameters that are absolutely needed are:

```model_path = path to the model```

```input_path = path to the audio file to convert (or tts output audio file)```

Then, it can simply be called like the following:

```
from rvc_infer import rvc_convert

rvc_convert(model_path="your_model_path_here", input_path="your_audio_path_here")
```

The docstrings of rvc_convert details out other values you may want to play around with, probably the most important being pitch and f0method.

## Notes
Currently, the github packages ONLY work if you install them in editable mode.  Why exactly, I am not too sure but may have to do with package structure etc. If a time comes to where "-e" is not suitable for my projects, I will look for a solution when that comes.

## Acknowledgements
Huge thanks to the RVC creators as none of this would be possible without them.  This takes and uses a lot of their code in order to make this possible.
