import os,sys,pdb,torch
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)
import sys
import torch

from multiprocessing import cpu_count
from rvc.infer.modules.vc.modules import VC
from rvc.configs.config import Config

from scipy.io import wavfile


config = Config()
    
def get_path(name):
    '''
    Built to get the path of a file based on where the initial script is being run from.
    
    Args:
        - name(str) : name of the file/folder
    '''
    return os.path.join(os.getcwd(), name)

def create_directory(name):
    '''
    Creates a directory based on the location from which the script is run. Relies on
    get_path()
    
    Args:
        - name(str) : name of the file/folder
    '''
    dir_name = get_path(name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def rvc_convert(model_name,
            f0_up_key=0,
            input_path=None, 
            output_dir_path=None,
            _is_half="False",
            f0method="rmvpe",
            file_index="",
            file_index2="",
            index_rate=1,
            filter_radius=3,
            resample_sr=48000,
            rms_mix_rate=0.5,
            protect=0.33,
            verbose=False
          ):  
    '''
    Function to call for the rvc voice conversion.  All parameters are the same present in that of the webui

    Args: 
        model_name (str) : path to the rvc voice model you're using (should be in the rvc weights folder)
        f0_up_key (int) : transpose of the audio file, changes pitch (positive makes voice higher pitch)
        input_path (str) : path to audio file (use wav file)
        output_dir_path (str) : path to output directory, defaults to parent directory in output folder
        _is_half (str) : Determines half-precision
        f0method (str) : picks which f0 method to use: dio, harvest, crepe, rmvpe (requires rmvpe.pt)
        file_index (str) : path to file_index, defaults to None (should be in the rvc indexes folder)
        file_index2 (str) : path to file_index2, defaults to None.  #honestly don't know what this is for
        index_rate (int) : strength of the index file if provided
        filter_radius (int) : if >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.
        resample_sr (int) : quality at which to resample audio to, defaults to no resample
        rmx_mix_rate (int) : adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume
        protect (int) : protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy

    Returns:
        output_file_path (str) : file path and name of the output wav file

    '''
    global config, now_dir, hubert_model, tgt_sr, vc, device, is_half

    if not verbose:
        logging.getLogger('fairseq').setLevel(logging.ERROR)
        logging.getLogger('rvc').setLevel(logging.ERROR)
    
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps:0"
    else:
        print("Cuda or MPS not detected")

    is_half = _is_half

    if file_index == "":
        pass
    else:
        file_index = os.path.join(os.getcwd(), file_index)

    if output_dir_path == None:
        output_dir_path = "output"
        output_file_name = "out.wav"
        output_dir = os.getcwd()
        output_file_path = os.path.join(output_dir,output_dir_path, output_file_name)
    else:
        # Mainly for Jarod's Vivy project, specify entire path + wav name
        output_file_path = output_dir_path
        pass

    create_directory(output_dir_path)
    output_dir = get_path(output_dir_path)

    if(is_half.lower() == 'true'):
        is_half = True
    else:
        is_half = False

    config=Config(device,is_half)
    now_dir=os.getcwd()
    sys.path.append(now_dir)

    hubert_model=None

    vc = VC(config)
    vc.get_vc(model_name)
    _, (tgt_sr, audio_opt) = vc.vc_single(0, input_path, f0_up_key, None, f0method, file_index, file_index2, index_rate, filter_radius, resample_sr, rms_mix_rate, protect)
    
    wavfile.write(output_file_path, tgt_sr, audio_opt)
    print(f"\nFile finished writing to: {output_file_path}")

    return output_file_path

def main():
    rvc_convert(f0_up_key=6, model_name="rvc_voices/azasu.pth", input_path="delilah.wav")

if __name__ == "__main__":
    main()