# import librosa
# import soundfile as sf
# import pysndfile
# import wave
import os
import subprocess
import glob
from scipy.io import wavfile
# import pydub


def resample_audio(input_folder, output_folder):
    ffmpeg_path = r"C:\Users\amitk\Desktop\ffmpeg-20200330-8d019db-win64-static\bin\ffmpeg"
    input_wav_paths = glob.glob(os.path.join(input_folder, '*.wav'))
    for input_wav_path in input_wav_paths:
        wav_filename = os.path.basename(input_wav_path)
        output_wav_path = os.path.join(output_folder, wav_filename)
        print(input_wav_path, wav_filename, output_wav_path)
        ffmpeg_command = r"{ffmpeg_path} -i {input} -acodec pcm_s16le -ac 1 -ar 16000 {output}".format(ffmpeg_path=ffmpeg_path,
                                                                                                   input=input_wav_path,
                                                                                                   output=output_wav_path)
        subprocess.call(ffmpeg_command)
    return


def denoise_audio(input_folder, output_folder):
    from clean import clean
    import random
    import numpy as np
    import torch
    CKPT_PATH = r"C:\Users\amitk\data_for_experiments\gan_denoising"
    G_PRETRAINED_CKPT = "segan+_generator.ckpt"
    g_pretrained_ckpt = os.path.join(CKPT_PATH, G_PRETRAINED_CKPT)
    cfg_file = "train.opts"
    seed = 111
    cuda = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    clean(cfg_file, [input_folder], g_pretrained_ckpt, cuda, output_folder)


if __name__ == "__main__":
    print("ok")
    original_wav = r"C:\Users\amitk\data_for_experiments\gan_denoising\noisy_testset_wav\p232_023.wav"
    downsampled_wav = r"C:\Users\amitk\data_for_experiments\gan_denoising\my_test_inputs\downsampled_ffmpeg.wav"
    # ffmpeg_path = r"C:\Users\amitk\Desktop\ffmpeg-20200330-8d019db-win64-static\bin\ffmpeg"
    input_folder = r"C:\Users\amitk\data_for_experiments\gan_denoising\noisy_testset_wav"
    output_folder = r"C:\Users\amitk\data_for_experiments\gan_denoising\my_noisy_testset_wav"
    denoised_output_folder = r"C:\Users\amitk\data_for_experiments\gan_denoising\my_denoised_testset_wav"
    # ffmpeg_command = r"{ffmpeg_path} -i {input} -acodec pcm_s16le -ac 1 -ar 16000 {output}".format(ffmpeg_path=ffmpeg_path,
    #                                                                                                input=original_wav,
    #                                                                                                 output=downsampled_wav)
    # subprocess.call(ffmpeg_command)
    # resample_audio(input_folder, output_folder)
    denoise_audio(output_folder, denoised_output_folder)