# import argparse
import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from segan.models import *
from model import *
# from segan.datasets import *
from scipy.io import wavfile
# from torch.autograd import Variable
import numpy as np
import random
# import librosa
# import matplotlib
import timeit

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import json
import glob
import os
from wav_ops_for_running_clean import *


class ArgParser(object):
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)


def clean(cfg_file, test_files, g_pretrained_ckpt, cuda, synthesis_path):
    with open(cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))
    args.cuda = cuda
    if hasattr(args, 'wsegan') and args.wsegan:
        segan = WSEGAN(args)
    else:
        segan = SEGAN(args)
    segan.G.load_pretrained(g_pretrained_ckpt, True)
    if cuda:
        segan.cuda()
    segan.G.eval()
    # if opts.h5:
    #     with h5py.File(opts.test_files[0], 'r') as f:
    #         twavs = f['data'][:]
    # else:
        # process every wav in the test_files
    if len(test_files) == 1:
            # assume we read directory
        twavs = glob.glob(os.path.join(test_files[0], '*.wav'))
    else:
            # assume we have list of files in input
        twavs = test_files
    print('Cleaning {} wavs'.format(len(twavs)))
    beg_t = timeit.default_timer()
    for t_i, twav in enumerate(twavs, start=1):
        tbname = os.path.basename(twav)
        rate, wav = wavfile.read(twav)
        wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, args.preemph)
        pwav = torch.FloatTensor(wav).view(1, 1, -1)
        if cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav)
        out_path = os.path.join(synthesis_path,
                                tbname)
        wavfile.write(out_path, 16000, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t - beg_t))
        beg_t = timeit.default_timer()


if __name__ == '__main__':
    # CKPT_PATH = "ckpt_segan+"
    CKPT_PATH = r"C:\Users\amitk\data_for_experiments\gan_denoising"
    # please specify the path to your G model checkpoint
    # as in weights_G-EOE_<iter>.ckpt
    G_PRETRAINED_CKPT = "segan+_generator.ckpt"
    # please specify the path to your folder containing
    # noisy test files, each wav in there will be processed

    # TEST_FILES_PATH = "data_veu4/expanded_segan1_additive/noisy_testset/"
    TEST_FILES_PATH = [r"C:\Users\amitk\data_for_experiments\gan_denoising\my_test_inputs"]
    # please specify the output folder where cleaned files
    # will be saved
    # SAVE_PATH = "synth_segan+"
    SAVE_PATH = r"C:\Users\amitk\data_for_experiments\gan_denoising\my_test_outputs"
    g_pretrained_ckpt = os.path.join(CKPT_PATH, G_PRETRAINED_CKPT)
    test_files = TEST_FILES_PATH
    cfg_file = "train.opts"
    synthesis_path = SAVE_PATH
    seed = 111
    cuda = False
    if not os.path.exists(synthesis_path):
        pass
        # os.makedirs(synthesis_path)

    # seed initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    clean(cfg_file, test_files, g_pretrained_ckpt, cuda, synthesis_path)
    # if cuda:
    #     torch.cuda.manual_seed_all(seed)
