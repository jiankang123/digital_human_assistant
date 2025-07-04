import os, sys
sys.path.append('/home/kai.zhou2221/workspace/hotword/speech/hotword/bazel-bin')
sys.path.append('/home/kai.zhou2221/workspace/hotword/speech/hotword')
import torch
import json
import numpy as np
import tcn_sep
from hotword.train.feature import FilterbankExtractor
from wav_test import wav_test, get_CMVN


def main():
    wav_dir_path = sys.argv[1]
    model = sys.argv[2]
    cmvn = sys.argv[3]
    
    net = torch.load(model, map_location='cpu').eval()
    try:
        net = net.module
    except:
        pass
    mean, scale = get_CMVN(cmvn)
    filterbank_extractor = FilterbankExtractor()
    pic_dir = os.path.join(wav_dir_path, os.path.basename(model)+'_wav_infer_figs')
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    for root, dirs, files in os.walk(wav_dir_path):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                wav_test(wav_path, net, mean, scale, filterbank_extractor, pic_dir)
    

if __name__ == "__main__":
    main()