import os
import sys
import torch
import sys
sys.path.append("/ssd1/kai.zhou/workspace/hotword/torch/")
from scipy.io import wavfile
# from torch_filterbank import compute_torch_fbank
from filterbank_matrix import compute_fbank_matrix
import numpy as np

if __name__ == '__main__':
    sr,data = wavfile.read('/ssd1/kai.zhou/workspace/hotword/torch/feat/test.wav')
    data = torch.tensor(data,dtype=torch.float)
    fbank1 = compute_fbank_matrix(torch.stack([data,data],dim=0), use_gpu=True)
    fbank2 = compute_fbank_matrix(torch.stack([data,data],dim=0), use_gpu=False)
    np.save('/media/hdd2/qjh/torch2tf/wav_fbank.npy',np.array(fbank1.detach().cpu().numpy()))
    # fbank = compute_torch_fbank(data)
    # print(fbank1)
    # print(fbank2)
