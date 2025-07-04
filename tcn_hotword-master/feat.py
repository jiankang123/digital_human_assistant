import os
import sys
import subprocess
import numpy as np
import torch
import json

# from filterbank_matrix import compute_fbank_matrix
# from dataloader import trans_test_dataset, trans_test_Collate_fn

FEAT_DIM = 23


def compute_fbank(wav_path):
    exe_file = "/ssd/nfs06/kai.zhou2221/workspace/hotword_torch/feat/compute_fbank_main"
    feat_path = "/tmp/tmp_feat.bin"
    cmd = exe_file + " " + wav_path + " " + feat_path
    subprocess.check_output(cmd, shell=True)
    np_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, FEAT_DIM)
    os.remove(feat_path)
    return np_feat


# def compute_fbank_torch(wav_path_list, label_list):
#     batch_fbank_list = []
#     batch_lengths_list = []
#     batch_label_list = []
    
#     test_dataset = trans_test_dataset(wav_path_list, label_list)
#     dataloader = torch.utils.data.DataLoader(
#         dataset=test_dataset, 
#         batch_size=512, 
#         shuffle=False, 
#         sampler=None, 
#         batch_sampler=None, 
#         num_workers=6, 
#         collate_fn=trans_test_Collate_fn, 
#         pin_memory=False, 
#         drop_last=False
#     )
#     for data,lengths,label in dataloader:
#         fbank = compute_fbank_matrix(data)
#         batch_fbank_list.append(fbank)
#         batch_lengths_list.append(lengths)
#         batch_label_list.append(label)
#     return batch_fbank_list,batch_lengths_list,batch_label_list
    

def get_CMVN(CMVN_json):
    with open(CMVN_json) as f:
        stat = json.load(f)
        mean = torch.tensor(np.array(stat['mean']), dtype=torch.float32)
        scale = torch.tensor(np.array(stat['scale']), dtype=torch.float32)
    return mean, scale