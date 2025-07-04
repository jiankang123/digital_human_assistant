import sys
import os
import torch
import json

def main():
    trans = sys.argv[1]
    cmvn_json = sys.argv[2]
    if len(sys.argv) > 3:
        feat_dim=int(sys.argv[3])
    else:
        feat_dim=23
    acc_mean = torch.zeros(feat_dim)
    acc_scale = torch.zeros(feat_dim)
    num_frames = 0
    num_utts = 0
    count = 0
    with open(trans, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                pt_path = line.split('\t')[0]
            else:
                pt_path = line.split(' ')[0]
            temp = torch.load(pt_path)
            if len(temp) == 2:
                fbank, seq_len = torch.load(pt_path)
            else:
                fbank = torch.load(pt_path)
                seq_len = torch.tensor(list(map(int,line.strip().split('\t')[-1].split(','))))
            num_utts += fbank.shape[0]
            num_frames += seq_len.sum()
            acc_mean += fbank.sum(1).sum(0)
            acc_scale += torch.square(fbank).sum(1).sum(0)
            count += 1
            if count % 1000 == 0:
                print(f"processed {count}")
        
    mean = acc_mean / num_frames
    scale = 1. / torch.sqrt(acc_scale / num_frames - torch.square(mean))
    stat = {
        'utterances': int(num_utts),
        'frames': int(num_frames),
        'mean': mean.tolist(),
        'scale': scale.tolist()
    }
    with open(cmvn_json, 'w', encoding='utf-8') as f:
        json.dump(stat, f, indent=4)


if __name__ == "__main__":
    main()