import sys, os
import torch
import time
import numpy as np
from tqdm import tqdm
from feat import compute_fbank_torch, get_CMVN
from dataloader import trans_test_dataset, trans_test_Collate_fn
from filterbank_matrix import compute_fbank_matrix
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

# @profile
def main():
    """
    python trans_test.py 
    
    """
    net_path = sys.argv[1]
    cmvn = sys.argv[2]
    hotwords_str = sys.argv[3]
    trans_path = sys.argv[4]
    
    use_gpu = True
    out_dir = net_path + trans_path.replace("/", "_")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 将热词组织成字典，key为热词，value为类别
    hotwords_dict = {}
    hotwords = ["GARBAGE"] + hotwords_str.split(',')
    num_classes = len(hotwords)
    for index, hotword in enumerate(hotwords):
        if '/' in hotword:
            hotword_arr = hotword.split('/')
            for h in hotword_arr:
                hotwords_dict[h.upper()] = index
        else:
            hotwords_dict[hotword.upper()] = index
    print("hotwords:", hotwords_dict)

    confident_ark = open(os.path.join(out_dir, 'confidence.ark'), 'w', 
                         encoding='utf-8')
    net = torch.load(net_path, map_location='cpu').eval()
    mean, scale = get_CMVN(cmvn)
    if use_gpu:
        net = net.cuda()
        mean, scale = mean.cuda(), scale.cuda()
    test_dataset = trans_test_dataset(trans_path, hotwords_dict)
    dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=trans_test_Collate_fn, 
        pin_memory=True, 
        drop_last=False
    )
    
    with torch.no_grad():
        for index, (audio, lengths, label, path) in enumerate(tqdm(dataloader)):
            # compute fbank
            fbank = compute_fbank_matrix(audio, use_gpu=use_gpu)
            # fbank计算函数，outshape: 512*padded_length*23
            batch_size, length, nmel = fbank.shape
            lengths = [int(torch.ceil(l/160)) for l in lengths]
            # CMVN
            fbank = (fbank - mean) * scale
            pred = net(fbank)
            if use_gpu:
                pred = pred.cpu()
            pred = pred.detach().numpy()
            # pred = torch.round(pred, decimals=4).detach().numpy()
            for wav_index in range(batch_size):
                pred_per_wav = pred[wav_index,:lengths[wav_index]]
                path_per_wav = path[wav_index]
                name_per_wav = os.path.basename(path_per_wav).replace('.wav', '')
                # label_ = int(label[wav_index].item())
                for r in range(1, pred_per_wav.shape[1]):
                    pred_tmp = ['%.4f'%x for x in pred_per_wav[:,r]]
                    confident_ark.write(name_per_wav + ' ' + str(r) + ' '
                                        + ' '.join(pred_tmp)+'\n')

    confident_ark.close()

    print("done")


if __name__ == "__main__":
    main()
