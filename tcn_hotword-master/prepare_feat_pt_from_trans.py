import os, sys
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from dataloader import wav_dataset, wav_Collate_fn
from filterbank_matrix import compute_fbank_matrix


def main():
    trans_in = sys.argv[1]
    trans_out = sys.argv[2]
    out_dir = sys.argv[3]
    has_boundary = int(sys.argv[4])

    has_boundary = has_boundary != 0
    out_dir = os.path.abspath(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fout = open(trans_out, 'w', encoding='utf-8')
    batch_size = 1024
    use_gpu = False
    dataset = wav_dataset(trans_in, has_boundary)  #return torch.from_numpy(np.array(audio)), text, begin, end, wav_path
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0, 
                            collate_fn=wav_Collate_fn, 
                            pin_memory=use_gpu, 
                            drop_last=False
    )
    total_batch = len(dataloader)
    start_time = time.time()
    
    for batch, batch_data in enumerate(dataloader):
        #print("*******************")
        if has_boundary:
            batch_audio, lengths, batch_text, batch_begin, batch_end, \
                batch_path = batch_data
        else:
            batch_audio, lengths, batch_text, batch_path = batch_data
        #print("processed one batch")
        X = compute_fbank_matrix(batch_audio, use_gpu=use_gpu)
        # 适用于10ms一帧
        # feat_lengths = torch.ceil(lengths / 160)
        feat_lengths = torch.floor(lengths / 160)
        block_file = os.path.join(out_dir, "block-%09d.pt" % batch)
        torch.save((X, feat_lengths), block_file)
        if has_boundary:
            fout.write(block_file + '\t' + ','.join(batch_text) + '\t' + 
                       ','.join(batch_begin) + '\t' + ','.join(batch_end)
                        + '\n')
        else:
            fout.write(block_file + '\t' + ','.join(batch_text) + '\n')
        cost_time = time.time() - start_time
        start_time = time.time()
        print(f'process {batch+1} batch cost time(s) {cost_time:>0.1f}, ' 
              + f'batch_size {len(batch_text)}, total batch {total_batch}, '
              + f'need {(total_batch-batch)*cost_time/60:>0.1f}min')
    fout.close()


if __name__ == "__main__":
    main()