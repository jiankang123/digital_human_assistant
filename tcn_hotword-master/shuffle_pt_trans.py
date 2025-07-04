import numpy as np
import torch
import os
import sys
import random

def shuffle_chunk_batch(chunk_line, batch_size, out_dir):
    trans_content = ''

    chunk_list = []
    for line in chunk_line:
        arr = line.strip().split('\t')
        assert len(arr) == 4
        ptfile, text, begin, end = arr
        fbank, seq_len = torch.load(ptfile)
        text_list = text.split(',')
        begin_list = begin.split(',')
        end_list = end.split(',')
        for i in range(fbank.shape[0]):
            chunk_data = (fbank[i][:int(seq_len[i].item()),:], 
                          seq_len[i], text_list[i], 
                          begin_list[i], end_list[i])
            chunk_list.append(chunk_data)

    random.shuffle(chunk_list)
    batch_count = 0
    while batch_count * batch_size < len(chunk_list):
        batch_len = []
        batch_text = []
        batch_begin = []
        batch_end = []
        if batch_count * batch_size + batch_size < len(chunk_list):
            real_batch_size = batch_size
        else:
            real_batch_size = len(chunk_list) - batch_count * batch_size

        for i in range(real_batch_size):
            chunk_data = chunk_list[batch_count * batch_size + i]
            batch_len.append(chunk_data[1])
            batch_text.append(chunk_data[2])
            batch_begin.append(chunk_data[3])
            batch_end.append(chunk_data[4])
        batch_feat = torch.zeros((real_batch_size, int(max(batch_len).item()),
                                  chunk_data[0].shape[-1]))
        for i in range(real_batch_size):
            chunk_data = chunk_list[batch_count * batch_size + i]
            batch_feat[i][:int(batch_len[i].item()), :] = chunk_data[0]
        
        # save pt file
        assert batch_feat.shape[0] == len(batch_text)
        assert batch_feat.shape[0] == len(batch_begin)
        assert batch_feat.shape[0] == len(batch_end)
        assert batch_feat.shape[0] == len(batch_len)

        block_file = os.path.join(out_dir, "block-%09d.pt" % batch_count)
        torch.save((batch_feat, torch.tensor(batch_len)), block_file)
        trans_content += (block_file + '\t' + 
                          ','.join(batch_text) + '\t' + 
                          ','.join(batch_begin) + '\t' + 
                          ','.join(batch_end) + '\n')
        batch_count += 1

    return trans_content

def main():
    transes = sys.argv[1]
    trans_out = sys.argv[2]
    out_dir = sys.argv[3]
    batch_size = int(sys.argv[4])

    out_dir = os.path.abspath(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fout = open(trans_out, 'w', encoding='utf-8')

    
    chunk_size = 1000
    trans_list = transes.split(',')
    lines = []
    for trans in trans_list:
        with open(trans, 'r', encoding='utf-8') as fin:
            lines.extend(fin.readlines())
    random.shuffle(lines)

    chunk_count = 0
    while chunk_count * chunk_size < len(lines):
        if chunk_count * chunk_size + chunk_size < len(lines):
            real_chunk_size = chunk_size
        else:
            real_chunk_size = len(lines) - chunk_count * chunk_size

        chunk_line = lines[chunk_count * chunk_size: 
                           chunk_count * chunk_size + real_chunk_size]
        trans_content = shuffle_chunk_batch(chunk_line, batch_size, out_dir)
        fout.write(trans_content)
        chunk_count += 1
        print(f"processd {real_chunk_size} batch ")
    fout.close()
    print("Done")

if __name__ == "__main__":
    main()