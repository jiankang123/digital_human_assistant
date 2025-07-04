import os, sys
import numpy as np
import torch
import time
import random
import multiprocessing as mlp
from tqdm import tqdm
from feat_cpp.feature import FilterbankExtractor


def process_lines(index, lines, batch_size, out_dir, has_boundary):
    filterbank_extractor = FilterbankExtractor()
    re_str = ""
    batch_features = []
    batch_lens = []
    batch_texts = []
    batch_begins = []
    batch_ends = []
    batch_max_len = 0
    batch = 1
    for i in tqdm(range(len(lines)), desc="task_{}".format(index) , position=index+1):
        line = lines[i]
        wav_path, other = line.strip().split('\t', 1)
        feature = filterbank_extractor.extract(wav_path)
        batch_features.append(feature)
        batch_lens.append(feature.shape[0])
        batch_max_len = max(feature.shape[0], batch_max_len)
        if has_boundary:
            arr = other.split('\t')
            if len(arr) == 3:
                text, begin, end = arr
            else:
                text, begin, end = arr[0], "0", "0"
            batch_texts.append(text)
            batch_begins.append(begin)
            batch_ends.append(end)
        else:
            text = other
        if len(batch_features) == batch_size:
            batch_feature = torch.zeros((batch_size, batch_max_len, 
                                        feature.shape[1]))
            for i, feature in enumerate(batch_features):
                batch_feature[i][:batch_lens[i]] = torch.tensor(feature)
            feat_lengths = torch.tensor(batch_lens)
            block_file = os.path.join(out_dir, "block-%02d-%09d.pt" % (index, batch))
            torch.save((batch_feature, feat_lengths), block_file)
            if has_boundary:
                re_str += (block_file + '\t' + ','.join(batch_texts) + '\t' + 
                        ','.join(batch_begins) + '\t' + ','.join(batch_ends)
                            + '\n')
            else:
                re_str += (block_file + '\t' + ','.join(batch_texts) + '\n')
            batch_features = []
            batch_lens = []
            batch_texts = []
            batch_begins = []
            batch_ends = []
            batch_max_len = 0
            batch += 1
    
    if len(batch_features) != 0:
        batch_feature = torch.zeros((len(batch_features), batch_max_len, 
                                        feature.shape[1]))
        for i, feature in enumerate(batch_features):
            batch_feature[i][:batch_lens[i]] = torch.tensor(feature)
        feat_lengths = torch.tensor(batch_lens)
        block_file = os.path.join(out_dir, "block-%02d-%09d.pt" % (index, batch))
        torch.save((batch_feature, feat_lengths), block_file)
        if has_boundary:
            re_str += (block_file + '\t' + ','.join(batch_texts) + '\t' + 
                    ','.join(batch_begins) + '\t' + ','.join(batch_ends)
                        + '\n')
        else:
            re_str += (block_file + '\t' + ','.join(batch_texts) + '\n')
    return re_str


def main():
    trans_in = sys.argv[1]
    trans_out = sys.argv[2]
    out_dir = sys.argv[3]
    has_boundary = int(sys.argv[4])

    has_boundary = has_boundary != 0
    out_dir = os.path.abspath(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(trans_in, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    random.shuffle(lines)

    num_process = 50
    tasks = []
    pool = mlp.Pool(num_process)
    random.shuffle(lines)
    batch_size = 1024

    total_batch = len(lines) // batch_size 
    if len(lines) % batch_size:
        total_batch += 1

    block_num_per_task = total_batch // num_process + 1
    line_num_per_task = block_num_per_task * batch_size

    print(f"all samples num: {len(lines)}, batch num: {total_batch}, ")
    print(f"block_num_per_task: {block_num_per_task}, ")
    print(f"line_num_per_task: {line_num_per_task}")

    for i in range(num_process):
        if (i+1)*line_num_per_task <= len(lines):
            tasks.append(
                pool.apply_async(process_lines, (i, 
                    lines[i*line_num_per_task: (i+1)*line_num_per_task], 
                    batch_size, out_dir, has_boundary))
            )
        else:
            tasks.append(
                pool.apply_async(process_lines, (i, lines[i*line_num_per_task:],
                    batch_size, out_dir, has_boundary))
            )

    print("Doing tasks ...")
    fout = open(trans_out, 'w', encoding='utf-8')
    for i in range(len(tasks)):
        fout.write(tasks[i].get())
    fout.close()

if __name__ == "__main__":
    main()