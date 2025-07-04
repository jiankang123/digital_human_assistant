import sys
import os
import multiprocessing as mlp
from tqdm import tqdm

from wav_process import split_channel

def main():
    trans_in = sys.argv[1]
    out_dir = sys.argv[2]
    trans_out = sys.argv[3]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fout = open(trans_out, 'w', encoding='utf-8')

    with open(trans_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                file_path, text = line.strip().split('\t')
            else:
                file_path, text = line.strip().split(' ', 1)
            new_path_list = split_channel(file_path, out_dir)
            for p in new_path_list:
                fout.write(p + '\t' + text + '\n')
    
    fout.close()
    print("Done")



if __name__ == "__main__":
    main()
