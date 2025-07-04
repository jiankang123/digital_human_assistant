import json
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans_in", default="")
    parser.add_argument("--trans_out", default="")
    args = parser.parse_args()

    with open(args.trans_in, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    fout = open(args.trans_out, 'w', encoding='utf-8')
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if '\t' in line:
            wav_path, text = line.strip().split('\t')
        else:
            wav_path, text = line.strip().split(' ', 1)
        
        fout.write(wav_path + '\t' + text + '\t0\t0\n')
        
    fout.close()


if __name__ == "__main__":
    main()