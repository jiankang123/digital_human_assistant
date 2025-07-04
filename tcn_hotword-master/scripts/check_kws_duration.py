#!/usr/bin/env python3
# encoding: utf-8

import sys
import os
from tqdm import tqdm
import shutil
import multiprocessing as mlp


def process_line(line):
    wav_path, label, begin, end = line.strip().split('\t')
    duration = int(end) - int(begin)
    if duration > 170:
        return False, wav_path, duration
    else:
        return True, wav_path, duration


def main():
    trans = sys.argv[1]

    outdir = trans+'_kws_duration_problem'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tasks = []
    pool = mlp.Pool(mlp.cpu_count())
    with open(trans, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tasks.append(pool.apply_async(process_line, (line,)))

    for i in tqdm(range(len(tasks))):
        valid, wav_path, duration = tasks[i].get()
        if not valid:
            print(f"{wav_path} kws duration is {duration}")
            newpath = os.path.join(outdir, os.path.basename(wav_path))
            shutil.copy(wav_path, newpath)


if __name__ == "__main__":
    main()