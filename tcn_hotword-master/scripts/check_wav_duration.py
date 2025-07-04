#!/usr/bin/env python3
# encoding: utf-8

import sys
import os
import shutil
from tqdm import tqdm
import torchaudio
import multiprocessing as mlp
torchaudio.set_audio_backend("sox_io")


def process_line(line):
    items = line.strip().split()
    wav_path = items[0]
    waveform, rate = torchaudio.load(wav_path)
    assert rate == 16000
    duration = len(waveform[0]) / float(rate)
    if duration < 1.0 or duration > 5:
        return False, wav_path, duration
    else:
        return True, wav_path, duration


def main():
    trans = sys.argv[1]
    outdir = sys.argv[2]
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
            print(f"{wav_path} duration is {duration}")
            new_wav_path = os.path.join(outdir, os.path.basename(wav_path))
            shutil.copy(wav_path, new_wav_path)

if __name__ == "__main__":
    main()