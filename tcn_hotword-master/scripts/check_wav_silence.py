#!/usr/bin/env python3
# encoding: utf-8

import sys
import os
from tqdm import tqdm
import webrtcvad
import shutil
import numpy as np
from get_wav_vad import get_vad_result
import multiprocessing as mlp

sample_rate = 16000


def check_silence(vad_result):
    vad_result = np.array(vad_result)
    if vad_result.sum() < 5:
        return False, "blank"
    if vad_result[:10].sum() / 10 > 0.5:
        if vad_result[:20].sum() / 20 > 0.5:
            return False, "begin"
    if vad_result[-10:].sum() / 10 > 0.7:
        return False, "end"
    return True, "valid"


# def process_line(line):
#     items = line.strip().split()
#     wav_path = items[0]
#     waveform, rate = torchaudio.load(wav_path)
#     assert rate == 16000
#     duration = len(waveform[0]) / float(rate)
#     if duration < 1.0 or duration > 10:
#         return False, wav_path, duration
#     else:
#         return True, wav_path, duration

def main():
    trans = sys.argv[1]
    outdir = sys.argv[2]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # tasks = []
    # pool = mlp.Pool(mlp.cpu_count())
    count = 0
    with open(trans, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split()
            wav_path = items[0]
            vad_result = get_vad_result(3, wav_path)
            valid, error_type = check_silence(vad_result)
            if not valid:
                print(f"{wav_path} has {error_type} probem")
                print(vad_result)
                count += 1
                new_wav_path = os.path.join(outdir, os.path.basename(wav_path))
                shutil.copy(wav_path, new_wav_path)
    
    print(f"problem wav num {count}!")
    #         tasks.append(pool.apply_async(process_line, (line,)))

    # for i in tqdm(range(len(tasks))):
    #     valid, wav_path, duration = tasks[i].get()
    #     if not valid:
    #         print(f"{wav_path} duration is {duration}")

if __name__ == "__main__":
    main()

