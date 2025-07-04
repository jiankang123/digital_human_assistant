#!/usr/bin/env python3
# encoding: utf-8

import sys
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
    new_line = wav_path + '\t' + str(duration) + '\n'
    return new_line, duration


def main():
    trans = sys.argv[1]
    dur_scp = sys.argv[2]

    tasks = []
    pool = mlp.Pool(mlp.cpu_count())
    fout = open(dur_scp, 'w', encoding='utf-8')
    with open(trans, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tasks.append(pool.apply_async(process_line, (line,)))

    total_duration = 0
    for i in tqdm(range(len(tasks))):
        new_line, duration = tasks[i].get()
        total_duration += duration
        fout.write(new_line)
    fout.close()
    print('process {} utts'.format(len(tasks)))
    print('total {} s'.format(total_duration))
    print('total %.2f h' % (total_duration/3600))

if __name__ == "__main__":
    main()