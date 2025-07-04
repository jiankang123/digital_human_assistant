import sys
import os
from scipy.io import wavfile
import numpy as np

def pad_wav(wav_path, new_wav_path):
    sample_rate, wav_data = wavfile.read(wav_path)
    new_wav_data = np.concatenate((wav_data, np.zeros(16000, dtype=np.int16)), axis=0)
    wavfile.write(new_wav_path, sample_rate, new_wav_data)


def main():
    trans_in = sys.argv[1]
    out_dir = sys.argv[2]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trans_out = os.path.join(out_dir, "trans.txt")
    fout = open(trans_out, 'w', encoding='utf-8')

    with open(trans_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                file_path, text = line.strip().split('\t')
            else:
                file_path, text = line.strip().split(' ', 1)
            out_path = os.path.join(out_dir, os.path.basename(file_path))
            pad_wav(file_path, out_path)
            fout.write(out_path + '\t' + text + '\n')
    fout.close()
    print("Done")


if __name__ == "__main__":
    main()