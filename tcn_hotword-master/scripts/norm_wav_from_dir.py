import sys
import os
import multiprocessing as mlp
from tqdm import tqdm

from wav_process import normalize_wav

def main():
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                out_path = os.path.join(out_dir, os.path.basename(file_path))
                if not os.path.exists(out_path):
                    normalize_wav(file_path, out_path)
    print("Done")


if __name__ == "__main__":
    main()
