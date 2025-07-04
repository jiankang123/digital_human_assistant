import sys
import os
import multiprocessing as mlp
from tqdm import tqdm

from wav_process import split_duration

def main():
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    trans_out = os.path.join(out_dir, "trans.txt")
    fout = open(trans_out, 'w', encoding='utf-8')

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                new_path_list = split_duration(file_path, 5, out_dir)
                if new_path_list:
                    for p in new_path_list:
                        fout.write(p + '\t' + "GARBAGE" + '\n')
    
    fout.close()
    print("Done")



if __name__ == "__main__":
    main()
