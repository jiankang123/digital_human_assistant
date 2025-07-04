import os, sys
import multiprocessing as mlp
import subprocess
from tqdm import tqdm 

def fun(cmd):
    subprocess.check_output(cmd, shell=True, stderr=subprocess.PIPE)


def main():
    test_trans = sys.argv[1]
    out_dir = sys.argv[2]
    exe_path = sys.argv[3]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tasks = []
    pool = mlp.Pool(24)

    with open(test_trans, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    for i in range(len(lines)):
        line = lines[i]
        wav_path = line.split('\t')[0]
        score_path = os.path.join(out_dir, os.path.basename(wav_path)[:-4]+'.txt')
        cmd = '{} {} > {}'.format(exe_path, wav_path, score_path)
        tasks.append(pool.apply_async(fun, (cmd,)))


    for i in tqdm(range(len(tasks))):
        tasks[i].get()

if __name__ == "__main__":
    main()