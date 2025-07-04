import sys
import os
import multiprocessing as mlp
from tqdm import tqdm

from wav_process import split_duration

def main():
    trans_in = sys.argv[1]
    out_dir = sys.argv[2]
    
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trans_out = os.path.join(out_dir, "trans.txt")
    fout = open(trans_out, 'w', encoding='utf-8')

    tasks = []
    pool = mlp.Pool(mlp.cpu_count())
    print(f"Using {mlp.cpu_count()} cpus")
    print("Preparing tasks ...")
    with open(trans_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                file_path, text = line.strip().split('\t')
            else:
                file_path, text = line.strip().split(' ', 1)

            tasks.append(pool.apply_async(split_duration, (file_path, 5, out_dir)))
            
    print("Doing tasks ...")
    for i in tqdm(range(len(tasks))):
        new_path_list = tasks[i].get()
        if new_path_list:
            for p in new_path_list:
                fout.write(p + '\t' + "GARBAGE" + '\n')
    fout.close()
    print("Done")



if __name__ == "__main__":
    main()
