import sys
import os
import subprocess
import multiprocessing as mlp
from tqdm import tqdm

def process_line(line, outdir):
    wav_path, other = line.split('\t', 1)
    wav_name = os.path.basename(wav_path)
    tmp_path = os.path.join(outdir, wav_name.replace('.wav', '_-1.ogg'))
    new_wav_path = os.path.join(outdir, wav_name.replace('.wav', '_ogg-1.wav'))
    cmd1 = f"sox {wav_path} -t ogg -C -1 {tmp_path}"
    cmd2 = f"sox {tmp_path} -t wav {new_wav_path}"
    subprocess.check_output(cmd1, shell=True)
    subprocess.check_output(cmd2, shell=True)
    os.remove(tmp_path)
    return new_wav_path+'\t'+other

def main():
    trans_in = sys.argv[1]
    outdir = sys.argv[2]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    trans_out = os.path.join(outdir, 'trans.txt')
    tasks = []
    pool = mlp.Pool(mlp.cpu_count())
    with open(trans_in, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    print("Preparing tasks ......")
    for line in lines:
        tasks.append(pool.apply_async(process_line, (line, outdir)))
    
    print("Doing tasks ......")
    with open(trans_out, 'w', encoding='utf-8') as fout:
        for i in tqdm(range(len(tasks))):
            fout.write(tasks[i].get())
    print("Done")


if __name__ == "__main__":
    main()