import sys
import os
import multiprocessing as mlp
from tqdm import tqdm
from apply_dsp_kws import apply_oners_kws, apply_onex2_kws
from wav_process import add_channel, split_channel, split_duration, compute_duration


def process_oners_kws(wav_path, out_dir):
    wav_name = os.path.basename(wav_path)
    new_wav_path = os.path.join('/tmp', wav_name)
    kws_wav_path = os.path.join('/tmp', wav_name.replace('.wav', '_kws_out.wav'))
    add_channel(wav_path, new_wav_path)
    apply_oners_kws(new_wav_path, kws_wav_path)
    split_channel(kws_wav_path, out_dir)
    os.remove(new_wav_path)
    os.remove(kws_wav_path)


def process_onex2_kws(wav_path, out_dir):
    wav_name = os.path.basename(wav_path)
    kws_wav_path = os.path.join('/tmp', wav_name.replace('.wav', '_kws_out.wav'))
    apply_onex2_kws(wav_path, kws_wav_path)
    split_channel(kws_wav_path, out_dir)
    os.remove(kws_wav_path)


def main():
    mode = sys.argv[1]
    wav_dir = sys.argv[2]
    if mode in ["oners", "onex2", "split"]:
        out_dir = sys.argv[3]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    tasks = []
    pool = mlp.Pool(processes=mlp.cpu_count())
    print("Preparing tasks .....")
    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                if mode == "oners":
                    tasks.append(pool.apply_async(process_oners_kws, (wav_path, out_dir)))
                elif mode == "onex2":
                    tasks.append(pool.apply_async(process_onex2_kws, (wav_path, out_dir)))
                elif mode == "split":
                    tasks.append(pool.apply_async(split_duration, (wav_path, 5, out_dir)))
                elif mode == "duration":
                    tasks.append(pool.apply_async(compute_duration, (wav_path, )))
                else:
                    print("Not support {} mode".format(mode))
    
    print("Doing process ......")
    if mode == "duration":
        all_duration = max_duration = 0
        min_duration = 1000
    for i in tqdm(range(len(tasks))):
        if mode == "duration":
            wav_path, duration = tasks[i].get()
            all_duration += duration
            if duration > max_duration:
                max_duration = duration
                max_wav_path = wav_path
            if duration < min_duration:
                min_duration = duration
                min_wav_path = wav_path
        else:
            tasks[i].get()

    if mode == "duration":
        print("All duration: {}".format(all_duration))
        print("Max duration: {} {}".format(max_duration, max_wav_path))
        print("Min duration: {} {}".format(min_duration, min_wav_path))


    print("Done")




if __name__ == "__main__":
    main()
