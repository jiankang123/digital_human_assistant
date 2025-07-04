import os, sys
from tqdm import tqdm
import shutil

def main():
    trans_in = sys.argv[1]
    out_dir = sys.argv[3]
    hotwords = sys.argv[2].split(',')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trans_out = open(os.path.join(out_dir, "trans.txt"), 'w', encoding='utf-8')
    trans_outs = {}
    for hotword in hotwords:
        hotword_dir = os.path.join(out_dir, hotword)
        if not os.path.exists(hotword_dir):
            os.makedirs(hotword_dir)
        trans_outs[hotword] = open(os.path.join(hotword_dir, "trans.txt"),
                                   'w', encoding='utf-8')
    hotword_dir = os.path.join(out_dir, "GARBAGE")
    if not os.path.exists(hotword_dir):
        os.makedirs(hotword_dir)
    trans_outs["GARBAGE"] = open(os.path.join(hotword_dir, "trans.txt"),
                                   'w', encoding='utf-8')
    
    with open(trans_in, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    
    for i in tqdm(range(len(lines))):
        line = lines[i]
        wav_path, text = line.strip().split('\t')
        if not text in hotwords:
            text = "GARBAGE"
        target_dir = os.path.join(out_dir, text)
        wav_name = os.path.basename(wav_path)
        new_path = os.path.join(target_dir, wav_name)
        shutil.copy(wav_path, new_path)
        trans_out.write(new_path + "\t" + text + "\n")
        trans_outs[text].write(new_path + "\t" + text + "\n")
    
    trans_out.close()
    for hotword in hotwords:
        trans_outs[hotword].close()
        
if __name__ == "__main__":
    main()