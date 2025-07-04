import sys
import os
import multiprocessing as mlp
from tqdm import tqdm
from pydub import AudioSegment


def normalize_wav(file_path, out_path):
    file_format = file_path.lower().split('.')[-1]
    if not file_format == 'wav':
        out_path = '.'.join(out_path.split('.')[:-1]) + '.wav'
    audio = AudioSegment.from_file(file_path, file_format)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(out_path, format="wav")


def main():
    trans_in = sys.argv[1]
    out_dir = sys.argv[2]
    

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    trans_out = os.path.join(out_dir, "trans.txt")
    fout = open(trans_out, 'w', encoding='utf-8')

    with open(trans_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                if '\t' in line:
                    file_path, text = line.strip().split('\t')
                    #file_path = line.strip().split('\t')[0]
                    #text = line.strip().split('\t')[1]
                else:
                    file_path, text = line.strip().split(' ', 1)
                    #file_path = line.strip().split(' ')[0]
                    #text = line.strip().split(' ')[1]
            except:
                print(f"{line} error")
            out_path = os.path.join(out_dir, os.path.basename(file_path))
            if not os.path.exists(out_path):
                normalize_wav(file_path, out_path)
            fout.write(out_path + '\t' + text + '\n')
    fout.close()
    print("Done")



if __name__ == "__main__":
    main()
