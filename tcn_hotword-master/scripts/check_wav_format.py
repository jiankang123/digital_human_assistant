import sys
import os
import multiprocessing as mlp
from tqdm import tqdm
from pydub import AudioSegment


def check_wav_format(wav_path):
    file_format = wav_path.lower().split('.')[-1]
    if not file_format == 'wav':
        return False
    audio = AudioSegment.from_file(wav_path, file_format)
    if audio.frame_rate != 16000:
        return False
    if audio.sample_width != 2:
        return False
    if audio.channels != 1:
        return False
    return True


def main():
    trans = sys.argv[1]
    
    count = 0
    with open(trans, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                file_path, text = line.strip().split('\t')
            else:
                file_path, text = line.strip().split(' ', 1)
            if not check_wav_format(file_path):
                print(f"{file_path} has format problem")
                count += 1

    print(f"All done, problem nums: {count}")


if __name__ == "__main__":
    main()