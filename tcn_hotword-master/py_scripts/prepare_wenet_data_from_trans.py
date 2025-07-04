import json
import os
import argparse
from tqdm import tqdm
import torchaudio
torchaudio.set_audio_backend("sox_io")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trans", default="")
    parser.add_argument("--datalist", default="")
    args = parser.parse_args()

    with open(args.trans, 'r', encoding='utf-8') as fin:
      
        lines = fin.readlines()
    fout = open(args.datalist, 'w', encoding='utf-8')
    for i in tqdm(range(len(lines))):
        line = lines[i]
        wav_path = line.strip().split('\t')[0]
        text = line.strip().split('\t')[1]
        key, ext = os.path.splitext(os.path.basename(wav_path))
        waveform, rate = torchaudio.load(wav_path)
        assert rate == 16000
        duration = len(waveform[0]) / float(rate)
        line_dict = {
            "key": key,
            "wav": wav_path,
            "txt": text,
            "duration": duration
        }
        fout.write(json.dumps(line_dict, ensure_ascii=False)+'\n')
    fout.close()


if __name__ == "__main__":
    main()
