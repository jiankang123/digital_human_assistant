import os
import sys
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from tqdm import tqdm
from pydub import AudioSegment

def comp(elem):
    seq = int(elem.split('.pcm')[0].split('view_')[1])
    return seq


def wav2c_concat_and_split(wav_dir_in, wav_dir_out, trans_dur_path, trans_out_path, offset):
    lines = open(trans_dur_path, 'r', encoding='utf-8').readlines()
    # offset = 32.8 * 1000 # clean
    # offset = 39 * 1000      # office
    # offset = 20.5 * 1000      # riverside
    # offset = 36.9 * 1000      # street
    wav2c_list = []
    wav_index = 0
    for root, dirs, files in os.walk(wav_dir_in):
        for file in files:
            if file.endswith(".wav"):
                wav2c_list.append(os.path.join(root, file))
    wav2c_list.sort(key=comp)
    print(f"get {len(wav2c_list)} wavs")
    fout = open(trans_out_path, 'w', encoding='utf-8')
    for i in tqdm(range(len(lines))):
        line = lines[i]
        wav_path, text, dur = line.strip().split('\t')
        duration = float(dur)
        wav_name = os.path.basename(wav_path)
        new_path = os.path.join(wav_dir_out, wav_name)
        audio = AudioSegment.from_wav(wav2c_list[wav_index])
        if offset + duration*1000 > len(audio):
            wav_index += 1
            tmp_audio = audio[offset:]
            offset = 0
            audio = AudioSegment.from_wav(wav2c_list[wav_index])
            new_audio = tmp_audio + audio[offset: offset+duration*1000-len(tmp_audio)]
            offset += duration*1000-len(tmp_audio)
        elif offset + duration*1000 == len(audio):
            new_audio = audio[offset:]
            wav_index += 1
            offset = 0
            audio = AudioSegment.from_wav(wav2c_list[wav_index])
        else:
            new_audio = audio[offset: offset+duration*1000]
            offset += duration*1000
        
        new_audios =  new_audio.split_to_mono()
        for i, au in enumerate(new_audios):
            
            au.export(new_path.replace(".wav", "_{}.wav".format(i)), format="wav")
            fout.write(new_path.replace(".wav", "_{}.wav".format(i)) + '\t' + text + '\n')



if __name__ == "__main__":
    wav_dir_in = sys.argv[1]
    wav_dir_out = sys.argv[2]
    trans_dur_path = sys.argv[3]
    trans_out_path = sys.argv[4]
    offset = float(sys.argv[5])
    wav2c_concat_and_split(wav_dir_in, wav_dir_out, trans_dur_path, trans_out_path, offset)