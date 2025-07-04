import os
import sys
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import subprocess
from tqdm import tqdm
from pydub import AudioSegment
"""
将4通道48k32bit的wav,过dsp的信号处理转换为两个单通道16k16bit的wav。
用于python测试以及ROC计算
"""


def apply_oners_kws(wav_in, wav_kws):
    exe_path = "/ssd1/kai.zhou/workspace/dsp/sdk/old_build.embedded_linux-x86_64-oners.Release"
    cmd = "{}/mobvoi_dsp_insta360_demo 3 1 0 17 0 0 0 {} 1 {}".format(exe_path, wav_in, wav_kws)
    subprocess.check_output(cmd, shell=True)


def apply_onex2_kws(wav_in, wav_kws):
    exe_path = "/ssd1/kai.zhou/workspace/dsp/sdk/old_build.embedded_linux-x86_64-onex2.Release"
    cmd = "{}/mobvoi_dsp_insta360_demo 1 1 0 16 0 0 0 {} 1 {}".format(exe_path, wav_in, wav_kws)
    subprocess.check_output(cmd, shell=True)


def apply_onex3_kws(wav_in, wav_kws, mic_type):
    exe_path = "/ssd1/kai.zhou/workspace/dsp/sdk/old_build.embedded_linux-x86_64-onex3.Release"
    cmd = "{}/mobvoi_dsp_insta360_demo 4 1 0 16 0 {} 0 {} 1 {}".format(exe_path, mic_type, wav_in, wav_kws)
    subprocess.check_output(cmd, shell=True)


def process_oners_kws(wav_path, out_dir):
    wav_name = os.path.basename(wav_path)
    kws_wav_path = os.path.join('/tmp', wav_name.replace('.wav', '_kws_out.wav'))
    apply_oners_kws(wav_path, kws_wav_path)
    wav_paths = split_channel(kws_wav_path, out_dir)
    os.remove(kws_wav_path)
    return wav_paths


def process_onex2_kws(wav_path, out_dir, label=None):
    wav_name = os.path.basename(wav_path)
    kws_wav_path = os.path.join('/tmp', wav_name.replace('.wav', '_kws_out.wav'))
    apply_onex2_kws(wav_path, kws_wav_path)
    wav_paths = split_channel(kws_wav_path, out_dir)
    os.remove(kws_wav_path)
    return wav_paths

def process_onex3_kws(wav_path, out_dir, label=None):
    wav_name = os.path.basename(wav_path)
    kws_wav_path = os.path.join('/tmp', wav_name.replace('.wav', '_kws_out.wav'))
    apply_onex3_kws(wav_path, kws_wav_path)
    wav_paths = split_channel(kws_wav_path, out_dir)
    os.remove(kws_wav_path)
    return wav_paths


def split_channel(wav_path, out_dir):
    wav_names = []
    sample_rate, wav_data = wavfile.read(wav_path)
    num_channel = wav_data.shape[1]
    wav_name = os.path.basename(wav_path)
    for i in range(num_channel):
        new_wav_path = os.path.join(out_dir, wav_name.replace('.wav', '_channel{}.wav'.format(i)))
        wavfile.write(new_wav_path, sample_rate, wav_data[:, i])
        wav_names.append(new_wav_path)
    return wav_names


def wav2c_concat_and_split(wav_dir_in, wav_dir_out, trans_dur_path):
    lines = open(trans_dur_path, 'r', encoding='utf-8').readlines()
    offset = 0
    wav2c_list = []
    wav_index = 0
    for root, dirs, files in os.walk(wav_dir_in):
        for file in files:
            if file.endswith(".wav"):
                wav2c_list.append(os.path.join(root, file))
    fout = open(trans_dur_path, 'w', encoding='utf-8')
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
            new_audio = np.concatenate(tmp_audio, audio[offset: offset+duration*1000-len(tmp_audio)])
            offset += duration*1000-len(tmp_audio)
        elif offset + duration*1000 == len(audio):
            new_audio = audio[offset:]
            wav_index += 1
            offset = 0
            audio = AudioSegment.from_wav(wav2c_list[wav_index])
        else:
            new_audio = audio[offset: offset+duration*1000]
            offset += duration*1000
        new_audio.export(new_path, format="wav")
        fout.write(new_path + '\t' + text + '\n')




if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    mode = sys.argv[3]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                if mode == "oners":
                    process_oners_kws(wav_path, out_dir)
                elif mode == "onex2":
                    process_onex2_kws(wav_path, out_dir)
                elif mode == "onex3":
                    process_onex3_kws(wav_path, out_dir)
                else:
                    print("error mode ")
                    exit(0)


