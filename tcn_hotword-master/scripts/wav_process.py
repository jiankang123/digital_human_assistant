import sys
import os
import random
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import subprocess


# sox归一化音量到指定DB，不会出现截辐
def norm_wav_by_sox(wav_path,out_path, norm_low, norm_high):
    norm_db = random.randint(norm_low,norm_high)
    ret_name = out_path.replace('.wav','_norm{}.wav'.format(norm_db))
    os.system('sox --norm={} {} {}'.format(norm_db, wav_path, ret_name))
    return ret_name

# sox最大化音量，不会出现截辐
def gain_wav_by_sox(wav_path,out_path):
    res = subprocess.Popen('sox {} -n stat -v'.format(wav_path),shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    # gain_persent = os.popen('python')
    result = res.stdout.readlines()
    gain_persent = float(result[0].strip().decode())
    ret_name = out_path.replace('.wav','_gain_{}.wav'.format(gain_persent))
    os.system('sox -v {} {} {}'.format(gain_persent, wav_path, ret_name))
    return ret_name

# +db,有可能出现截辐
def gain_wav(wav_path, gain_low, gain_high, out_path):
    voice_data = AudioSegment.from_wav(wav_path)
    gain_db = random.randint(gain_low,gain_high)
    voice_data = voice_data + gain_db
    ret_name = out_path.replace('.wav','_gain_{}.wav'.format(gain_db))
    voice_data.export(ret_name, format="wav")
    return ret_name


def pcm2wav(pcm_path, sample_width, frame_rate, channels):
    voice_data = AudioSegment.from_file(
        file=pcm_path, sample_width=sample_width, frame_rate=frame_rate,channels=channels,) 
    pcm_data = np.array(voice_data.get_array_of_samples())
    pcm_data = pcm_data/32768.0
    return pcm_data


def add_channel(wav_path, new_wav_path):
    sample_rate, wav_data = wavfile.read(wav_path)
    new_wav_data = np.concatenate((wav_data, wav_data[:, 2].reshape(-1, 1)), axis=1)
    wavfile.write(new_wav_path, sample_rate, new_wav_data)


def split_channel(wav_path, out_dir):
    sample_rate, wav_data = wavfile.read(wav_path)
    num_channel = wav_data.shape[1]
    wav_name = os.path.basename(wav_path)
    new_path_list = []
    for i in range(num_channel):
        new_wav_path = os.path.join(out_dir, wav_name.replace('.wav', '_channel{}.wav'.format(i)))
        new_path_list.append(new_wav_path)
        wavfile.write(new_wav_path, sample_rate, wav_data[:, i])
    return new_path_list


def split_duration(file_path, duration, out_dir):
    new_path_list = []

    file_name = os.path.basename(os.path.dirname(file_path)) + '_' + os.path.basename(file_path)

    audio = AudioSegment.from_wav(file_path)
    count = 0
    step = duration * 1000
    if len(audio) <= step:
        if len(audio) < 1000:
            return 
        segwav_path = os.path.join(out_dir, file_name.replace('.wav', '_seg000000001.wav'))
        audio.export(segwav_path, format="wav")
        new_path_list.append(segwav_path)
        return new_path_list
    while count * step < len(audio):
        if (count+1)*step < len(audio):
            tmp_audio = audio[count*step: (count+1)*step]
        else:
            tmp_audio = audio[-step:]
            if len(tmp_audio) < 1000:
                break
        out_path = os.path.join(out_dir, file_name.replace(".wav", "_seg%09d.wav" % count))
        tmp_audio.export(out_path, format="wav")
        new_path_list.append(out_path)
        count += 1 
    return new_path_list


def split_duration_with_overlap(file_path, duration, overlap, out_dir):
    new_path_list = []
    dir_name = os.path.basename(os.path.dirname(file_path))
    file_name = dir_name + '_' + os.path.basename(file_path)

    audio = AudioSegment.from_wav(file_path)
    count = 0
    step = duration * 1000
    if len(audio) <= step:
        if len(audio) < 1000:
            # short than 1 second, don't save
            return 
        segwav_path = os.path.join(out_dir, 
                                file_name.replace('.wav', '_seg000000001.wav'))
        audio.export(segwav_path, format="wav")
        new_path_list.append(segwav_path)
        return new_path_list
    while count * step < len(audio):
        if (count+1)*step < len(audio):
            tmp_audio = audio[count*step: (count+1)*step]
        else:
            tmp_audio = audio[-step:]
            if len(tmp_audio) < 1000:
                break
        out_path = os.path.join(out_dir, file_name.replace(".wav", "_seg%09d.wav" % count))
        tmp_audio.export(out_path, format="wav")
        new_path_list.append(out_path)
        count += 1 
    return new_path_list


def normalize_wav(file_path, out_path):
    file_format = file_path.lower().split('.')[-1]
    if not file_format == 'wav':
        out_path = '.'.join(out_path.split('.')[:-1]) + '.wav'
    audio = AudioSegment.from_file(file_path, file_format)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(out_path, format="wav")


def compute_duration(wav_path):
    sound= AudioSegment.from_wav(wav_path)
    duration = sound.duration_seconds
    return wav_path, duration


if __name__ == '__main__':
    wav_path = sys.argv[1]
    sum_num = 0
    for root,dirs,files in os.walk(wav_path):
        for f in files:
            if f.endswith('.wav'):
                _,dura = compute_duration(os.path.join(root,f))
                sum_num += dura
    print(sum_num)
    # gain_wav_by_sox(wav_path)
    # new_wav_path = sys.argv[2]
    # add_channel(wav_path, new_wav_path)