import os
import sys
import io
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
from wav_process import *

from markerexporter import exportSampleOffset

class Cut_Wav_Make_Label():
    def __init__(self, out_dir, mode=None):
        self.out_dir = out_dir
        self.create_label_file(mode)
        self.cn_label_list = [
            '做个标记', '关闭相机', '开始录像', '停止录像', '拍张照片', 
            'mark that', 'shut down camera', 'start recording', 'stop recording', 'take a photo'
        ]
        self.en_label_list = [
            'mark that', 'shut down camera', 'start recording', 'stop recording', 'take a photo', 
            '11 segment'
        ]

    # 创建目录并写入
    def write_wav(self, index, sr, channel, wav_name, wav_frames, mode=None):
        # 写入音频文件
        # 把三通道变为四通道
        if channel == 3:
            wav_frames = np.concatenate((wav_frames,wav_frames[:,2].reshape(-1,1)), axis=1)

        path = os.path.join(self.out_dir, os.path.basename(wav_name)[:-4])
        if not os.path.isdir(path):
            os.mkdir(path)
        outname = os.path.join(path, os.path.basename(wav_name).split('.wav')[0]+ "{:03d}".format(index) + '.wav')
        # print(wav_frames.shape)
        wavfile.write(outname, sr, wav_frames.astype(np.int16))
        if mode == 'x3':
            write_dir = os.path.dirname(os.path.dirname(os.path.dirname(outname)))
            path_list = split_channel(outname,write_dir)
            self.write_x3_1c_trans.write('{}\t{}\n'.format(os.path.join(write_dir,os.path.basename(outname.replace('.wav','_channel0.wav'))),self.labels[index].strip()))
            self.write_x3_1c_trans.write('{}\t{}\n'.format(os.path.join(write_dir,os.path.basename(outname.replace('.wav','_channel1.wav'))),self.labels[index].strip()))
        # 写入label
        if self.labels:
            self.trans.write(os.path.abspath(outname) + '\t' + self.labels[index].strip()+'\n')
        else:
            label_list_index = index//2
            print('index',index)
            if label_list_index >= len(self.cn_label_list):
                label_list_index = label_list_index%len(self.cn_label_list)
            print('label_list_index',label_list_index)
            self.trans.write(os.path.abspath(outname) + '\t' + self.cn_label_list[label_list_index] + '\n')
            

    # 读取mark获得切分音频
    # 标注为wav同级目录下同名的txt文件
    def wav_split(self, wav_name, mode=None):
        print('process: ', wav_name)
        label_path = wav_name.replace('.wav', '.txt')
        if os.path.exists(label_path):
            self.labels = open(label_path, 'r', encoding='utf-8').readlines()
        else:
            self.labels = None
        cut_list = exportSampleOffset(wav_name)
        # cut_list = [24000,48000, 72000, 96000]
        if len(cut_list) == 0:
            print("{} has no marker".format(wav_name))
            return
        # print(cut_list)
        sr, data = wavfile.read(wav_name)
        channel = len(data[0])
        append_list = np.full_like(data,0)
        count = 0

        start = cut_list[0]
        for i in cut_list:
            if count == 0:
                count += 1
                continue
            end = i
            audio_dst = data[int(start): int(end)]
            # print(audio_dst.shape)
            # append_list = []

            # 尾部追加
            # audio_dst = np.concatenate((audio_dst,append_list[:sr]), axis=0)
            self.write_wav(count-1, sr, channel, wav_name, audio_dst, mode)
            count += 1
            start = end


    def create_label_file(self, mode=None):
        self.trans = open(os.path.join(self.out_dir, 'trans.txt'), 
                            'w', encoding='utf-8')
        if mode == 'x3':
            self.write_x3_1c_trans = open(os.path.join(os.path.dirname(self.out_dir),'trans.txt'),'w',encoding='utf-8')

    def close_label_file(self):
        self.trans.close()

    # 遍历音频目录
    def run(self, wav_dir):
        
        self.wav_dir = wav_dir
        self.create_label_file()
        for root, dirs, files in os.walk(self.wav_dir):
            for name in files:
                if name.endswith('.wav'):
                    if 'CHN' in root:
                        self.language_flag = 'cn'
                    else:
                        self.language_flag = 'en'
                    # self.cut_file = open(os.path.join(root, name.split('.wav')[0]+'.srt'),'r',encoding='utf-8').readlines()
                    self.wav_split(os.path.join(root, name))
        self.close_label_file()



# 读取wav文件并抛弃wav头，转成pcm文件
def drop_wav_head(input_dir, out_dir):
    if input_dir.endswith('.wav'):
        if not os.path.exists(out_dir):
            with open(input_dir,'rb') as f1:
                with open(out_dir,'wb') as f2:
                    count = 0
                    while True:
                        # 音频过大，分块读取和写入
                        strb = f1.read(1024)
                        if count == 0:
                            f2.write(strb[44:])
                            count += 1
                        else:
                            if strb == b"":
                                break
                            f2.write(strb)

            f1.close()
            f2.close()
            print('drop wav head and save pcm done')
            os.system('rm {}'.format(input_dir))


# 读取pcm文件, 采样宽度4字节是32bit，采样率48000，通道3
def pcm_read(out_pcm_path, sample_width=4, frame_rate=48000, channels=3):

    voice_data = AudioSegment.from_file(
        file=out_pcm_path, sample_width=sample_width, frame_rate=frame_rate, channels=channels,)
    
    print('pcm read done')
    return voice_data

def mk_1c_trans(path_4c, path_1c):
    lines = open(os.path.join(path_4c,'trans.txt'),'r',encoding='utf-8').readlines()
    w = open(os.path.join(path_1c,'trans.txt'),'w',encoding='utf-8')
    for line in lines:
        path, label = line.strip().split('\t')
        kws_path = path.replace('4cwav','1cwav')[:-4] + '_kws_out_channel0.wav'
        w.write('{}\t{}\n'.format(kws_path,label))
        kws_path = path.replace('4cwav','1cwav')[:-4] + '_kws_out_channel1.wav'
        w.write('{}\t{}\n'.format(kws_path,label))


def gain_channel2(voice_data):
    np_data = np.array(voice_data.get_array_of_samples()).reshape(-1, 4)
    gain_chan = np_data[:,0]
    wav_io = io.BytesIO()
    wavfile.write(wav_io, 48000, gain_chan)
    wav_io.seek(0)
    gain_channel0 = AudioSegment.from_wav(wav_io)


    gain_chan = np_data[:,1]
    wav_io = io.BytesIO()
    wavfile.write(wav_io, 48000, gain_chan)
    wav_io.seek(0)
    gain_channel1 = AudioSegment.from_wav(wav_io)

    gain_chan = np_data[:,2]
    wav_io = io.BytesIO()
    wavfile.write(wav_io, 48000, gain_chan)
    wav_io.seek(0)
    channel2 = AudioSegment.from_wav(wav_io)
    gain_channel2 = channel2.apply_gain(+9)

    gain_chan = np_data[:,3]
    wav_io = io.BytesIO()
    wavfile.write(wav_io, 48000, gain_chan)
    wav_io.seek(0)
    gain_channel3 = AudioSegment.from_wav(wav_io)

    pcm_merge = AudioSegment.from_mono_audiosegments(gain_channel0,gain_channel1,gain_channel2,gain_channel3)
    return pcm_merge

def comp(elem):
    seq = int(elem.split('.pcm')[0].split('view_')[1])
    return seq


def concat_x3_pcm(pcm_path):
    pcm_list = []
    total_pcm = []
    for root,dirs,files in os.walk(pcm_path):
        for f in files:
            if f.endswith('.PCM'):
                pcm_dir = os.path.join(pcm_path,f)
                pcm_list.append(pcm_dir)
    pcm_list.sort(key=comp)
    
    for i in pcm_list:
        voice_data = AudioSegment.from_file(file=i, sample_width=4, frame_rate=48000,channels=4,)
        total_pcm.append(voice_data)
        print('{} load'.format(i))
        print(voice_data.channels,voice_data.frame_rate,voice_data.duration_seconds)
    ret = total_pcm[0]
    for index,i in enumerate(total_pcm):
        if index == 0:
            continue
        else:
            ret += i

    print(ret.channels,ret.frame_rate,ret.duration_seconds)
    return ret