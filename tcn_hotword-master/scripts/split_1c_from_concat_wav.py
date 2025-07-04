import os
import sys
from scipy.io import wavfile
import numpy as np
from split_insta_wav import *

'''
    从拼接的方法录制的音频中切分出单个音频
    params:
    - concat_wav_path:  path/to/concat.wav
        从相机中拿到的原始音频位置, 是一个.wav文件
    - out_pcm_path:     path/to/out.pcm
        输出的pcm文件位置, 一般输出名为"日期_备注.pcm"
    - out_path:         path/to/output/
        输出目录, 会创建两个子目录4cwav和1cwav
    - trans_path:       path/to/trans.txt
        拼接音频的总trans, 依据总trans划分和生成各自的trans
    - offset:           type: float
        播放开始位置距音频起始位置的偏移量
    - mode:             oners或者onex2
        相机类型
    
'''
if __name__ == '__main__':
    concat_wav_path = sys.argv[1]
    out_pcm_path = sys.argv[2]
    out_path = sys.argv[3]
    trans_path = sys.argv[4]
    offset = float(sys.argv[5])
    mode = sys.argv[6]

    path_4c = os.path.join(out_path,'4cwav')
    path_1c = os.path.join(out_path,'1cwav')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    if not os.path.exists(path_4c):
        os.mkdir(path_4c)
    if not os.path.exists(path_1c):
        os.mkdir(path_1c)

    if 'oners' in mode:
        drop_wav_head(concat_wav_path, out_pcm_path)
        voice_data = pcm_read(out_pcm_path, sample_width=4, frame_rate=48000, channels=3)
    else:
        voice_data = pcm_read(concat_wav_path, sample_width=2, frame_rate=16000, channels=1)
    
    # 通过concat wav的trans读取音频时长
    lines = open(trans_path,'r',encoding='utf-8').readlines()

    lengths = []
    wav_name = []
    labels = []
    sum_num = offset
    for line in lines:
        path, label = line.strip().split('\t')
        sr, data = wavfile.read(path)
        sum_num += data.shape[0]/sr
        lengths.append(sum_num)
        wav_name.append(os.path.basename(path))
        labels.append(label)
    
    print('trans read done')
    # print(lengths[:100])
    start = offset
    if 'oners' in mode:
        w = open(os.path.join(path_4c,'trans.txt'),'w',encoding='utf-8')
    else:
        w = open(os.path.join(path_1c,'trans.txt'),'w',encoding='utf-8')
    for index,i in enumerate(lengths):
        # 读取的音频以ms为单位，所以 second*1000
        temp = voice_data[int(start*1000):int(i*1000)]
        # print(int(start*1000), int(i*1000))

        np_data = np.array(temp.get_array_of_samples())
        if 'oners' in mode:
            # insta三通道变四通道
            wav_data = np_data.reshape(-1, 3)
            new_audio_dst = np.concatenate((wav_data, wav_data[:, 2].reshape(-1, 1)), axis=1)
        out_path = os.path.join(path_4c, wav_name[index])
        if 'oners' in mode:
            wavfile.write(out_path, 48000, np.array(new_audio_dst))
        else:
            out_path = out_path.replace('4cwav','1cwav')
            wavfile.write(out_path, 16000, np.array(np_data))
        
        w.write('{}\t{}\n'.format(out_path, labels[index]))
        
        start = i
        if index%1000 == 0:
            print('process {} wavs'.format(index))
    w.close()

    if 'oners' in mode:
        os.system('python kws_split.py {} {} {}'.format(path_4c, path_1c, mode))
        mk_1c_trans(path_4c, path_1c)

    print('all done')