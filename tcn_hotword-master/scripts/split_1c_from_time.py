from scipy.io import wavfile
import numpy as np
import os
import sys
from split_insta_wav import *

'''
    从记录时间的方法录制的音频中切分出单个音频
    params:
    - input_wav_path:  path/to/insta.wav
        从相机中拿到的原始音频位置, 是一个.wav文件
    - out_pcm_path:     path/to/out.pcm
        输出的pcm文件位置, 一般输出名为"日期_备注.pcm"
    - out_path:         path/to/output/
        输出目录, 会创建两个子目录4cwav和1cwav
    - time_label:       path/to/time.txt
        记录播放时间的label, 依据播放时间划分和生成各自音频
    - offset:           type: float
        播放开始位置距音频起始位置的偏移量
    - mode:             oners或者onex2
        相机类型
    
'''

if __name__ == "__main__":
    
    input_wav_path = sys.argv[1]
    out_pcm_path = sys.argv[3]
    out_path = sys.argv[4]
    time_label = sys.argv[5]
    offset = sys.argv[6]
    mode = sys.argv[7]
    
    path_4c = os.path.join(out_path,'4cwav')
    path_1c = os.path.join(out_path,'1cwav')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    if not os.path.exists(path_4c):
        os.mkdir(path_4c)
    if not os.path.exists(path_1c):
        os.mkdir(path_1c)

    drop_wav_head(cat_wav_path, out_pcm_path)

    voice_data = pcm_read(out_pcm_path)

    # 通过time label读取音频位置
    cut_file = open(time_file,'r',encoding='utf-8').readlines()

    count = 0
    total = len(cut_file)
    w = open(os.path.join(path_4c,'trans.txt'),'w',encoding='utf-8')
    for i in self.cut_file:
        wav_name = os.path.basename(i.split('\t')[0])
        start = float(i.split('\t')[1])+self.offset
        end = float(i.split('\t')[2])+self.offset
        label = i.strip().split('\t')[-1]
        
        audio_dst = voice_data[int(start*1000):int(end*1000)]
        
        np_data = np.array(audio_dst.get_array_of_samples())
        wav_data = np_data.reshape(-1, channels)
        new_audio_dst = np.concatenate((wav_data, wav_data[:, 2].reshape(-1, 1)), axis=1)
        out_path = os.path.join(path_4c,wav_name)
        wavfile.write(out_path, 48000, np.array(new_audio_dst))
        
        w.write('{}\t{}\n'.format(out_path,label))
        
        if count%1000 == 0:
            print('processed {} wavs'.format(count))
        count += 1


    os.system('python kws_split.py {} {} {}'.format(path_4c, path_1c, mode))
    mk_1c_trans(path_4c, path_1c)

    print('all done')
