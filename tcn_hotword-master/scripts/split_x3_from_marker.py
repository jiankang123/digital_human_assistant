import os
import sys
from split_insta_wav import *
from wav_process import *
from kws_split import process_oners_kws, process_onex2_kws

'''
    从标注好的音频中切分出单个音频
    params:
    - marker_dir:   path/to/insta.wav
        标注好的wav文件位置
    - out_path:     path/to/output/
        输出目录, 会创建两个子目录4cwav和1cwav
    - mode:         oners或者onex2
        相机类型
    
'''

if __name__ == '__main__':
    marker_dir = sys.argv[1]
    out_path = sys.argv[2]

    path_1c = os.path.join(out_path,'1cwav')

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not os.path.exists(path_1c):
        os.mkdir(path_1c)

    s = Cut_Wav_Make_Label(path_1c,mode='x3')
    for root,dirs,files in os.walk(marker_dir):
        for file in files:
            if file.endswith('.wav'):
                s.wav_split(os.path.join(root, file),mode='x3')

    s.close_label_file()