import os
import sys
from split_insta_wav import *
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
    mode = sys.argv[3]

    path_4c = os.path.join(out_path,'4cwav')
    path_1c = os.path.join(out_path,'1cwav')

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    if not os.path.exists(path_4c):
        os.mkdir(path_4c)
    if not os.path.exists(path_1c):
        os.mkdir(path_1c)


    # 切分后为4通道音频
    s = Cut_Wav_Make_Label(path_4c)
    for root,dirs,files in os.walk(marker_dir):
        for file in files:
            if file.endswith('.wav'):
                s.wav_split(os.path.join(root, file))
    s.close_label_file()

    fout = open(os.path.join(path_1c, 'trans.txt'), 'w', encoding='utf-8')
    fin = open(os.path.join(path_4c, 'trans.txt'), 'r', encoding='utf-8').readlines()
    print(len(fin))
    for line in fin:
        print(line.strip().split('\t'))
        wav_path, label = line.strip().split('\t')
        print(wav_path)
        if mode == "oners":
            wav_paths = process_oners_kws(wav_path, path_1c)

        elif mode == "onex2":
            wav_paths = process_onex2_kws(wav_path, path_1c)
        else:
            print("error mode ")
            exit(0)
        for wav in wav_paths:
            fout.write(os.path.abspath(wav) + "\t" + label + "\n")

