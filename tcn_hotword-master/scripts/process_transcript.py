import sys
import os
import multiprocessing as mlp
from tqdm import tqdm


def main():
    trans_in = sys.argv[1]
    trans_out = sys.argv[2]
    
    label_map = {
        "zuogebiaoji": "做个标记",
        "guanbixiangji": "关闭相机",
        "kaishiluxiang": "开始录像",
        "tingzhiluxiang": "停止录像",
        "paizhangzhaopian": "拍张照片",
        "GARBAGE": "GARBAGE",
        "garbage": "GARBAGE"
    }

    out_lines = []
    with open(trans_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            if '\t' in line:
                file_path, text = line.strip().split('\t')
            else:
                file_path, text = line.strip().split(' ', 1)
            if os.path.exists(file_path):
                if text in label_map.keys():
                    text = label_map[text]
                    out_lines.append(file_path + '\t' + text + '\n')
                elif text in label_map.items():
                    out_lines.append(file_path + '\t' + text + '\n')
                else:
                    print("{} is not target word".format(text))
            else:
                print("{} not exists!".format(file_path))
    
    with open(trans_out, 'w', encoding='utf-8') as fout:
        for line in out_lines:
            fout.write(line)
    print("Done")



if __name__ == "__main__":
    main()
