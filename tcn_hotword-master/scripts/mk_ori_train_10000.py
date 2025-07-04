import os
import sys


if __name__ == '__main__':
    # ripping 10000
    trans_in = sys.argv[1]
    # ori trans
    trans_ori = sys.argv[2]
    # out ori 10000
    trans_out = sys.argv[3]

    lines = open(trans_in,'r',encoding='utf-8').readlines()
    path_set = set()
    for line in lines:
        path,label = line.strip().split('\t')
        new_path = path.split('/')[-1]
        # print(new_path)
        path_set.add(new_path)

    with open(trans_ori,'r',encoding='utf-8') as r:
        lines = r.readlines()
        with open(trans_out,'w',encoding='utf-8') as w:
            for line in lines:
                path,label = line.strip().split('\t')
                wav = path.split('/')[-1]
                if wav in path_set:
                    w.write(line)

