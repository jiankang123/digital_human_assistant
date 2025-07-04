import os
import sys


if __name__ == '__main__':
    trans_in = sys.argv[1]
    trans_out = sys.argv[2]

    lines = open(trans_in.replace('ripping','gain_maxed').split('.tr')[0]+'.trans','r',encoding='utf-8').readlines()
    path_dic = {}
    for line in lines:
        path,label = line.strip().split('\t')
        new_path = path.split('_gain_')[0].replace('wavs/gain_maxed_m510','wavs/m510')
        if 'add_noise' in new_path:
            new_path = new_path.replace('/1cwav','')
        new_path = new_path+'.wav'
        # print(new_path)
        path_dic[new_path] = path

    with open(trans_in,'r',encoding='utf-8') as r:
        lines = r.readlines()
        with open(trans_out,'w',encoding='utf-8') as w:
            for line in lines:
                path,label = line.strip().split('\t')
                # print('key:',path)
                w.write('{}\t{}\n'.format(path_dic[path],label))

