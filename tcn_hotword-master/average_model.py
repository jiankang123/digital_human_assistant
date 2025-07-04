# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
import os
import argparse
import glob

import yaml
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--pt_dir', required=True, help='averaged model')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='averaged model')
    parser.add_argument('--num',
                        default=20,
                        type=int,
                        help='nums for averaged model')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    info_path = os.path.join(args.pt_dir, 'info.txt')
    out_pt = os.path.join(args.pt_dir, 'average{}.pt'.format(args.num))
    info_list = []
    with open(info_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            info_list.append([line.strip().split('\t')[0], line.strip().split('\t')[2]])
    info_arr = np.array(info_list)
    sort_idx = np.argsort(info_arr[:, -1])
    sorted_val_scores = info_arr[sort_idx][::1]
    print("best val scores = " + str(sorted_val_scores[:args.num, 1]))
    print("selected epochs = " +
            str(sorted_val_scores[:args.num, 0].astype(np.int64)))
    path_list = [
        args.pt_dir + '/epoch{}.pt'.format(int(epoch)+1)
        for epoch in sorted_val_scores[:args.num, 0]
    ]

    print(path_list)
    avg = None
    num = args.num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        model = torch.load(glob.glob(path[:-3]+'*.pt')[0], map_location=torch.device('cpu'))
        states = model.state_dict()
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(out_pt))
    model.load_state_dict(avg)
    torch.save(model, out_pt)


if __name__ == '__main__':
    main()
