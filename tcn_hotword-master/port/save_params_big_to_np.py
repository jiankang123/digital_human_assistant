#!/usr/bin/env python
# Copyright 2022 Mobvoi Inc. All Rights Reserved.
# Author: kai.zhou2221@mobvoi.com (Kai Zhou)

import sys, os
import argparse
import numpy as np
import torch
from torch.nn.utils import remove_weight_norm

"""
    for 140k big tcn model, apply in ONE
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", default="")
    # parser.add_argument("--out_dir", default="")
    args = parser.parse_args()
    pt_file = args.pt_file
    out_dir = pt_file+'_npys'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 加载模型
    net = torch.load(pt_file, map_location='cpu')
    
    # BN merge
    if hasattr(net.convs[0], "bn"):
        if hasattr(net.convs[0].bn, "running_mean"):
            for i in range(len(net.convs)):
                remove_weight_norm(net.convs[i].conv)
            merged_net = torch.quantization.fuse_modules(net,
                [
                    ['convs.0.conv', 'convs.0.bn'],
                    ['convs.1.conv', 'convs.1.bn'],
                    ['convs.2.conv', 'convs.2.bn'],
                    ['convs.3.conv', 'convs.3.bn'],
                    ['convs.4.conv', 'convs.4.bn'],
                ]
            )
            net = merged_net
    
    
    #convs
    for i in range(len(net.convs)):
        w = net.convs[i].conv.weight.detach()
        b = net.convs[i].conv.bias.detach()
        np.save(os.path.join(out_dir, f'conv{i}_w'), w.numpy())
        np.save(os.path.join(out_dir, f'conv{i}_b'), b.numpy())

    # last linear
    dense_w = net.dense.weight.detach()
    dense_b = net.dense.bias.detach()
    np.save(os.path.join(out_dir, f'dense_w'), dense_w.numpy())
    np.save(os.path.join(out_dir, f'dense_b'), dense_b.numpy())

    
    print("write {} successfully!".format(out_dir))


if __name__ == "__main__":
    main()