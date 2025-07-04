#!/usr/bin/env python
# Copyright 2022 Mobvoi Inc. All Rights Reserved.
# Author: kai.zhou2221@mobvoi.com (Kai Zhou)

import sys, os
import argparse
import numpy as np
import torch
from torch.nn.utils import remove_weight_norm

"""
    for 140k big tcn model, apply in dsp

"""


def weight2str(w):
    assert len(w.shape) <= 2
    cpp_str = ""
    k = 0
    num_per_line = 6
    for wi in w.reshape(-1):
        if k > 1 and k % num_per_line == 0:
            cpp_str += "\n"
        k += 1
        cpp_str += "{:12f},".format(wi)
    return cpp_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    pt_file = args.pt_file
    date = args.date
    project = args.project

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
    
    
    # 初始信息
    cpp_src = "// Auto generated! Do **NOT** edit!\n\n"
    cpp_src += "// Generated from {}, big model\n".format(
        pt_file
    )
    cpp_src += "// Copyright 2022 Mobvoi Inc. All Rights Reserved.\n"
    cpp_src += "// Author: kai.zhou2221@mobvoi.com (Kai Zhou)\n\n"

    cpp_src += "#pragma once\n\n"
    cpp_src += "#include <stdint.h>\n\n"
    
    #convs
    for i in range(len(net.convs)):
        w = net.convs[i].conv.weight.permute(0, 2, 1).reshape(64, -1)
        b = net.convs[i].conv.bias
        cpp_src += "static const float LAYER{}_W_BIG[{}] = {{\n".format(
            i+1, w.shape[0] * w.shape[1]
        )
        cpp_src += weight2str(w)
        cpp_src += "\n};\n\n"
        cpp_src += "static const float LAYER{}_B_BIG[{}] = {{\n".format(
            i+1, b.shape[0]
        )
        cpp_src += weight2str(b)
        cpp_src += "\n};\n\n"
        
    # last linear
    dense_w = net.dense.weight
    dense_b = net.dense.bias
    cpp_src += "static const float BOTTLENECK_W_BIG[{}] = {{\n".format(
        dense_w.shape[0] * dense_w.shape[1]
    )
    cpp_src += weight2str(dense_w)
    cpp_src += "\n};\n\n"
    cpp_src += "static const float BOTTLENECK_B_BIG[{}] = {{\n".format(
        dense_b.shape[0]
    )
    cpp_src += weight2str(dense_b)
    cpp_src += "\n};\n\n"
    
    # save cpp_src
    cpp_src_filename = "{}_weights_bias_{}_140k.h".format(
        date, project
    )
    if args.name:
        cpp_src_filename = args.name
    model_dir = os.path.dirname(pt_file)
    with open(model_dir + "/" + cpp_src_filename, "w") as f:
        f.write(cpp_src[:-1])
    print("write\n  {}\nsuccessfully!".format(cpp_src_filename))


if __name__ == "__main__":
    main()