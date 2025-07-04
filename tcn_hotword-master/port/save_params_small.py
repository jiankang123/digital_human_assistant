#!/usr/bin/env python
# Copyright 2022 Mobvoi Inc. All Rights Reserved.
# Author: kai.zhou2221@mobvoi.com (Kai Zhou)

import sys, os
import argparse
import numpy as np
import torch
from save.qformat import *

"""
    for 20k small tcn model, apply in dsp

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
    if hasattr(net.layer0, "bn0"):
        merged_net = torch.quantization.fuse_modules(net,
            [
                ['layer0', 'bn0'],
                ['SepConv1ds.0.pointwise_conv', 'SepConv1ds.0.bn'],
                ['SepConv1ds.1.pointwise_conv', 'SepConv1ds.1.bn'],
                ['SepConv1ds.2.pointwise_conv', 'SepConv1ds.2.bn'],
                ['SepConv1ds.3.pointwise_conv', 'SepConv1ds.3.bn'],
            ]
        )
        net = merged_net
    
    # 初始信息
    cpp_src = "// Auto generated! Do **NOT** edit!\n\n"
    cpp_src += "// Generated from {}, samll model\n".format(
        pt_file
    )
    cpp_src += "// Copyright 2022 Mobvoi Inc. All Rights Reserved.\n"
    cpp_src += "// Author: kai.zhou2221@mobvoi.com (Kai Zhou)\n\n"

    cpp_src += "#pragma once\n\n"
    cpp_src += "#include <stdint.h>\n\n"
    
    # first conv layer
    w = net.layer0.weight.squeeze(-1).detach().numpy()
    b = net.layer0.bias.detach().numpy()
    cpp_src += "static const float LAYER0_W[{}] = {{\n".format(
        w.shape[0] * w.shape[1]
    )
    cpp_src += weight2str(w)
    cpp_src += "\n};\n\n"
    cpp_src += "static const float LAYER0_B[{}] = {{\n".format(
        b.shape[0]
    )
    cpp_src += weight2str(b)
    cpp_src += "\n};\n\n"
    
    # sep convs
    for i in range(0, len(net.SepConv1ds)):
        depth_w = net.SepConv1ds[i].depthwise_conv.weight.squeeze(1).T.detach().numpy()
        point_w = net.SepConv1ds[i].pointwise_conv.weight.squeeze(-1).detach().numpy()
        b = net.SepConv1ds[i].pointwise_conv.bias.detach().numpy()
        cpp_src += "static const float LAYER{}_DEPTH_CONV_W[{}] = {{\n".format(
            i+1, depth_w.shape[0] * depth_w.shape[1]
        )
        cpp_src += weight2str(depth_w)
        cpp_src += "\n};\n\n"
        cpp_src += "static const float LAYER{}_POINTWISE_CONV_W[{}] = {{\n".format(
            i+1, point_w.shape[0] * point_w.shape[1]
        )
        cpp_src += weight2str(point_w)
        cpp_src += "\n};\n\n"
        cpp_src += "static const float LAYER{}_POINTWISE_CONV_B[{}] = {{\n".format(
            i+1, b.shape[0]
        )
        cpp_src += weight2str(b)
        cpp_src += "\n};\n\n"
        
    # last linear
    dense_w = net.dense.weight.detach().numpy()
    dense_b = net.dense.bias.detach().numpy()
    cpp_src += "static const float BOTTLENECK_W[{}] = {{\n".format(
        dense_w.shape[0] * dense_w.shape[1]
    )
    cpp_src += weight2str(dense_w)
    cpp_src += "\n};\n\n"
    cpp_src += "static const float BOTTLENECK_B[{}] = {{\n".format(
        dense_b.shape[0]
    )
    cpp_src += weight2str(dense_b)
    cpp_src += "\n};\n\n"
    
    # save cpp_src
    cpp_src_filename = "{}_weights_bias_{}_cmds_20k_float.h".format(
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