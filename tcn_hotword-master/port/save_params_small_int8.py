#!/usr/bin/env python
# Copyright 2022 Mobvoi Inc. All Rights Reserved.
# Author: kai.zhou2221@mobvoi.com (Kai Zhou)

import sys, os
import argparse
import json
import numpy as np
import torch
from save.qformat import *

"""
    for 20k small tcn model, apply in dsp with quant

"""
def get_min_max_value(value):
    min_v = np.amin(value)
    max_v = np.amax(value)
    #  print("min max: ", min_v, max_v)
    return min_v, max_v


def get_weights_num_frac_bits(w):
    w = w.detach().numpy()
    min_v, max_v = get_min_max_value(w)
    assert -128 <= min_v <= 127
    assert -128 <= max_v <= 127
    v = max(abs(min_v), abs(max_v))
    '''
    bits = [
        (1, 7),  # q0.7
        (2, 6),  # q1.6
        (4, 5),  # q2.5
        (8, 4),  # q3.4
        (16, 3),  #q4.3
        (32, 2),  #q5.2
        (64, 1),  #q6.1
        (128, 0),  #q7.0
    ]
    '''
    bits = list()
    for i in reversed(range(0, 8)):
        bits.append((1 << (7 - i), i))

    res = 7
    for i in bits:
        if v <= i[0]:
            res = i[1]
            break
    return res

def get_weights_num_frac_16bits(w):
    w = w.detach().numpy()
    min_v, max_v = get_min_max_value(w)
    assert np.iinfo(np.int16).min <= min_v <= np.iinfo(np.int16).max
    assert np.iinfo(np.int16).min <= max_v <= np.iinfo(np.int16).max
    v = max(abs(min_v), abs(max_v))
    '''
    bits = [
        (1, 15),  # q0.15
        (2, 14),  # q1.14
        (4, 13),  # q2.13
        (8, 12),  # q3.12
        (16, 11),  #q4.11
        (32, 10),  #q5.10
        (64, 9),  #q6.9
        (128, 8),  #q7.8
        ...
    ]
    '''
    bits = list()
    for i in reversed(range(0, 16)):
        bits.append((1 << (15 - i), i))

    res = 15
    for i in bits:
        if v <= i[0]:
            res = i[1]
            break
    return res

def get_bias_num_frac_bits(b):
    min_v, max_v = get_min_max_value(b)
    assert np.iinfo(np.int32).min <= min_v <= np.iinfo(np.int32).max
    assert np.iinfo(np.int32).min <= max_v <= np.iinfo(np.int32).max
    v = max(abs(min_v), abs(max_v))
    '''
    bits = [
        (1, 31),  # q0.31
        (2, 30),  # q1.30
        (4, 29),  # q2.29
        ...
    ]
    '''
    bits = list()
    for i in reversed(range(0, 32)):
        bits.append((1 << (31 - i), i))

    res = 31
    for i in bits:
        if v <= i[0]:
            res = i[1]
            break
    return res

def weight_float2int8(w, frac):
    assert len(w.shape) <= 2
    w = w.detach().numpy()
    qw = []
    if len(w.shape) == 1:
        for wi in w:
            qw.append(float_to_q7(wi, frac))
        return np.array(qw).astype(np.int8)
    else:
        for wi in w:
            qwi = []
            for wii in wi:
                qwi.append(float_to_q7(wii, frac))
            qw.append(qwi)
        return np.array(qw).astype(np.int8)


def weight2vec(w):
    vec = []
    lv = np.array_split(w, w.shape[0]//4, axis=0)
    for l in lv:
        if l.shape[1] % 2 == 0:
            llv = np.array_split(l, l.shape[1]//2, axis=1)
        else:
            llv = np.array_split(l, l.shape[1]//2+1, axis=1)
        for ll in llv:
            lllv = np.array_split(ll, ll.shape[0]//2, axis=0)
            for lll in lllv:
                vec.append(lll.T.reshape(-1))
    vec = np.concatenate(vec)
    return vec


def weight2str(w):
    assert len(w.shape) <= 2
    cpp_str = ""
    k = 0
    num_per_line = 16
    if len(w.shape) == 2:
        v = weight2vec(w)
    else:
        v = w
    for wi in v:
        if k > 1 and k % num_per_line == 0:
            cpp_str += "\n"
        k += 1
        cpp_str += "{:4d},".format(wi)
    return cpp_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--date", default="")
    args = parser.parse_args()

    pt_file = args.pt_file
    date = args.date

    # fbank = torch.load('test_fbank.pt')
    # 加载模型
    net = torch.load(pt_file, map_location='cpu')
    
    # BN merge
    merged_net = torch.quantization.fuse_modules(net,
        [
            ['layer0', 'bn0'],
            ['SepConv1ds.0.pointwise_conv', 'SepConv1ds.0.bn'],
            ['SepConv1ds.1.pointwise_conv', 'SepConv1ds.1.bn'],
            ['SepConv1ds.2.pointwise_conv', 'SepConv1ds.2.bn'],
            ['SepConv1ds.3.pointwise_conv', 'SepConv1ds.3.bn'],
        ]
    )
    # print(net(fbank))
    # print(merged_net(fbank))
    
    # 定点化
    weights = {}
    qweights = []
    
    # first conv layer
    w = merged_net.layer0.weight.squeeze(-1)
    b = merged_net.layer0.bias
    weights["layer0_w"] = {}
    weights["layer0_w"]["float_val"] = w
    frac = get_weights_num_frac_bits(w)
    qw = weight_float2int8(w, frac)
    weights["layer0_w"]["int8_val"] = qw
    weights["layer0_w"]["int8_frac"] = frac
    
    weights["layer0_b"] = {}
    weights["layer0_b"]["float_val"] = b
    frac = get_weights_num_frac_bits(b)
    qb = weight_float2int8(b, frac)
    weights["layer0_b"]["int8_val"] = qb
    weights["layer0_b"]["int8_frac"] = frac
    
    # if True:
    #     fbank = torch.load("test_fbank.pt")
    #     frac = get_weights_num_frac_16bits(fbank)
    #     print(frac)
    #     rf = torch.matmul(weights["layer0_w"]["float_val"], fbank.T)
    #     rq = torch.matmul(weights["layer0_w"]["int_val"], fbank.T)
    
    
    
    # sep convs
    for i in range(0, len(net.SepConv1ds)):
        depth_w = merged_net.SepConv1ds[i].depthwise_conv.weight.squeeze(1).T
        point_w = merged_net.SepConv1ds[i].pointwise_conv.weight.squeeze(-1)
        b = merged_net.SepConv1ds[i].pointwise_conv.bias
        weights["sep_conv{}_depth_w".format(i)] = {}
        weights["sep_conv{}_depth_w".format(i)]["float_val"] = depth_w
        frac = get_weights_num_frac_bits(depth_w)
        qw = weight_float2int8(depth_w, frac)
        weights["sep_conv{}_depth_w".format(i)]["int8_val"] = qw
        weights["sep_conv{}_depth_w".format(i)]["int8_frac"] = frac
        
        weights["sep_conv{}_point_w".format(i)] = {}
        weights["sep_conv{}_point_w".format(i)]["float_val"] = point_w
        frac = get_weights_num_frac_bits(point_w)
        qw = weight_float2int8(point_w, frac)
        weights["sep_conv{}_point_w".format(i)]["int8_val"] = qw
        weights["sep_conv{}_point_w".format(i)]["int8_frac"] = frac
        
        weights["sep_conv{}_b".format(i)] = {}
        weights["sep_conv{}_b".format(i)]["float_val"] = b
        frac = get_weights_num_frac_bits(b)
        qb = weight_float2int8(b, frac)
        weights["sep_conv{}_b".format(i)]["int8_val"] = qb
        weights["sep_conv{}_b".format(i)]["int8_frac"] = frac
        
    # last linear
    dense_w = merged_net.dense.weight
    dense_b = merged_net.dense.bias
    weights["dense_w"] = {}
    weights["dense_w"]["float_val"] = dense_w
    frac = get_weights_num_frac_bits(dense_w)
    qw = weight_float2int8(dense_w, frac)
    weights["dense_w"]["int8_val"] = qw
    weights["dense_w"]["int8_frac"] = frac
    
    weights["dense_b"] = {}
    weights["dense_b"]["float_val"] = dense_b
    frac = get_weights_num_frac_bits(dense_b)
    qb = weight_float2int8(dense_b, frac)
    weights["dense_b"]["int8_val"] = qb
    weights["dense_b"]["int8_frac"] = frac
    
    # print(json.dumps(weights, indent=4))
    
    
    # 初始信息
    cpp_src = "// Auto generated! Do **NOT** edit!\n\n"
    cpp_src += "// Generated from {}, samll model with int8 quant\n".format(
        pt_file
    )
    cpp_src += "// Copyright 2022 Mobvoi Inc. All Rights Reserved.\n"
    cpp_src += "// Author: kai.zhou2221@mobvoi.com (Kai Zhou)\n\n"

    cpp_src += "#pragma once\n\n"
    cpp_src += "#include <stdint.h>\n\n"
    
    cpp_src += "#if defined(CONST_TABLE_IN_FLASH)\n"
    cpp_src += "#define CONST const\n"
    cpp_src += "#else\n"
    cpp_src += "#define CONST\n"
    cpp_src += "#endif\n\n"

    cpp_src += "#define CMDS_LAYER0_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER0_W_COLS 23\n"
    cpp_src += "#define CMDS_LAYER0_W_FRAC {}\n\n".format(weights["layer0_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER0_B_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER0_B_FRAC {}\n\n".format(weights["layer0_b"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER1_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER1_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define CMDS_LAYER1_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv0_depth_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER1_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER1_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define CMDS_LAYER1_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv0_point_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER1_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER1_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv0_b"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER2_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER2_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define CMDS_LAYER2_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv1_depth_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER2_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER2_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define CMDS_LAYER2_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv1_point_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER2_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER2_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv1_b"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER3_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER3_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define CMDS_LAYER3_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv2_depth_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER3_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER3_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define CMDS_LAYER3_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv2_point_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER3_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER3_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv2_b"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER4_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER4_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define CMDS_LAYER4_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv3_depth_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER4_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER4_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define CMDS_LAYER4_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv3_point_w"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER4_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define CMDS_LAYER4_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv3_b"]["int8_frac"])

    cpp_src += "#define CMDS_BOTTLENECK_W_ROWS 9\n"
    cpp_src += "#define CMDS_BOTTLENECK_W_COLS 64\n"
    cpp_src += "#define CMDS_BOTTLENECK_W_FRAC {}\n\n".format(weights["dense_w"]["int8_frac"])

    cpp_src += "#define CMDS_BOTTLENECK_B_ROWS 9\n"
    cpp_src += "#define CMDS_BOTTLENECK_B_FRAC {}\n\n".format(weights["dense_b"]["int8_frac"])

    cpp_src += "#define CMDS_LAYER1_DILATION 2\n"
    cpp_src += "#define CMDS_LAYER1_NUM_ELEMENTS 8\n\n"
    
    cpp_src += "#define CMDS_LAYER2_DILATION 4\n"
    cpp_src += "#define CMDS_LAYER2_NUM_ELEMENTS 8\n"

    cpp_src += "#define CMDS_LAYER3_DILATION 4\n"
    cpp_src += "#define CMDS_LAYER3_NUM_ELEMENTS 8\n\n"

    cpp_src += "#define CMDS_LAYER4_DILATION 8\n"
    cpp_src += "#define CMDS_LAYER4_NUM_ELEMENTS 8\n\n"
    
    
    
    # first conv layer
    w = weights["layer0_w"]["int8_val"]
    w_frac = weights["layer0_w"]["int8_frac"]
    b = weights["layer0_b"]["int8_val"]
    b_frac  = weights["layer0_b"]["int8_frac"]
    cpp_src += "// q{}.{}\n".format(7 - w_frac, w_frac)
    cpp_src += "CONST int8_t CMDS_LAYER0_W[{}] = {{\n".format(
        w.shape[0] * w.shape[1]
    )
    cpp_src += weight2str(w)
    cpp_src += "\n};\n\n"
    cpp_src += "// q{}.{}\n".format(7 - b_frac, b_frac)
    cpp_src += "CONST int8_t CMDS_LAYER0_B[{}] = {{\n".format(
        b.shape[0]
    )
    cpp_src += weight2str(b)
    cpp_src += "\n};\n\n"
    
    # sep convs
    for i in range(0, len(net.SepConv1ds)):
        depth_w = weights["sep_conv{}_depth_w".format(i)]["int8_val"]
        depth_w_frac = weights["sep_conv{}_depth_w".format(i)]["int8_frac"]
        point_w = weights["sep_conv{}_point_w".format(i)]["int8_val"]
        point_w_frac = weights["sep_conv{}_point_w".format(i)]["int8_frac"]
        b = weights["sep_conv{}_b".format(i)]["int8_val"]
        b_frac = weights["sep_conv{}_b".format(i)]["int8_frac"]
        cpp_src += "// q{}.{}\n".format(7 - depth_w_frac, depth_w_frac)
        cpp_src += "CONST int8_t CMDS_LAYER{}_DEPTH_CONV_W[{}] = {{\n".format(
            i+1, depth_w.shape[0] * depth_w.shape[1]
        )
        cpp_src += weight2str(depth_w)
        cpp_src += "\n};\n\n"
        cpp_src += "// q{}.{}\n".format(7 - point_w_frac, point_w_frac)
        cpp_src += "CONST int8_t CMDS_LAYER{}_POINTWISE_CONV_W[{}] = {{\n".format(
            i+1, point_w.shape[0] * point_w.shape[1]
        )
        cpp_src += weight2str(point_w)
        cpp_src += "\n};\n\n"
        cpp_src += "// q{}.{}\n".format(7 - b_frac, b_frac)
        cpp_src += "CONST int8_t CMDS_LAYER{}_POINTWISE_CONV_B[{}] = {{\n".format(
            i+1, b.shape[0]
        )
        cpp_src += weight2str(b)
        cpp_src += "\n};\n\n"
        
    # last linear
    dense_w = weights["dense_w"]["int8_val"]
    dense_w_frac = weights["dense_w"]["int8_frac"]
    dense_b = weights["dense_b"]["int8_val"]
    dense_b_frac = weights["dense_b"]["int8_frac"]
    cpp_src += "// q{}.{}\n".format(7 - dense_w_frac, dense_w_frac)
    cpp_src += "CONST int8_t CMDS_BOTTLENECK_W[{}] = {{\n".format(
        dense_w.shape[0] * dense_w.shape[1]
    )
    cpp_src += weight2str(dense_w)
    cpp_src += "\n};\n\n"
    cpp_src += "// q{}.{}\n".format(7 - dense_b_frac, dense_b_frac)
    cpp_src += "CONST int8_t CMDS_BOTTLENECK_B[{}] = {{\n".format(
        dense_b.shape[0]
    )
    cpp_src += weight2str(dense_b)
    cpp_src += "\n};\n\n"
    
    # save cpp_src
    cpp_src_filename = "{}_weights_bias_{}_cmds_20k_only_int8.h".format(
        date, args.project
    )
    model_dir = os.path.dirname(pt_file)
    with open(model_dir + "/" + cpp_src_filename, "w") as f:
        f.write(cpp_src[:-1])
    print("write\n  {}\nsuccessfully!".format(cpp_src_filename))
    
    
def get_CMVN(CMVN_json):
    with open(CMVN_json) as f:
        stat = json.load(f)
        mean = np.array(stat['mean'])
        scale = np.array(stat['scale'])
    return mean, scale


if __name__ == "__main__":
    main()
    # b = np.array([x for x in range(23*64)]).reshape(64, 23)
    # a = weight2vec(b)
