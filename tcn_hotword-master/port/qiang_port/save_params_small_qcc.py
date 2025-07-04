#!/usr/bin/env python
# Copyright 2022 Mobvoi Inc. All Rights Reserved.
# Author: kai.zhou2221@mobvoi.com (Kai Zhou)

import sys, os
import argparse
import numpy as np
import torch
from qformat import *

"""
    for 20k small tcn model, apply in dsp with quant

"""

def get_min_max_value(value):
    min_v = np.amin(value)
    max_v = np.amax(value)
    #  print("min max: ", min_v, max_v)
    return min_v, max_v

# 计算weight 8bit定点化的frac
def get_weights_num_frac_8bits(w):
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

# 计算bias 32bit定点化的frac
def get_bias_num_frac_32bits(b):
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

# weight定点化
def weight_float2int8(w, frac):
    assert len(w.shape) <= 2
    qw = []
    for wi in w:
        qwi = []
        for wii in wi:
            qwi.append(float_to_q7(wii, frac))
        qw.append(qwi)
    return np.array(qw).astype(np.int8)

# bias定点化
def bias_float2int32(b, frac):
    qb = []
    for bi in b:
        qb.append(float_to_q31(bi, frac))
    return np.array(qb).astype(np.int32)


# 将多维的weight转为一维的向量
# note：需要根据具体的c代码实现的算子调整参数顺序
def weight2vec(w):
    return w.reshape(-1)


# 将numpy.array类型的weight和bias转为字符串
def weight2str(w): 
    assert len(w.shape) <= 2
    cpp_str = ""
    k = 0
    
    if len(w.shape) == 2:
        v = weight2vec(w)
        num_per_line = 16
    else:
        v = w
        num_per_line = 6
    for vi in v:
        if k > 1 and k % num_per_line == 0:
            cpp_str += "\n"
        k += 1
        if len(w.shape) == 2:
            cpp_str += "{:4d},".format(vi)
        elif len(w.shape) == 1:
            cpp_str += "{:11d},".format(vi)
    return cpp_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_file", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    pt_file = args.pt_file
    project = args.project
    date = args.date
    name = args.name

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
    
    # 定点化
    weights = {}
    
    # first conv layer
    w = merged_net.layer0.weight.squeeze(-1).detach().numpy()
    b = merged_net.layer0.bias.detach().numpy()

    weights["layer0_w"] = {}
    weights["layer0_w"]["float_val"] = w
    frac = get_weights_num_frac_8bits(w)
    qw = weight_float2int8(w, frac)
    weights["layer0_w"]["int8_val"] = qw
    weights["layer0_w"]["int8_frac"] = frac
    
    weights["layer0_b"] = {}
    weights["layer0_b"]["float_val"] = b
    frac = get_bias_num_frac_32bits(b)
    qb = bias_float2int32(b, frac)
    weights["layer0_b"]["int32_val"] = qb
    weights["layer0_b"]["int32_frac"] = frac
    
    # sep convs
    for i in range(0, len(net.SepConv1ds)):
        # depth_w = merged_net.SepConv1ds[i].depthwise_conv.weight.squeeze(1).T.detach().numpy()
        depth_w = merged_net.SepConv1ds[i].depthwise_conv.weight.squeeze(1).detach().numpy()
        point_w = merged_net.SepConv1ds[i].pointwise_conv.weight.squeeze(-1).detach().numpy()
        b = merged_net.SepConv1ds[i].pointwise_conv.bias.detach().numpy()
        
        weights["sep_conv{}_depth_w".format(i)] = {}
        weights["sep_conv{}_depth_w".format(i)]["float_val"] = depth_w
        frac = get_weights_num_frac_8bits(depth_w)
        qw = weight_float2int8(depth_w, frac)
        weights["sep_conv{}_depth_w".format(i)]["int8_val"] = qw
        weights["sep_conv{}_depth_w".format(i)]["int8_frac"] = frac
        
        weights["sep_conv{}_point_w".format(i)] = {}
        weights["sep_conv{}_point_w".format(i)]["float_val"] = point_w
        frac = get_weights_num_frac_8bits(point_w)
        qw = weight_float2int8(point_w, frac)
        weights["sep_conv{}_point_w".format(i)]["int8_val"] = qw
        weights["sep_conv{}_point_w".format(i)]["int8_frac"] = frac
        
        weights["sep_conv{}_b".format(i)] = {}
        weights["sep_conv{}_b".format(i)]["float_val"] = b
        frac = get_bias_num_frac_32bits(b)
        qb = bias_float2int32(b, frac)
        weights["sep_conv{}_b".format(i)]["int32_val"] = qb
        weights["sep_conv{}_b".format(i)]["int32_frac"] = frac
        
    # last linear
    dense_w = merged_net.dense.weight.detach().numpy()
    dense_b = merged_net.dense.bias.detach().numpy()
    class_num = dense_b.shape[0]
    weights["dense_w"] = {}
    weights["dense_w"]["float_val"] = dense_w
    frac = get_weights_num_frac_8bits(dense_w)
    qw = weight_float2int8(dense_w, frac)
    weights["dense_w"]["int8_val"] = qw
    weights["dense_w"]["int8_frac"] = frac
    
    weights["dense_b"] = {}
    weights["dense_b"]["float_val"] = dense_b
    frac = get_bias_num_frac_32bits(dense_b)
    qb = bias_float2int32(dense_b, frac)
    weights["dense_b"]["int32_val"] = qb
    weights["dense_b"]["int32_frac"] = frac
    
    # 初始信息
    cpp_src = "// Auto generated! Do **NOT** edit!\n\n"
    cpp_src += "// Generated from {}, samll model with int8 quant\n".format(
        pt_file
    )
    cpp_src += "// Copyright 2022 Mobvoi Inc. All Rights Reserved.\n"
    cpp_src += "// Author: kai.zhou2221@mobvoi.com (Kai Zhou)\n\n"

    cpp_src += "#ifndef NN_MODEL_{}_{}_{}_H\n".format(project, name, date)
    cpp_src += "#define NN_MODEL_{}_{}_{}_H\n\n".format(project, name, date)

    cpp_src += "#define LAYER0_W_ROWS 64\n"
    cpp_src += "#define LAYER0_W_COLS 23\n"
    cpp_src += "#define LAYER0_W_FRAC {}\n\n".format(weights["layer0_w"]["int8_frac"])

    cpp_src += "#define LAYER0_B_ROWS 64\n"
    cpp_src += "#define LAYER0_B_FRAC {}\n\n".format(weights["layer0_b"]["int32_frac"])

    cpp_src += "#define LAYER1_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER1_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define LAYER1_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv0_depth_w"]["int8_frac"])

    cpp_src += "#define LAYER1_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER1_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define LAYER1_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv0_point_w"]["int8_frac"])

    cpp_src += "#define LAYER1_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define LAYER1_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv0_b"]["int32_frac"])

    cpp_src += "#define LAYER2_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER2_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define LAYER2_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv1_depth_w"]["int8_frac"])

    cpp_src += "#define LAYER2_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER2_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define LAYER2_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv1_point_w"]["int8_frac"])

    cpp_src += "#define LAYER2_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define LAYER2_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv1_b"]["int32_frac"])

    cpp_src += "#define LAYER3_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER3_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define LAYER3_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv2_depth_w"]["int8_frac"])

    cpp_src += "#define LAYER3_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER3_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define LAYER3_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv2_point_w"]["int8_frac"])

    cpp_src += "#define LAYER3_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define LAYER3_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv2_b"]["int32_frac"])

    cpp_src += "#define LAYER4_DEPTH_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER4_DEPTH_CONV_W_COLS 8\n"
    cpp_src += "#define LAYER4_DEPTH_CONV_W_FRAC {}\n\n".format(weights["sep_conv3_depth_w"]["int8_frac"])

    cpp_src += "#define LAYER4_POINTWISE_CONV_W_ROWS 64\n"
    cpp_src += "#define LAYER4_POINTWISE_CONV_W_COLS 64\n"
    cpp_src += "#define LAYER4_POINTWISE_CONV_W_FRAC {}\n\n".format(weights["sep_conv3_point_w"]["int8_frac"])

    cpp_src += "#define LAYER4_POINTWISE_CONV_B_ROWS 64\n"
    cpp_src += "#define LAYER4_POINTWISE_CONV_B_FRAC {}\n\n".format(weights["sep_conv3_b"]["int32_frac"])

    cpp_src += "#define BOTTLENECK_W_ROWS {}\n".format(class_num)
    cpp_src += "#define BOTTLENECK_W_COLS 64\n"
    cpp_src += "#define BOTTLENECK_W_FRAC {}\n\n".format(weights["dense_w"]["int8_frac"])

    cpp_src += "#define BOTTLENECK_B_ROWS {}\n".format(class_num)
    cpp_src += "#define BOTTLENECK_B_FRAC {}\n\n".format(weights["dense_b"]["int32_frac"])

    cpp_src += "#define LAYER1_DILATION 2\n"
    cpp_src += "#define LAYER1_NUM_ELEMENTS 8\n\n"
    
    cpp_src += "#define LAYER2_DILATION 4\n"
    cpp_src += "#define LAYER2_NUM_ELEMENTS 8\n\n"

    cpp_src += "#define LAYER3_DILATION 4\n"
    cpp_src += "#define LAYER3_NUM_ELEMENTS 8\n\n"

    cpp_src += "#define LAYER4_DILATION 8\n"
    cpp_src += "#define LAYER4_NUM_ELEMENTS 8\n\n"
    
    
    
    # first conv layer
    w = weights["layer0_w"]["int8_val"]
    w_frac = weights["layer0_w"]["int8_frac"]
    b = weights["layer0_b"]["int32_val"]
    b_frac  = weights["layer0_b"]["int32_frac"]
    cpp_src += "// q{}.{}\n".format(7 - w_frac, w_frac)
    cpp_src += "static const int8_t LAYER0_W[{}] = {{\n".format(
        w.shape[0] * w.shape[1]
    )
    cpp_src += weight2str(w)
    cpp_src += "\n};\n\n"
    cpp_src += "// q{}.{}\n".format(31 - b_frac, b_frac)
    cpp_src += "static const int32_t LAYER0_B[{}] = {{\n".format(
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
        b = weights["sep_conv{}_b".format(i)]["int32_val"]
        b_frac = weights["sep_conv{}_b".format(i)]["int32_frac"]
        cpp_src += "// q{}.{}\n".format(7 - depth_w_frac, depth_w_frac)
        cpp_src += "static const int8_t LAYER{}_DEPTH_CONV_W[{}] = {{\n".format(
            i+1, depth_w.shape[0] * depth_w.shape[1]
        )
        cpp_src += weight2str(depth_w)
        cpp_src += "\n};\n\n"
        cpp_src += "// q{}.{}\n".format(7 - point_w_frac, point_w_frac)
        cpp_src += "static const int8_t LAYER{}_POINTWISE_CONV_W[{}] = {{\n".format(
            i+1, point_w.shape[0] * point_w.shape[1]
        )
        cpp_src += weight2str(point_w)
        cpp_src += "\n};\n\n"
        cpp_src += "// q{}.{}\n".format(31 - b_frac, b_frac)
        cpp_src += "static const int32_t LAYER{}_POINTWISE_CONV_B[{}] = {{\n".format(
            i+1, b.shape[0]
        )
        cpp_src += weight2str(b)
        cpp_src += "\n};\n\n"
        
    # last linear
    dense_w = weights["dense_w"]["int8_val"]
    dense_w_frac = weights["dense_w"]["int8_frac"]
    dense_b = weights["dense_b"]["int32_val"]
    dense_b_frac = weights["dense_b"]["int32_frac"]

    # w
    cpp_src += "// q{}.{}\n".format(7 - dense_w_frac, dense_w_frac)
    cpp_src += "static const int8_t BOTTLENECK_W[{}] = {{\n".format(
        dense_w.shape[0] * dense_w.shape[1]
    )
    cpp_src += weight2str(dense_w)
    cpp_src += "\n};\n\n"

    # b
    cpp_src += "// q{}.{}\n".format(31 - dense_b_frac, dense_b_frac)
    cpp_src += "static const int32_t BOTTLENECK_B[{}] = {{\n".format(
        dense_b.shape[0]
    )
    
    cpp_src += weight2str(dense_b)
    cpp_src += "\n};\n"
    cpp_src += "#endif  //NN_MODEL_{}_{}_{}_H\n\n".format(project, name, date)
    # save cpp_src
    cpp_src_filename = "nn_model_{}_{}_{}.h".format(
        args.project.lower(), args.name.lower(), args.date, 
    )
    model_dir = os.path.dirname(pt_file)
    with open(model_dir + "/" + cpp_src_filename, "w") as f:
        f.write(cpp_src[:-1])
    print("write\n  {}\nsuccessfully!".format(cpp_src_filename))


if __name__ == "__main__":
    main()
