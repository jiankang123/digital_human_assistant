import sys, os
import argparse
import numpy as np
import torch
from net.tcn_sep import TCN_SEP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_file", default="")
    parser.add_argument("--net_path", default="")
    args = parser.parse_args()

    h_file = args.h_file
    net_path = args.net_path
    params = []

    with open(h_file, 'r', encoding='utf-8') as fin:
        load = False
        for line in fin:
            if line.strip().endswith('{'):
                layer_param = []
                load = True
                continue
            elif line.strip().endswith('};'):
                params.append(layer_param)
                load = False
                continue

            if load:
                for p in line.strip().split(','):
                    if len(p) > 0:
                        layer_param.append(float(p))

    net_params = []

    w = torch.tensor(params[0], dtype=torch.float32).reshape(64, -1, 1)
    b = torch.tensor(params[1], dtype=torch.float32)
    net_params.append((w, b))
    for i in range(4):
        w1 = torch.tensor(params[i*3+2], dtype=torch.float32).reshape(8, 64, 1).permute(1, 2, 0)
        w2 = torch.tensor(params[i*3+3], dtype=torch.float32).reshape(64, 64, 1)
        b = torch.tensor(params[i*3+4], dtype=torch.float32)
        net_params.append((w1, w2, b))       
    w = torch.tensor(params[14], dtype=torch.float32).reshape(-1, 64)
    b = torch.tensor(params[15], dtype=torch.float32)
    net_params.append((w, b))

    # 新建模型，加载参数
    num_layers = 5
    channels = [64] * num_layers
    kernels = [1] + [8] * num_layers
    dilations = [1, 2, 4, 4, 8]
    num_classes = 6
    model = TCN_SEP(channels, kernels, dilations, num_classes, residual=False,
                batch_norm=False)
    model.layer0.weight.data.copy_(net_params[0][0])
    model.layer0.bias.data.copy_(net_params[0][1])

    for i in range(4):
        model.SepConv1ds[i].depthwise_conv.weight.data.copy_(net_params[i+1][0])
        model.SepConv1ds[i].pointwise_conv.weight.data.copy_(net_params[i+1][1])
        model.SepConv1ds[i].pointwise_conv.bias.data.copy_(net_params[i+1][2])
    model.dense.weight.data.copy_(net_params[-1][0])
    model.dense.bias.data.copy_(net_params[-1][1])

    model = model.eval()
    torch.save(model, net_path)
    
    print("Done")
            




if __name__ == "__main__":
    main()