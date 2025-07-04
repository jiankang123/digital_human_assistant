import sys, os
import argparse
import numpy as np
import torch
from net.tcn import TCN


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
    for i in range(0, len(params), 2):
        if i < len(params) - 2:
            w = torch.tensor(params[i], dtype=torch.float32).reshape(64, 8, -1).permute(0, 2, 1)
            b = torch.tensor(params[i+1], dtype=torch.float32)
        else:
            w = torch.tensor(params[i], dtype=torch.float32).reshape(-1, 64)
            b = torch.tensor(params[i+1], dtype=torch.float32)
        net_params.append((w, b))
    

    # 新建模型，加载参数
    num_layers = 5
    channels = [64] * num_layers
    kernels = [8] * num_layers
    dilations = [1, 2, 4, 4, 8]
    num_classes = 6
    model = TCN(channels, kernels, dilations, num_classes, residual=False,
                use_weight_norm=False, batch_norm=False)
    for i in range(num_layers):
        model.convs[i].conv.weight.data.copy_(net_params[i][0])
        model.convs[i].conv.bias.data.copy_(net_params[i][1])
    model.dense.weight.data.copy_(net_params[-1][0])
    model.dense.bias.data.copy_(net_params[-1][1])

    model = model.eval()
    torch.save(model, net_path)
    print("Done")
            




if __name__ == "__main__":
    main()