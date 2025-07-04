import torch
from net.tcn_sep import TCN_SEP
from net.tcn import TCN
from net.mdtcn import MDTC

def build_model(config):
    if config["model_type"] == 'mdtcn':
        print(f"not supprt config build yet, using default config")
        num_layers = 3
        net = MDTC(num_layers, 4, 23, 64, 5, True, 2)
    elif config["model_type"] == 'tcn':
        net = TCN(**config["model_param"])
    elif config["model_type"] == 'dstcn':
        net = TCN_SEP(**config["model_param"])
    else:
        print(f"not support model type {config['model_type']}")

    print(f"net type: {config['model_type']}")
    print(f"layers {config['num_layers']}")
    print(f"receptive field {net.receptive_field()}")
    print(f"parameter numbers {net.paramter_nums()}")
    return net


if __name__ == "__main__":
    config = {
        "model_type": "tcn",
        "num_layers": 5,
        "model_param": {
            "channels": [64] * 5,
            "kernels": [8] * 5,
            "dilations": [1, 2, 4, 4, 8],
            "num_classes": 2,
            "residual": False
        }
    }
    build_model(config)