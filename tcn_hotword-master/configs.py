
net_configs = {
    "config_dstcn5x128_rf126": {
        "model_type": "dstcn",
        "num_layers": 5,
        "model_param": {
            "channels": [128] * 5,
            "kernels": [1] + [8] * 4,
            "dilations": [1, 2, 4, 4, 8],
            "num_classes": 2,
            "residual": False
        }
    },
    "config_dstcn5x256_rf126": {
        "model_type": "dstcn",
        "num_layers": 5,
        "model_param": {
            "channels": [256] * 5,
            "kernels": [1] + [8] * 4,
            "dilations": [1, 2, 4, 4, 8],
            "num_classes": 2,
            "residual": False
        }
    },
    "config_dstcn5x512_rf126": {
        "model_type": "dstcn",
        "num_layers": 5,
        "model_param": {
            "channels": [512] * 5,
            "kernels": [1] + [8] * 4,
            "dilations": [1, 2, 4, 4, 8],
            "num_classes": 2,
            "residual": False
        }
    },
    "config_dstcn5x64_rf210": {
        "model_type": "dstcn",
        "num_layers": 5,
        "model_param": {
            "channels": [64] * 5,
            "kernels": [1] + [8] * 4,
            "dilations": [1, 2, 4, 8, 16],
            "num_classes": 2,
            "residual": False
        }
    },
    "config_dstcn5x64_rf126": {
        "model_type": "dstcn",
        "num_layers": 5,
        "model_param": {
            "channels": [64] * 5,
            "kernels": [1] + [8] * 4,
            "dilations": [1, 2, 4, 4, 8],
            "num_classes": 2,
            "residual": False
        }
    }



}



