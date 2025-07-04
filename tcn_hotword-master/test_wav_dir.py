import sys, os
import torch
from torch.nn.utils import remove_weight_norm
from feat import compute_fbank, get_CMVN
from test_wav import wav_test, wav_simulate_test, wav_simulate_test_TCN


def main():
    test_mode = sys.argv[1]
    net_path = sys.argv[2]
    cmvn = sys.argv[3]
    wav_dir_path = sys.argv[4]


    wav_paths = []
    for root, dirs, files in os.walk(wav_dir_path):
        for file in files:
            if file.endswith('.wav'):
                wav_paths.append(os.path.join(root, file))

    net = torch.load(net_path, map_location='cpu').eval()

    mean, scale = get_CMVN(cmvn)
    if test_mode == "py":
        for wav_path in wav_paths:
            wav_test(wav_path, net, mean, scale)
    elif test_mode == 'c':
        if hasattr(net, "bn0"):
            if hasattr(net.bn0, "running_mean"):
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
        else:
            if hasattr(net.convs[0].bn, "running_mean"):
                remove_weight_norm(net.convs[0].conv)
                remove_weight_norm(net.convs[1].conv)
                remove_weight_norm(net.convs[2].conv)
                remove_weight_norm(net.convs[3].conv)
                remove_weight_norm(net.convs[4].conv)

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
        if hasattr(net, "bn0"):
            for wav_path in wav_paths:
                wav_simulate_test(wav_path, net, mean, scale)
        else:
            for wav_path in wav_paths:
                wav_simulate_test_TCN(wav_path, net, mean, scale)


if __name__ == "__main__":
    main()