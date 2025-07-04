import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
FILTERBANK_LEN = 23


class TCN_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation, residual=False,
                use_weight_norm=True, batch_norm=True):
        super(TCN_Conv1d, self).__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.use_weight_norm = use_weight_norm
        self.batch_norm = batch_norm
        if self.use_weight_norm:
            self.conv = weight_norm(
                            nn.Conv1d(in_channels, out_channels, 
                            kernel_size=kernel, dilation=dilation)
                        )
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, 
                            kernel_size=kernel, dilation=dilation)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels, affine=False)
        self.residual = residual
        if self.residual:
            if in_channels != out_channels:
                self.downsample = nn.Conv1d(in_channels, out_channels, 1)
            else:
                self.downsample = None
            self.bn2 = nn.BatchNorm1d(out_channels, affine=False)

         
    def forward(self, x):
        out = F.pad(x, [(self.kernel-1)*self.dilation, 0])
        out = self.conv(out)
        if self.batch_norm:
            out = self.bn(out)
        out = F.relu(out)
        if self.residual:
            if self.downsample:
                x = self.downsample(x)
            x = self.bn2(x)
            out = F.relu(x + out)
        return out

    def receptive_field(self, ):
        return (self.kernel-1)*self.dilation


class TCN(nn.Module):
    def __init__(self, channels, kernels, dilations, num_classes, residual=False,
                use_weight_norm=True, batch_norm=True):
        super(TCN, self).__init__()
        self.num_layers = len(channels)
        self.kernels = kernels
        self.dilations = dilations
        channels.append(FILTERBANK_LEN)
        convs = []
        for i in range(self.num_layers):
            convs.append(
                TCN_Conv1d(channels[i-1], channels[i], kernels[i], dilations[i], residual,
                            use_weight_norm, batch_norm)
            )
        self.convs = nn.Sequential(*convs)
        self.dense = nn.Linear(channels[-2], num_classes)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = x.permute(0, 2, 1)
        x = self.dense(x)
        x = F.softmax(x, -1)
        return x


    def get_param_num(self):
        return sum(p.numel() for p in self.parameters())


    def receptive_field(self, ):
        rf_size = 0
        for c in self.convs:
            rf_size += c.receptive_field()
        return rf_size

    def paramter_nums(self, ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test():
    channels = [64] * 5
    kernels = [8] * 5
    dilations = [1, 2, 4, 8, 16]
    num_classes = 5
    net = TCN(channels, kernels, dilations, num_classes)
    print(net)
    print(net.receptive_field())
    input_ = torch.rand((1, 8, 23), dtype=torch.float32)
    output = net(input_)
    for name, param in net.named_parameters():
        print(name, ":", param.size())
    print("num of parameters:", sum(p.numel() for p in net.parameters()))
    print("num of trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    
if __name__ == "__main__":
    test()
        
        
        
