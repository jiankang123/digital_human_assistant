import os
import torch
import torch.nn.functional as F
from torch import batch_norm, nn
from torch.nn.utils import weight_norm
from net.qiang_tcn import StatisticsPooling

FILTERBANK_LEN = 23


class SepConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation, residual=False,
                batch_norm=True):
        super(SepConv1d, self).__init__()
        self.batch_norm = batch_norm
        self.residual = residual
        self.kernel = kernel
        self.dilation = dilation
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, 
                                    kernel_size=kernel, dilation=dilation, 
                                    bias=False, groups=in_channels)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 
                                kernel_size=1, dilation=1, 
                                groups=1)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_channels, affine=False)
        
        if self.residual:
            if in_channels != out_channels:
                self.downsample = nn.Conv1d(in_channels, out_channels, 1)
            else:
                self.downsample = None
            self.bn2 = nn.BatchNorm1d(out_channels, affine=False)
        

    def forward(self, x):
        out = F.pad(x, [(self.kernel-1)*self.dilation, 0])
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        if self.batch_norm:
            out = self.bn(out)
        out = F.relu(out)
        if self.residual:
            if self.downsample:
                x = self.downsample(x)
            x = self.bn2(x)
            out = F.relu(x + out)
        return out
    

class TCN_SEP(nn.Module):
    def __init__(self, channels, kernels, dilations, num_classes, residual=False,
                batch_norm=True,con_dim=64):
        super(TCN_SEP, self).__init__()
        self.batch_norm = batch_norm
        self.layer0 = nn.Conv1d(FILTERBANK_LEN, channels[0], 
                                kernel_size=kernels[0], 
                                dilation=dilations[0])
        if self.batch_norm:
            self.bn0 = nn.BatchNorm1d(channels[0], affine=False)
        self.num_layers = len(channels)
        
        sepconv1ds = [
            SepConv1d(channels[i-1], channels[i], kernels[i], dilations[i], residual, batch_norm) 
            for i in range(1, self.num_layers)
        ]
        self.SepConv1ds = nn.Sequential(*sepconv1ds)
        self.dense = nn.Linear(channels[-1], num_classes)
        self.pooling = StatisticsPooling()
        self.con_dense = nn.Linear(2*channels[-2], con_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.layer0(x)
        if self.batch_norm:
            x = self.bn0(x)
        x = F.relu(x)
        x = self.SepConv1ds(x)
        x = x.permute(0, 2, 1)
        statistic_x = self.pooling(x)
        con_x = self.con_dense(statistic_x)
        x = self.dense(x)
        x = F.softmax(x, -1)
        return x, con_x


def test():
    channels = [64] * 5
    kernels = [1] + [8] * 4
    dilations = [1, 2, 4, 4, 8]
    num_classes = 9
    net = TCN_SEP(channels, kernels, dilations, num_classes)
    for name, param in net.named_parameters():
        print(name, ":", param.size())
    print("num of parameters:", sum(p.numel() for p in net.parameters()))
    print("num of trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    
if __name__ == "__main__":
    test()
        
        
        