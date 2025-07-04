import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
from line_profiler import LineProfiler
FILTERBANK_LEN = 23


class StatisticsPooling(nn.Module):
    def  __init__(self):
        super().__init__()

        self.eps = 1e-5
    
    def forward(self, x):
        mean = []
        std = []
        for snt_id in range(x.shape[0]):
            # Avoiding padded time steps
            actual_size = int(x[snt_id].shape[0])
            # print(actual_size)

            # computing statistics
            mean.append(torch.mean(x[snt_id, 0:actual_size,:], dim=0))
            std.append(torch.std(x[snt_id, 0:actual_size,:], dim=0))

        mean = torch.stack(mean)
        std = torch.stack(std)

        gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
        gnoise = gnoise
        mean += gnoise
        std = std + self.eps

        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        # pooled_stats = pooled_stats.unsqueeze(1)

        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.
        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise


class TCN_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation, residual=False):
        super(TCN_Conv1d, self).__init__()
        self.kernel = kernel
        self.dilation = dilation
        self.conv = weight_norm(
                        nn.Conv1d(in_channels, out_channels, 
                        kernel_size=kernel, dilation=dilation)
                    )
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
        out = self.bn(out)
        out = F.relu(out)
        if self.residual:
            if self.downsample:
                x = self.downsample(x)
            x = self.bn2(x)
            out = F.relu(x + out)
        return out


class TCN(nn.Module):
    def __init__(self, channels, kernels, dilations, num_classes, residual=False, con_dim=64):
        super(TCN, self).__init__()
        self.num_layers = len(channels)
        self.kernels = kernels
        self.dilations = dilations
        self.con_dim = con_dim
        channels.append(FILTERBANK_LEN)
        convs = []
        for i in range(self.num_layers):
            convs.append(
                TCN_Conv1d(channels[i-1], channels[i], kernels[i], dilations[i], residual)
            )
        self.convs = nn.Sequential(*convs)
        self.dense = nn.Linear(channels[-2], num_classes)
        self.pooling = StatisticsPooling()
        self.con_dense = nn.Linear(2*channels[-2], con_dim)
        self.adv_dense = nn.Linear(2*channels[-2], con_dim)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = x.permute(0, 2, 1)
        # print('conv:{}'.format(x.shape))
        statistic_x = self.pooling(x)
        con_x = self.con_dense(statistic_x)
        adv_x = self.adv_dense(statistic_x)
        # print('con_x:{}'.format(con_x.shape))

        x = self.dense(x)
        x = F.softmax(x, -1)
        return x, con_x, adv_x


    def get_param_num(self):
        return sum(p.numel() for p in self.parameters())



def test():
    channels = [64] * 5
    kernels = [8] * 5
    dilations = [1, 2, 4, 4, 8]
    num_classes = 5
    net = TCN(channels, kernels, dilations, num_classes)
    batch_data = torch.rand(1024,16000,23)
    out = net(batch_data)
    for name, param in net.named_parameters():
        print(name, ":", param.size())
    print("num of parameters:", sum(p.numel() for p in net.parameters()))
    print("num of trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    
if __name__ == "__main__":
    test()
    '''lp = LineProfiler()
    lp.add_function(test)
    lp.add_function(TCN)
    lp_wrapper = lp(test)
    lp_wrapper()
    lp.print_stats()'''
        
        